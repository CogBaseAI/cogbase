"""Deterministic compliance-check workflow for the contract compliance demo.

Exports
-------
run_compliance_check(doc_id, *, vector_store, structured_store, embedder, llm, ...)
    Async generator. For each clause in ``contract_clauses``, retrieves matching
    rule passages from ``rule_chunks``, calls the LLM judge, saves the finding to
    ``clause_compliance_findings``, and yields the ClauseComplianceFinding.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from pydantic import ValidationError, create_model

from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm
from cogbase.core.models import Document
from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.stores import Col, StructuredStoreBase, VectorStoreBase

from examples.contract_compliance_demo.schema import (
    CLAUSE_COMPLIANCE_FINDINGS_SCHEMA,
    ClauseComplianceFinding,
)

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are a contract compliance reviewer. Determine whether a contract clause complies
with the company's internal policies, using ONLY the company policy excerpts provided.

Rules:
- Ground every finding exclusively in the provided policy excerpts.
- Do not invent policy or apply general legal knowledge not present in the excerpts.
- If the excerpts are insufficient to determine compliance, set status=needs_review.
- Every non_compliant finding MUST include at least one matched_rule_id and matched_rule_quote.
- Populate recommended_redline with revised clause language for non_compliant findings; null otherwise.
- Return ONLY valid JSON — no markdown fences, no explanation.
"""

_FINDING_SCHEMA_HINT = cls_json_schema_for_llm(ClauseComplianceFinding)


async def run_compliance_check(
    doc_id: str,
    *,
    vector_store: VectorStoreBase,
    structured_store: StructuredStoreBase,
    embedder: EmbeddingBase,
    llm: LLMBase,
    top_k: int = 5,
) -> AsyncGenerator[ClauseComplianceFinding, None]:
    """Yield one ClauseComplianceFinding per clause in ``contract_clauses`` for *doc_id*.

    For each clause the function:
    1. Embeds the clause text and searches ``rule_chunks`` for the top matching passages.
    2. Sends the clause + rule passages to the LLM judge.
    3. Validates and saves the finding to ``clause_compliance_findings``.
    4. Yields the finding.

    Clauses whose judge response cannot be parsed are skipped with a logged error.
    """
    await structured_store.create_collection(CLAUSE_COMPLIANCE_FINDINGS_SCHEMA)

    clause_records = await structured_store.query(
        "contract_clauses",
        filters=[Col("doc_id") == doc_id],
    )
    if not clause_records:
        logger.warning("run_compliance_check: no clauses found for doc_id=%s", doc_id)
        return

    for clause_dict in clause_records:
        clause_id = clause_dict.get("clause_id", "")
        clause_type = clause_dict.get("clause_type") or ""
        clause_text = clause_dict.get("text", "")

        query_text = f"{clause_type}\n{clause_text}" if clause_type else clause_text
        (embedding,) = await embedder.embed([query_text])

        rule_chunks = await vector_store.search(
            "rule_chunks",
            query_text,
            embedding,
            top_k=top_k,
        )

        rule_context = "\n\n---\n\n".join(
            f"[Rule chunk ID: {c.chunk_id}]\n{c.text}" for c in rule_chunks
        )

        user_prompt = (
            f"Contract clause:\n"
            f"  clause_id: {clause_id}\n"
            f"  clause_type: {clause_type or 'unknown'}\n"
            f"  doc_id: {doc_id}\n\n"
            f"{clause_text}\n\n"
            f"---\n\n"
            f"Company policy excerpts ({len(rule_chunks)} chunks retrieved):\n"
            f"{rule_context}\n\n"
            f"---\n\n"
            f"Return a JSON compliance finding:\n"
            f"{_FINDING_SCHEMA_HINT}"
        )

        result = await llm.complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = result["content"]
        if not content:
            logger.error(
                "run_compliance_check: no judge response for clause_id=%s", clause_id
            )
            continue

        try:
            finding = ClauseComplianceFinding.model_validate_json(content)
        except (ValidationError, ValueError):
            logger.exception(
                "run_compliance_check: parse failed clause_id=%s content=%s",
                clause_id,
                content[:300],
            )
            continue

        await structured_store.save("clause_compliance_findings", [finding])
        logger.info(
            "run_compliance_check: saved finding clause_id=%s status=%s severity=%s",
            clause_id,
            finding.status,
            finding.severity,
        )
        yield finding
