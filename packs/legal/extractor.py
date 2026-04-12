"""LLM-backed clause extractor for the legal contract review pack.

Extracts typed contract clauses from document text using an OpenAI-compatible
async client.  Each clause is returned as a ``Clause`` Pydantic record and
written to the ``clauses`` structured store collection.

Typical usage::

    import openai
    from packs.legal.extractor import ClauseExtractor

    client = openai.AsyncOpenAI(api_key="...")
    extractor = ClauseExtractor(client, model="claude-sonnet-4-6")

    records = await extractor.extract(contract_text, doc_id="contract-001")
    # returns list[Clause]
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel

from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores.schema import CollectionSchema
from packs.legal.schema import CLAUSES_COLLECTION, CLAUSES_SCHEMA, Clause

_SYSTEM_PROMPT = """\
You are a legal contract analyst.  Extract every significant clause from the
contract text provided by the user.

Return a JSON array where each element has exactly these fields:
  - "type":       one of: payment, termination, liability, notice,
                  governing_law, confidentiality, indemnification,
                  dispute_resolution, other
  - "text":       the verbatim clause text, copied exactly from the contract
  - "confidence": your confidence in the extraction, a float in [0.0, 1.0]

Rules:
- Include only clauses you can quote verbatim from the contract.
- Do not paraphrase or invent text.
- If no clauses are found, return an empty array: []
- Return ONLY the JSON array — no explanation, no markdown fences.
"""


class ClauseExtractor(ExtractorBase):
    """Extracts typed contract clauses using an LLM.

    Args:
        client:     Async OpenAI-compatible client
                    (``openai.AsyncOpenAI``, Anthropic compat endpoint, etc.).
        model:      Model name (e.g. ``"claude-sonnet-4-6"``).
        max_tokens: Maximum tokens for the LLM response.

    The extractor sends the full document text to the LLM in a single call and
    parses the returned JSON array into ``Clause`` records.  For very long
    contracts, chunk the text upstream and call ``extract`` per chunk.
    """

    def __init__(self, client: Any, model: str, max_tokens: int = 4096) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def collection(self) -> str:
        return CLAUSES_COLLECTION

    @property
    def schema(self) -> CollectionSchema:
        return CLAUSES_SCHEMA

    async def extract(self, text: str, doc_id: str) -> list[BaseModel]:
        """Extract clauses from *text* and return them as ``Clause`` records.

        Args:
            text:   Full or chunked contract text.
            doc_id: Stable identifier for the source document.

        Returns:
            List of ``Clause`` instances.  Empty list when no clauses are found
            or the LLM returns unparseable output.
        """
        if not text.strip():
            return []

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw, doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str, doc_id: str) -> list[Clause]:
        """Parse the LLM JSON response into ``Clause`` records."""
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            return []

        clauses: list[Clause] = []
        type_counts: dict[str, int] = {}

        for item in items:
            if not isinstance(item, dict):
                continue
            clause_type = str(item.get("type", "other")).lower()
            idx = type_counts.get(clause_type, 0)
            type_counts[clause_type] = idx + 1

            clause_id = f"{doc_id}_{clause_type}_{idx}_{uuid.uuid4().hex[:8]}"

            try:
                confidence = float(item.get("confidence", 0.8))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.8

            clauses.append(
                Clause(
                    clause_id=clause_id,
                    doc_id=doc_id,
                    type=clause_type,
                    text=str(item.get("text", "")),
                    page=item.get("page"),
                    confidence=confidence,
                )
            )

        return clauses
