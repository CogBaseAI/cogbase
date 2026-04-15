"""LLM-backed contract extractor for the legal contract review pack.

Extracts a structured summary from a contract document using an OpenAI-compatible
async client.  Each document produces exactly one ``ContractRecord`` written to
the ``contracts`` structured store collection.

Typical usage::

    import openai
    from packs.legal.extractor import ContractExtractor

    client = openai.AsyncOpenAI(api_key="...")
    extractor = ContractExtractor(client, model="claude-sonnet-4-6")

    records = await extractor.extract(contract_text, doc_id="contract-001")
    # returns list[ContractRecord] with exactly one element
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel

from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores.schema import CollectionSchema
from packs.legal.schema import CONTRACTS_COLLECTION, CONTRACTS_SCHEMA, ContractRecord

_SYSTEM_PROMPT = """\
You are a legal contract analyst.  Extract structured information from the
contract provided by the user.

Return a single JSON object with these fields:

Contract basics (use null if not found in the text):
  - "contract_type":  type of contract, e.g. "NDA", "SaaS subscription",
                      "employment", "vendor", "lease", "service agreement"
  - "purpose":        one sentence describing what the contract is for
  - "effective_date": start date in YYYY-MM-DD format, or null
  - "expiry_date":    end/expiry date in YYYY-MM-DD format, or null
  - "party_a":        primary party name (client or buyer), or null
  - "party_b":        counterparty name (vendor or seller), or null
  - "contract_value": total monetary value as a number (no currency symbol), or null
  - "currency":       ISO 4217 code (e.g. "USD"), or null

Common clause text — copy verbatim from the contract; use null if the clause
is absent:
  - "payment_terms":      verbatim payment terms clause
  - "termination":        verbatim termination clause
  - "liability":          verbatim limitation of liability clause
  - "governing_law":      verbatim governing law clause
  - "confidentiality":    verbatim confidentiality clause
  - "indemnification":    verbatim indemnification clause
  - "dispute_resolution": verbatim dispute resolution clause

Clause-level numeric (null if not present):
  - "notice_period_days": integer days required for termination notice, or null
  - "liability_cap":      liability cap amount as a number, or null

Flexible extraction:
  - "key_terms": array of {"term": "<name>", "description": "<brief description>"}
                 for significant defined terms, unusual provisions, or
                 contract-type-specific clauses not covered by the fields above.
                 Use [] if none.
  - "special_conditions": array of verbatim strings for conditions precedent,
                          carve-outs, custom provisions, or anything unusual.
                          Use [] if none.

Rules:
- Copy all clause text verbatim — do not paraphrase or summarise.
- Do not invent information not present in the contract.
- Return ONLY the JSON object — no explanation, no markdown fences.
"""


class ContractExtractor(ExtractorBase):
    """Extracts a structured summary from a contract document using an LLM.

    Each call to ``extract`` produces exactly one ``ContractRecord`` for the
    document — covering contract basics, common clause text, and two flexible
    JSON fields for terms and conditions that vary by contract type.

    Args:
        client:     Async OpenAI-compatible client
                    (``openai.AsyncOpenAI``, Anthropic compat endpoint, etc.).
        model:      Model name (e.g. ``"claude-sonnet-4-6"``).
        max_tokens: Maximum tokens for the LLM response.
    """

    def __init__(self, client: Any, model: str, max_tokens: int = 4096) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def collection(self) -> str:
        return CONTRACTS_COLLECTION

    @property
    def schema(self) -> CollectionSchema:
        return CONTRACTS_SCHEMA

    async def extract(self, text: str, doc_id: str) -> list[BaseModel]:
        """Extract a contract summary from *text*.

        Args:
            text:   Full contract text.
            doc_id: Stable identifier for the source document.

        Returns:
            A list containing a single ``ContractRecord``.  Returns an empty
            list when *text* is blank or the LLM returns unparseable output.
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

    def _parse(self, raw: str, doc_id: str) -> list[ContractRecord]:
        """Parse the LLM JSON response into a ``ContractRecord``."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, dict):
            return []

        contract_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"

        return [
            ContractRecord(
                contract_id=contract_id,
                doc_id=doc_id,
                # contract basics
                contract_type=_str_or_none(data.get("contract_type")),
                purpose=_str_or_none(data.get("purpose")),
                effective_date=_str_or_none(data.get("effective_date")),
                expiry_date=_str_or_none(data.get("expiry_date")),
                party_a=_str_or_none(data.get("party_a")),
                party_b=_str_or_none(data.get("party_b")),
                contract_value=_float_or_none(data.get("contract_value")),
                currency=_str_or_none(data.get("currency")),
                # common clause text
                payment_terms=_str_or_none(data.get("payment_terms")),
                termination=_str_or_none(data.get("termination")),
                liability=_str_or_none(data.get("liability")),
                governing_law=_str_or_none(data.get("governing_law")),
                confidentiality=_str_or_none(data.get("confidentiality")),
                indemnification=_str_or_none(data.get("indemnification")),
                dispute_resolution=_str_or_none(data.get("dispute_resolution")),
                # clause-level numeric
                notice_period_days=_int_or_none(data.get("notice_period_days")),
                liability_cap=_float_or_none(data.get("liability_cap")),
                # flexible extraction
                key_terms=_list_or_empty(data.get("key_terms")),
                special_conditions=_list_or_empty(data.get("special_conditions")),
            )
        ]


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _str_or_none(val: object) -> str | None:
    if val is None or val == "":
        return None
    return str(val)


def _float_or_none(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int_or_none(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _list_or_empty(val: object) -> list:
    if isinstance(val, list):
        return val
    return []
