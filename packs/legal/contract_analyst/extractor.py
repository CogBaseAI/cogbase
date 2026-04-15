"""LLM-backed contract extractor for the legal contract review pack.

Extracts a structured summary from a contract document using an OpenAI-compatible
async client.  Each document produces exactly one ``ContractRecord`` written to
the ``contracts`` structured store collection.

Typical usage::

    import openai
    from packs.legal.contract_analyst.extractor import ContractExtractor

    client = openai.AsyncOpenAI(api_key="...")
    extractor = ContractExtractor(client, model="claude-sonnet-4-6")

    record = await extractor.extract(Document(doc_id="contract-001", text=contract_text))
    # returns a ContractRecord, or None when the text is blank / unparseable
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, ValidationError

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores.schema import CollectionSchema
from cogbase.stores.schema_util import cls_json_schema_for_llm
from packs.legal.contract_analyst.schema import CONTRACTS_COLLECTION, CONTRACTS_SCHEMA, ContractExtraction, ContractRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a legal contract analyst.  Extract structured information from the\n"
    "contract provided by the user.\n\n"
    "Rules:\n"
    "- Copy all clause text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the contract.\n"
    "- Use null for any field not found in the contract.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object with these fields:\n\n"
    + cls_json_schema_for_llm(ContractExtraction)
)


class ContractExtractor(ExtractorBase):
    """Extracts a structured summary from a contract document using an LLM.

    Each call to ``extract`` produces one ``ContractRecord`` for the document —
    covering contract basics, common clause text, and two flexible JSON fields
    for terms and conditions that vary by contract type — or ``None`` when the
    text is blank or the LLM returns unparseable output after all retries.

    Args:
        client:      Async OpenAI-compatible client
                     (``openai.AsyncOpenAI``, Anthropic compat endpoint, etc.).
        model:       Model name (e.g. ``"claude-sonnet-4-6"``).
        max_tokens:  Maximum tokens for the LLM response.
        max_retries: Passed to ``ExtractorBase``; retries on unparseable JSON.
    """

    def __init__(self, client: Any, model: str, max_tokens: int = 4096, max_retries: int = 2) -> None:
        super().__init__(max_retries=max_retries)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def collection(self) -> str:
        return CONTRACTS_COLLECTION

    @property
    def schema(self) -> CollectionSchema:
        return CONTRACTS_SCHEMA

    async def _extract_once(self, doc: Document) -> ContractRecord | None:
        """Single LLM call; returns ``None`` when the response is unparseable."""
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": doc.text},
            ],
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw, doc.doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str, doc_id: str) -> ContractRecord | None:
        """Parse the LLM JSON response into a ``ContractRecord``."""
        try:
            extraction = ContractExtraction.model_validate_json(raw)
        except (ValidationError, ValueError):
            logger.exception("contract_extractor.parse_failed doc_id=%s", doc_id)
            return None

        contract_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
        return ContractRecord(contract_id=contract_id, doc_id=doc_id, **extraction.model_dump())
