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
from typing import Any, Type

from pydantic import BaseModel, ValidationError, create_model

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores.schema import CollectionSchema
from cogbase.stores.schema_util import cls_generate_schema, cls_json_schema_for_llm
from packs.legal.contract_analyst.schema import (
    CONTRACTS_COLLECTION,
    CONTRACTS_SCHEMA,
    ContractExtraction,
    ContractRecord,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PREFIX = (
    "You are a legal contract analyst.  Extract structured information from the\n"
    "contract provided by the user.\n\n"
    "Rules:\n"
    "- Copy all clause text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the contract.\n"
    "- Use null for any field not found in the contract.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object with these fields:\n\n"
)


def _build_record_model(extraction_model: Type[BaseModel]) -> Type[BaseModel]:
    """Extend *extraction_model* with ``contract_id`` and ``doc_id`` identity fields."""
    return create_model(
        "DynamicContractRecord",
        contract_id=(str, ...),
        doc_id=(str, ...),
        __base__=extraction_model,
    )


def _build_collection_schema(record_model: Type[BaseModel]) -> CollectionSchema:
    """Derive a ``CollectionSchema`` from *record_model*'s fields."""
    return CollectionSchema(
        name=CONTRACTS_COLLECTION,
        primary_fields=["contract_id"],
        fields=cls_generate_schema(record_model),
    )


class ContractExtractor(ExtractorBase):
    """Extracts a structured summary from a contract document using an LLM.

    Each call to ``extract`` produces one record for the document — covering
    contract basics, common clause text, and flexible JSON fields for terms
    that vary by contract type — or ``None`` when the text is blank or the LLM
    returns unparseable output after all retries.

    Args:
        client:           Async OpenAI-compatible client
                          (``openai.AsyncOpenAI``, Anthropic compat endpoint, etc.).
        model:            Model name (e.g. ``"claude-sonnet-4-6"``).
        extraction_model: Pydantic ``BaseModel`` class describing the fields to
                          extract.  When ``None`` the built-in
                          ``ContractExtraction`` schema is used.  Pass a class
                          built by ``build_model_from_json_schema`` to use a
                          custom schema without touching pack code.
        max_tokens:       Maximum tokens for the LLM response.
        max_retries:      Passed to ``ExtractorBase``; retries on unparseable JSON.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        extraction_model: Type[BaseModel] | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

        if extraction_model is None:
            self._extraction_model: Type[BaseModel] = ContractExtraction
            self._record_model: Type[BaseModel] = ContractRecord
            self._schema = CONTRACTS_SCHEMA
        else:
            self._extraction_model = extraction_model
            self._record_model = _build_record_model(extraction_model)
            self._schema = _build_collection_schema(self._record_model)

        self._system_prompt = (
            _SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(self._extraction_model)
        )

    @property
    def collection(self) -> str:
        return CONTRACTS_COLLECTION

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    async def _extract_once(self, doc: Document) -> BaseModel | None:
        """Single LLM call; returns ``None`` when the response is unparseable."""
        response = await self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": doc.text},
            ],
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw, doc.doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str, doc_id: str) -> BaseModel | None:
        """Parse the LLM JSON response into a record model instance."""
        try:
            extraction = self._extraction_model.model_validate_json(raw)
        except (ValidationError, ValueError):
            logger.exception("contract_extractor.parse_failed doc_id=%s", doc_id)
            return None

        contract_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
        return self._record_model(
            contract_id=contract_id, doc_id=doc_id, **extraction.model_dump()
        )
