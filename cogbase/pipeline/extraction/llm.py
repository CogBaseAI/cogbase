"""LLM-backed extractor — general-purpose structured extraction from documents.

``LLMExtractor`` calls an OpenAI-compatible LLM to extract structured data
from a document according to a caller-supplied Pydantic model.  It automatically
adds an identity field (``record_id`` by default, or a custom name via
*id_field*) and a ``doc_id`` field to each extracted record.

Typical usage::

    import openai
    from cogbase.pipeline.extraction.llm import LLMExtractor
    from examples.contract_analyst_demo.schema import ContractExtraction, CONTRACTS_COLLECTION

    client = openai.AsyncOpenAI(api_key="...")
    extractor = LLMExtractor(
        client=client,
        model="gpt-4o-mini",
        extraction_model=ContractExtraction,
        collection_name=CONTRACTS_COLLECTION,
        id_field="contract_id",
    )
    record = await extractor.extract(doc)
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

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT_PREFIX = (
    "Extract structured information from the document provided by the user.\n\n"
    "Rules:\n"
    "- Copy all text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the document.\n"
    "- Use null for any field not found in the document.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object with these fields:\n\n"
)


def _build_record_model(extraction_model: Type[BaseModel], id_field: str) -> Type[BaseModel]:
    """Extend *extraction_model* with *id_field* and ``doc_id`` identity fields."""
    return create_model(
        "_RecordModel",
        **{id_field: (str, ...), "doc_id": (str, ...)},
        __base__=extraction_model,
    )


def _build_collection_schema(
    record_model: Type[BaseModel],
    collection_name: str,
    id_field: str,
) -> CollectionSchema:
    """Derive a ``CollectionSchema`` from *record_model*'s fields."""
    return CollectionSchema(
        name=collection_name,
        primary_fields=[id_field],
        fields=cls_generate_schema(record_model),
    )


class LLMExtractor(ExtractorBase):
    """Extracts structured records from documents using an LLM.

    Each call to ``extract`` produces one record for the document or ``None``
    when the text is blank or the LLM returns unparseable output after all
    retries.

    The returned record type extends *extraction_model* with two identity
    fields: *id_field* (a UUID-based per-record identifier) and ``doc_id``
    (the document identifier passed in).

    Args:
        client:           Async OpenAI-compatible client.
        model:            Model name (e.g. ``"gpt-4o-mini"``).
        extraction_model: Pydantic ``BaseModel`` class describing the fields to
                          extract.  Its field descriptions are included in the
                          LLM prompt.
        collection_name:  Name of the structured store collection to write to.
        id_field:         Name of the auto-generated identity field added to
                          each record.  Defaults to ``"record_id"``.
        system_prompt:    Full system prompt for the LLM.  When ``None`` a
                          generic prompt is built from *extraction_model*'s
                          JSON schema.
        max_tokens:       Maximum tokens for the LLM response.
        max_retries:      Retries on unparseable JSON (passed to
                          ``ExtractorBase``).
    """

    def __init__(
        self,
        client: Any,
        model: str,
        extraction_model: Type[BaseModel],
        collection_name: str,
        *,
        id_field: str = "record_id",
        system_prompt: str | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._collection_name = collection_name
        self._id_field = id_field
        self._extraction_model = extraction_model
        self._record_model = _build_record_model(extraction_model, id_field)
        self._schema = _build_collection_schema(self._record_model, collection_name, id_field)
        self._system_prompt = system_prompt or (
            _DEFAULT_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(extraction_model)
        )

    @property
    def collection(self) -> str:
        return self._collection_name

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

    def _parse(self, raw: str, doc_id: str) -> BaseModel | None:
        """Parse the LLM JSON response into a record model instance."""
        try:
            extraction = self._extraction_model.model_validate_json(raw)
        except (ValidationError, ValueError):
            logger.exception("llm_extractor.parse_failed doc_id=%s", doc_id)
            return None

        record_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
        return self._record_model(
            **{self._id_field: record_id, "doc_id": doc_id},
            **extraction.model_dump(),
        )
