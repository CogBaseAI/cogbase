"""LLM-backed extractor — general-purpose structured extraction from documents.

``LLMExtractor`` calls an OpenAI-compatible LLM to extract structured data
from a document according to a caller-supplied Pydantic model.  It automatically
adds a ``doc_id`` identity field to each extracted record.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, ValidationError, create_model

from cogbase.core.models import Document
from cogbase.llms import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores import CollectionSchema
from cogbase.core.basemodel_to_schema import cls_generate_schema, cls_json_schema_for_llm

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


def _build_record_model(extraction_model: Type[BaseModel]) -> Type[BaseModel]:
    """Extend *extraction_model* with a ``doc_id`` identity field."""
    return create_model(
        "_RecordModel",
        doc_id=(str, ...),
        __base__=extraction_model,
    )


def _build_collection_schema(
    record_model: Type[BaseModel],
    collection_name: str,
    description: str,
) -> CollectionSchema:
    """Derive a ``CollectionSchema`` from *record_model*'s fields."""
    return CollectionSchema(
        name=collection_name,
        description=description,
        primary_fields=["doc_id"],
        fields=cls_generate_schema(record_model),
    )


class LLMExtractor(ExtractorBase):
    """Extracts structured records from documents using an LLM.

    Each call to ``extract`` produces one record for the document or ``None``
    when the text is blank or the LLM returns unparseable output after all
    retries.

    The returned record type extends *extraction_model* with a ``doc_id``
    identity field (the document identifier passed in).

    Args:
        llm:                    LLM backend.
        extraction_model:       Pydantic ``BaseModel`` class describing the fields to
                                extract.  Its field descriptions are included in the
                                LLM prompt.
        collection_name:        Name of the structured store collection to write to.
        collection_description: Short description shown to the LLM in the retrieval
                                prompt so it understands what this collection holds
                                and when to query it.
        system_prompt:          Full system prompt for the LLM.  When ``None`` a
                                generic prompt is built from *extraction_model*'s
                                JSON schema.
        max_tokens:             Maximum tokens for the LLM response.
        max_retries:            Retries on unparseable JSON (passed to
                                ``ExtractorBase``).
    """

    def __init__(
        self,
        llm: LLMBase,
        extraction_model: Type[BaseModel],
        collection_name: str,
        collection_description: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._llm = llm
        self._max_tokens = max_tokens
        self._collection_name = collection_name
        self._extraction_model = extraction_model
        self._record_model = _build_record_model(extraction_model)
        self._schema = _build_collection_schema(self._record_model, collection_name, collection_description)
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
        raw = await self._llm.complete(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": doc.text},
            ],
            max_tokens=self._max_tokens,
        )
        return self._parse(raw, doc.doc_id)

    def _parse(self, raw: str, doc_id: str) -> BaseModel | None:
        """Parse the LLM JSON response into a record model instance."""
        try:
            extraction = self._extraction_model.model_validate_json(raw)
        except (ValidationError, ValueError):
            logger.exception("llm_extractor.parse_failed doc_id=%s", doc_id)
            return None

        return self._record_model(
            doc_id=doc_id,
            **extraction.model_dump(),
        )
