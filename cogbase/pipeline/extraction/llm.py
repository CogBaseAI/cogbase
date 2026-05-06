"""LLM-backed extractor — general-purpose structured extraction from documents.

``LLMExtractor`` calls an OpenAI-compatible LLM to extract structured data
from a document according to a caller-supplied Pydantic model.  It automatically
adds a ``doc_id`` identity field to each extracted record.
"""

from __future__ import annotations

import logging
import time
from typing import Type

from pydantic import BaseModel, ValidationError, create_model

from cogbase.core.models import Document
from cogbase.llms import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
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

_DEFAULT_LIST_SYSTEM_PROMPT_PREFIX = (
    "Extract all matching items from the document provided by the user.\n\n"
    "Rules:\n"
    "- Copy all text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the document.\n"
    "- Use null for any field not found.\n"
    "- Return an empty array when no items are found.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
)


def _build_record_model(extraction_model: Type[BaseModel]) -> Type[BaseModel]:
    """Extend *extraction_model* with a ``doc_id`` identity field."""
    return create_model(
        "_RecordModel",
        doc_id=(str, ...),
        __base__=extraction_model,
    )


def _build_list_record_model(extraction_model: Type[BaseModel], item_id_field: str) -> Type[BaseModel]:
    """Extend *extraction_model* with ``doc_id`` and a configurable item-id identity field."""
    return create_model(
        "_ListItemRecordModel",
        doc_id=(str, ...),
        **{item_id_field: (str, ...)},
        __base__=extraction_model,
    )


class LLMExtractor(ExtractorBase):
    """Extracts structured records from documents using an LLM.

    In single-record mode (default) each call to ``extract`` produces one record
    per document.  In list mode (``extract_as_list=True``) the LLM returns a JSON
    object whose ``list_field`` key holds an array of items; each item becomes a
    separate row with an auto-generated item id (``"{doc_id}__{i:04d}"``).

    Args:
        llm:               LLM backend.
        extraction_model:  Pydantic ``BaseModel`` describing the fields the LLM
                           should extract.  In list mode this is the *item* type
                           (e.g. ``ContractClauseExtraction``), not a wrapper model.
                           Identity fields (``doc_id``, item id) must NOT appear here;
                           they are injected automatically.
        extract_as_list:   When ``True`` the LLM is asked to return a JSON object
                           with a single array key; each element becomes one row.
        list_field:        The JSON key that wraps the array in list mode.
        item_id_field:     Name of the per-item primary-key field injected into each
                           extracted row in list mode.
        system_prompt:     Full system prompt.  When ``None`` a generic prompt is
                           built from *extraction_model*'s JSON schema.
        max_retries:       Retries on unparseable JSON.
    """

    def __init__(
        self,
        llm: LLMBase,
        extraction_model: Type[BaseModel],
        *,
        extract_as_list: bool = False,
        list_field: str = "items",
        item_id_field: str = "item_id",
        system_prompt: str | None = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._llm = llm
        self._extraction_model = extraction_model
        self._extract_as_list = extract_as_list
        self._list_field = list_field
        self._item_id_field = item_id_field

        if extract_as_list:
            self._record_model = _build_list_record_model(extraction_model, item_id_field)
            self._wrapper_model: Type[BaseModel] = create_model(
                "_WrapperModel",
                **{list_field: (list[extraction_model], ...)},  # type: ignore[call-overload]
            )
            self._system_prompt = system_prompt or (
                _DEFAULT_LIST_SYSTEM_PROMPT_PREFIX
                + f'Return a JSON object with a single key "{list_field}" whose value is an array.\n'
                + "Each element must have these fields:\n\n"
                + cls_json_schema_for_llm(extraction_model)
            )
        else:
            self._record_model = _build_record_model(extraction_model)
            self._system_prompt = system_prompt or (
                _DEFAULT_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(extraction_model)
            )

    async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
        """Single LLM call; returns ``None`` when the response is unparseable."""
        t0 = time.monotonic()
        result = await self._llm.complete(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": doc.text},
            ],
        )

        content = result.get("content")
        if not content:
            logger.error(
                "extract returns no content, doc_id=%s, elapsed=%.3fs, system_prompt=%s, result=%s",
                doc.doc_id,
                time.monotonic() - t0,
                self._system_prompt,
                result,
            )
            return None

        logger.info(
            "llm.complete doc_id=%s elapsed=%.3fs, content=%s",
            doc.doc_id,
            time.monotonic() - t0,
            content[:50],
        )

        if self._extract_as_list:
            return self._parse_list(doc.doc_id, content, result)
        return self._parse_single(doc.doc_id, content, result)

    def _parse_single(self, doc_id: str, content: str, raw_result: dict) -> list[BaseModel] | None:
        try:
            extraction = self._extraction_model.model_validate_json(content)
        except (ValidationError, ValueError):
            logger.exception(
                "llm_extractor.parse_failed doc_id=%s, system_prompt=%s, result=%s",
                doc_id,
                self._system_prompt,
                raw_result,
            )
            return None
        return [self._record_model(doc_id=doc_id, **extraction.model_dump())]

    def _parse_list(self, doc_id: str, content: str, raw_result: dict) -> list[BaseModel] | None:
        try:
            wrapper = self._wrapper_model.model_validate_json(content)
        except (ValidationError, ValueError):
            logger.exception(
                "llm_extractor.parse_failed doc_id=%s, system_prompt=%s, result=%s",
                doc_id,
                self._system_prompt,
                raw_result,
            )
            return None
        items = getattr(wrapper, self._list_field)
        return [
            self._record_model(
                doc_id=doc_id,
                **{self._item_id_field: f"{doc_id}__{i:04d}"},
                **item.model_dump(),
            )
            for i, item in enumerate(items)
        ]
