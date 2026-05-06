"""LLM-backed extractor — general-purpose structured extraction from documents.

``LLMExtractor`` calls an OpenAI-compatible LLM to extract structured data
from a document according to a caller-supplied Pydantic model.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Literal, Type

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
    return create_model(
        "_RecordModel",
        doc_id=(str, ...),
        __base__=extraction_model,
    )


def _build_list_record_model(extraction_model: Type[BaseModel], item_id_field: str) -> Type[BaseModel]:
    return create_model(
        "_ListItemRecordModel",
        doc_id=(str, ...),
        **{item_id_field: (str, ...)},
        __base__=extraction_model,
    )


class LLMExtractor(ExtractorBase):
    """Extracts structured records from documents using an LLM.

    In single-record mode (``record_mode="one"``, default) each call to ``extract``
    produces one record per document.  In list mode (``record_mode="many"``) the LLM
    returns a JSON object whose ``response_field`` key holds an array; each item
    becomes a separate row.

    Args:
        llm:              LLM backend.
        extraction_model: Pydantic model describing what the LLM should return.
                          Injected identity fields (``doc_id``, item id) must NOT
                          appear here; supply them via *injected_fields*.
        record_model:     Pydantic model used for final validation and storage.
        record_mode:      ``"one"`` (default) or ``"many"``.
        response_field:   JSON key that wraps the array in many mode.
        injected_fields:  Mapping of field name to ``(doc, item, index) → value``
                          callables.  Defaults to ``doc_id``-only injection.
        system_prompt:    Full system prompt; auto-built from *extraction_model* when
                          ``None``.
        max_retries:      Retries on unparseable JSON.
    """

    def __init__(
        self,
        llm: LLMBase,
        extraction_model: Type[BaseModel],
        *,
        record_model: Type[BaseModel],
        record_mode: Literal["one", "many"] = "one",
        response_field: str = "items",
        injected_fields: dict[str, Callable] | None = None,
        system_prompt: str | None = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._llm = llm
        self._extraction_model = extraction_model
        self._record_mode = record_mode
        self._response_field = response_field
        self._record_model = record_model
        self._injected_fields: dict[str, Callable] = (
            injected_fields
            if injected_fields is not None
            else {"doc_id": lambda doc, item, index: doc.doc_id}
        )

        if record_mode == "many":
            self._wrapper_model: Type[BaseModel] = create_model(
                "_WrapperModel",
                **{response_field: (list[extraction_model], ...)},  # type: ignore[call-overload]
            )
            self._system_prompt = system_prompt or (
                _DEFAULT_LIST_SYSTEM_PROMPT_PREFIX
                + f'Return a JSON object with a single key "{response_field}" whose value is an array.\n'
                + "Each element must have these fields:\n\n"
                + cls_json_schema_for_llm(extraction_model)
            )
        else:
            self._system_prompt = system_prompt or (
                _DEFAULT_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(extraction_model)
            )

    async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
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

        if self._record_mode == "many":
            return self._parse_list(doc, content, result)
        return self._parse_single(doc, content, result)

    def _parse_single(self, doc: Document, content: str, raw_result: dict) -> list[BaseModel] | None:
        try:
            extraction = self._extraction_model.model_validate_json(content)
        except (ValidationError, ValueError):
            logger.exception(
                "llm_extractor.parse_failed doc_id=%s, system_prompt=%s, result=%s",
                doc.doc_id,
                self._system_prompt,
                raw_result,
            )
            return None
        injected = {k: fn(doc, extraction, 0) for k, fn in self._injected_fields.items()}
        try:
            record = self._record_model.model_validate({**extraction.model_dump(), **injected})
        except (ValidationError, ValueError):
            logger.exception(
                "llm_extractor.record_validation_failed doc_id=%s",
                doc.doc_id,
            )
            return None
        return [record]

    def _parse_list(self, doc: Document, content: str, raw_result: dict) -> list[BaseModel] | None:
        try:
            wrapper = self._wrapper_model.model_validate_json(content)
        except (ValidationError, ValueError):
            logger.exception(
                "llm_extractor.parse_failed doc_id=%s, system_prompt=%s, result=%s",
                doc.doc_id,
                self._system_prompt,
                raw_result,
            )
            return None
        items = getattr(wrapper, self._response_field)
        records = []
        for index, item in enumerate(items):
            injected = {k: fn(doc, item, index) for k, fn in self._injected_fields.items()}
            try:
                record = self._record_model.model_validate({**item.model_dump(), **injected})
            except (ValidationError, ValueError):
                logger.exception(
                    "llm_extractor.record_validation_failed doc_id=%s index=%d",
                    doc.doc_id,
                    index,
                )
                return None
            records.append(record)
        return records
