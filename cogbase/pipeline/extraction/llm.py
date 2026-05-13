"""LLM-backed extractor — general-purpose structured extraction from documents.

``LLMExtractor`` calls an OpenAI-compatible LLM to extract structured data
from a document according to a caller-supplied Pydantic model.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Type

from pydantic import BaseModel, ValidationError, create_model

from cogbase.config.config import ExtractorConfig, RecordMode
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
                          appear here; supply them via *config*.
        record_model:     Pydantic model used for final validation and storage.
        config:           ExtractorConfig describing record mode, prompt, and
                          injected identity fields.
        max_retries:      Retries on unparseable JSON.
    """

    def __init__(
        self,
        llm: LLMBase,
        extraction_model: Type[BaseModel],
        *,
        config: ExtractorConfig,
        record_model: Type[BaseModel],
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._llm = llm
        self._extraction_model = extraction_model
        self._record_model = record_model

        self._validate_schema_contract(config)

        self._record_mode = config.record_mode
        self._response_field = config.response_field
        self._injected_fields = self._build_injected_fields_from_config(config)

        if config.record_mode == RecordMode.MANY:
            self._wrapper_model: Type[BaseModel] = create_model(
                "_WrapperModel",
                **{config.response_field: (list[extraction_model], ...)},  # type: ignore[call-overload]
            )
            self._system_prompt = self._build_system_prompt_from_config(config)
        else:
            self._system_prompt = self._build_system_prompt_from_config(config)

    def _build_system_prompt_from_config(self, config: ExtractorConfig) -> str:
        if config.record_mode == RecordMode.MANY:
            base = (
                (config.prompt or "")
                if config.prompt
                else _DEFAULT_LIST_SYSTEM_PROMPT_PREFIX
            )
            return (
                base
                + f'\nReturn a JSON object with a single key "{config.response_field}" whose value is an array.\n'
                + "Each element must have these fields:\n\n"
                + cls_json_schema_for_llm(self._extraction_model)
            )
        base = (config.prompt or "") if config.prompt else _DEFAULT_SYSTEM_PROMPT_PREFIX
        return (
            base
            + cls_json_schema_for_llm(self._extraction_model)
        )

    def _validate_schema_contract(self, config: ExtractorConfig) -> None:
        extraction_fields = set(self._extraction_model.model_fields)
        if "doc_id" in extraction_fields:
            raise ValueError(
                "extraction_schema must not include 'doc_id' (it is injected by the pipeline)"
            )
        if config.record_mode == RecordMode.MANY and config.id_field and config.id_field in extraction_fields:
            raise ValueError(
                f"extraction_schema must not include '{config.id_field}' (it is injected by the pipeline)"
            )

        record_fields = set(self._record_model.model_fields)
        if "doc_id" not in record_fields:
            raise ValueError("record schema must include 'doc_id'")
        if config.record_mode == RecordMode.MANY and config.id_field and config.id_field not in record_fields:
            raise ValueError(
                f"record schema must include '{config.id_field}' (id_field) for record_mode=many"
            )

    @staticmethod
    def _build_injected_fields_from_config(config: ExtractorConfig) -> dict[str, Callable]:
        injected_fields: dict[str, Callable] = {
            "doc_id": lambda doc, item, index: doc.doc_id,
        }
        if config.record_mode == RecordMode.MANY and config.id_field:
            if config.id_template:
                template = config.id_template
                injected_fields[config.id_field] = (
                    lambda doc, item, index, t=template: t.format(doc_id=doc.doc_id, index=index)
                )
            else:
                injected_fields[config.id_field] = (
                    lambda doc, item, index: f"{doc.doc_id}__{index:04d}"
                )
        return injected_fields

    async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
        t0 = time.monotonic()
        result = await self._llm.complete(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": doc.text},
            ],
            model="mini",
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

        if self._record_mode == RecordMode.MANY:
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
