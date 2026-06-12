"""LLM-backed extractor — general-purpose structured extraction from documents."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import jsonschema

from cogbase.config.config import ExtractorConfig, RecordMode
from cogbase.core.models import Document
from cogbase.llms import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT_PREFIX = (
    "Extract structured information from the document provided by the user.\n\n"
    "Rules:\n"
    "- Copy all text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the document.\n"
    "- Use null for any field not found in the document.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object matching this JSON Schema:\n\n"
)

_DEFAULT_LIST_SYSTEM_PROMPT_PREFIX = (
    "Extract all matching items from the document provided by the user.\n\n"
    "Rules:\n"
    "- Copy all text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the document.\n"
    "- Use null for any field not found.\n"
    "- Return an empty array when no items are found.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object matching this JSON Schema:\n\n"
)


class LLMExtractor(ExtractorBase):
    """Extracts structured records from documents using an LLM.

    In single-record mode (``record_mode=\"one\"``, default) each call to ``extract``
    produces one record per document.  In list mode (``record_mode=\"many\"``) the LLM
    returns a JSON object whose ``response_field`` key holds an array; each item
    becomes a separate row.

    Args:
        llm:               LLM backend.
        extraction_schema: JSON Schema dict describing what the LLM should return.
                           Must not include injected identity fields (``doc_id``, item id).
        config:            ExtractorConfig describing record mode, prompt, and
                           injected identity fields.
        record_schema:     JSON Schema dict for the final stored record (includes injected
                           fields such as ``doc_id`` and optional item id).
        max_retries:       Retries on unparseable JSON.
        app_id:            Stable internal id of the owning application, included
                           in log lines for attribution.
    """

    def __init__(
        self,
        llm: LLMBase,
        extraction_schema: dict,
        *,
        config: ExtractorConfig,
        record_schema: dict,
        max_retries: int = 2,
        app_id: str = "",
    ) -> None:
        super().__init__(max_retries=max_retries, app_id=app_id)
        self._llm = llm
        self._extraction_schema = extraction_schema
        self._record_schema = record_schema

        self._validate_schema_contract(config)

        self._record_mode = config.record_mode
        self._response_field = config.response_field
        self._injected_fields = self._build_injected_fields_from_config(config)
        self._system_prompt = self._build_system_prompt_from_config(config)

    def _build_system_prompt_from_config(self, config: ExtractorConfig) -> str:
        schema_hint = json.dumps(self._extraction_schema, indent=2)
        if config.record_mode == RecordMode.MANY:
            base = (
                (config.prompt or "")
                if config.prompt
                else _DEFAULT_LIST_SYSTEM_PROMPT_PREFIX
            )
            return (
                base
                + f'\nReturn a JSON object with a single key \"{config.response_field}\" whose value is an array.\n'
                + "Each element must validate against this JSON Schema:\n\n"
                + schema_hint
            )
        if config.prompt:
            return config.prompt + "\n\nReturn a single JSON object matching this JSON Schema:\n\n" + schema_hint
        return _DEFAULT_SYSTEM_PROMPT_PREFIX + schema_hint

    def _validate_schema_contract(self, config: ExtractorConfig) -> None:
        extraction_fields = set(self._extraction_schema.get("properties", {}).keys())
        if "doc_id" in extraction_fields:
            raise ValueError(
                "extraction_schema must not include 'doc_id' (it is injected by the pipeline)"
            )
        if config.record_mode == RecordMode.MANY and config.id_field in extraction_fields:
            raise ValueError(
                f"extraction_schema must not include '{config.id_field}' (it is injected by the pipeline)"
            )

        record_fields = set(self._record_schema.get("properties", {}).keys())
        if "doc_id" not in record_fields:
            raise ValueError("record schema must include 'doc_id'")
        if config.record_mode == RecordMode.MANY and config.id_field not in record_fields:
            raise ValueError(
                f"record schema must include '{config.id_field}' (id_field) for record_mode=many"
            )

    @staticmethod
    def _build_injected_fields_from_config(config: ExtractorConfig) -> dict[str, Callable]:
        injected_fields: dict[str, Callable] = {
            "doc_id": lambda doc, item, index: doc.doc_id,
        }
        if config.record_mode == RecordMode.MANY:
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

    async def _extract_once(self, doc: Document) -> list[dict] | None:
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
                "llm_extractor.no_content app_id=%s doc_id=%s elapsed=%.3fs system_prompt=%s result=%s",
                self._app_id,
                doc.doc_id,
                time.monotonic() - t0,
                self._system_prompt,
                result,
            )
            return None

        logger.info(
            "llm_extractor.complete app_id=%s doc_id=%s elapsed=%.3fs content=%s",
            self._app_id,
            doc.doc_id,
            time.monotonic() - t0,
            content[:50],
        )

        if self._record_mode == RecordMode.MANY:
            return self._parse_list(doc, content, result)
        return self._parse_single(doc, content, result)

    def _try_unwrap_single_item_list(self, content: str) -> dict | None:
        """Return the first element if the LLM returned a one-item list instead of an object."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if len(lists) != 1:
                return None
            items = lists[0]
        else:
            return None
        if len(items) != 1:
            return None
        try:
            jsonschema.validate(instance=items[0], schema=self._extraction_schema)
            return items[0]
        except (jsonschema.ValidationError, Exception):
            return None

    def _parse_single(self, doc: Document, content: str, raw_result: dict) -> list[dict] | None:
        try:
            parsed = json.loads(content)
            jsonschema.validate(instance=parsed, schema=self._extraction_schema)
            extraction = parsed
        except (json.JSONDecodeError, jsonschema.ValidationError):
            extraction = self._try_unwrap_single_item_list(content)
            if extraction is None:
                logger.exception(
                    "llm_extractor.parse_failed app_id=%s doc_id=%s system_prompt=%s result=%s",
                    self._app_id,
                    doc.doc_id,
                    self._system_prompt,
                    raw_result,
                )
                return None
            logger.info(
                "llm_extractor.unwrapped_single_item_list app_id=%s doc_id=%s",
                self._app_id, doc.doc_id,
            )
        injected = {k: fn(doc, extraction, 0) for k, fn in self._injected_fields.items()}
        return [{**extraction, **injected}]

    def _parse_list(self, doc: Document, content: str, raw_result: dict) -> list[dict] | None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.exception(
                "llm_extractor.parse_failed app_id=%s doc_id=%s system_prompt=%s result=%s",
                self._app_id,
                doc.doc_id,
                self._system_prompt,
                raw_result,
            )
            return None
        if not isinstance(parsed, dict) or self._response_field not in parsed:
            logger.error(
                "llm_extractor.parse_failed app_id=%s doc_id=%s missing response_field=%s result=%s",
                self._app_id,
                doc.doc_id,
                self._response_field,
                raw_result,
            )
            return None
        items: Any = parsed[self._response_field]
        if not isinstance(items, list):
            logger.error(
                "llm_extractor.parse_failed app_id=%s doc_id=%s response_field not a list, result=%s",
                self._app_id,
                doc.doc_id,
                raw_result,
            )
            return None
        records = []
        for index, item in enumerate(items):
            injected = {k: fn(doc, item, index) for k, fn in self._injected_fields.items()}
            records.append({**item, **injected})
        return records
