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
from cogbase.llms.compaction import estimate_tokens
from cogbase.pipeline.chunking.langchain import split_text_by_tokens
from cogbase.pipeline.extraction.base import ExtractorBase

logger = logging.getLogger(__name__)

# Fraction of the llm model's context window reserved for one window of input
# document text. The remainder leaves room for the system prompt (schema hint +
# instructions) and the model's output, which for verbatim extraction can
# approach the size of the input. Documents larger than this are split into
# multiple windows, extracted independently, then merged.
_INPUT_WINDOW_RATIO = 0.5

# Floor for the per-window token budget, so a pathologically large system prompt
# can never drive the budget to zero (which would split every document to death).
_MIN_WINDOW_TOKENS = 1_000

# Fraction of the per-window budget that consecutive windows overlap. Window
# boundaries land between clauses where the document's structure allows, but a
# clause with no structural break can still straddle a cut; the overlap re-presents
# the tail of one window at the head of the next, so such a clause appears intact
# in at least one window (duplicates are removed when the windows are merged).
_WINDOW_OVERLAP_RATIO = 0.1


def _normalize(value: Any) -> Any:
    """Canonicalise a JSON value for content comparison.

    Whitespace inside strings is collapsed (the model may re-emit the same clause
    from an overlapping window with incidental whitespace differences); containers
    recurse. No case folding, so genuinely distinct items are not merged.
    """
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _content_key(item: Any) -> str:
    """Stable, comparable key for a record's content (ignoring whitespace noise)."""
    return json.dumps(_normalize(item), sort_keys=True, ensure_ascii=False)


def _concat_unique(current: list, incoming: list) -> list:
    """Append *incoming* elements to *current*, skipping ones already present."""
    seen = {_content_key(v) for v in current}
    result = list(current)
    for v in incoming:
        key = _content_key(v)
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result

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

    def _window_tokens(self) -> int:
        """Per-window input budget (tokens) for a single extraction call.

        Sized as a fraction of the model's context window minus the system
        prompt, so the document text, the schema-hint prompt, and the model's
        output all fit in one call. Floored at ``_MIN_WINDOW_TOKENS``.
        """
        window = self._llm.context_window("mini")
        budget = int(window * _INPUT_WINDOW_RATIO) - estimate_tokens(self._system_prompt)
        return max(_MIN_WINDOW_TOKENS, budget)

    async def _extract_once(self, doc: Document) -> list[dict] | None:
        """Extract from *doc*, splitting into token-bounded windows when needed.

        ``doc.text`` is split into overlapping windows so no single LLM call
        exceeds the model's context window. Each window is extracted
        independently, then the results are merged: list mode (``record_mode=many``)
        concatenates the per-window record lists as a map step, dropping the
        duplicates the overlap introduces; single mode (``record_mode=one``)
        reconciles the per-window records into one as a reduce step. Injected
        identity fields (``doc_id``, item id) are applied once, after the merge, so
        item ids index across the whole merged list rather than restarting per
        window. A short document yields a single window and behaves exactly like
        one call.
        """
        budget = self._window_tokens()
        windows = split_text_by_tokens(doc.text, budget, int(budget * _WINDOW_OVERLAP_RATIO))
        if self._record_mode == RecordMode.MANY:
            return await self._extract_many(doc, windows)
        return await self._extract_one(doc, windows)

    async def _complete_window(self, doc: Document, window_text: str) -> dict | None:
        """Run one LLM extraction call for *window_text*.

        Returns the raw ``CompletionResult`` on success, or ``None`` when the
        model returned no content (which propagates as a retry).
        """
        t0 = time.monotonic()
        result = await self._llm.complete(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": window_text},
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
        return result

    async def _extract_many(self, doc: Document, windows: list[str]) -> list[dict] | None:
        """Map step: extract per window, merge + dedup, then inject ids globally."""
        per_window: list[list[dict]] = []
        for window_text in windows:
            result = await self._complete_window(doc, window_text)
            if result is None:
                return None
            items = self._parse_list_items(doc, result["content"], result)
            if items is None:
                return None
            per_window.append(items)

        merged = self._dedup_across_windows(per_window)
        records = []
        for index, item in enumerate(merged):
            injected = {k: fn(doc, item, index) for k, fn in self._injected_fields.items()}
            records.append({**item, **injected})
        return records

    def _dedup_across_windows(self, per_window: list[list[dict]]) -> list[dict]:
        """Flatten per-window records, dropping overlap-induced duplicates.

        A record is dropped only when an identical record (by normalized content)
        appeared in the *immediately preceding* window — the only window it shares
        an overlap region with. Duplicates within a single window are kept (the
        model returned them deliberately), as are identical records that recur in
        non-adjacent windows (genuinely repeated content rather than overlap).
        """
        merged: list[dict] = []
        prev_keys: set[str] = set()
        for items in per_window:
            cur_keys: set[str] = set()
            for item in items:
                key = _content_key(item)
                if key in prev_keys:
                    continue
                merged.append(item)
                cur_keys.add(key)
            prev_keys = cur_keys
        return merged

    async def _extract_one(self, doc: Document, windows: list[str]) -> list[dict] | None:
        """Reduce step: extract per window, reconcile into one record, inject ids."""
        extractions: list[dict] = []
        for window_text in windows:
            result = await self._complete_window(doc, window_text)
            if result is None:
                return None
            extraction = self._parse_single_item(doc, result["content"], result)
            if extraction is None:
                return None
            extractions.append(extraction)

        extraction = self._reconcile_one(extractions)
        injected = {k: fn(doc, extraction, 0) for k, fn in self._injected_fields.items()}
        return [{**extraction, **injected}]

    def _reconcile_one(self, extractions: list[dict]) -> dict:
        """Fold per-window single-record extractions into one record.

        A single window (the common case) passes through unchanged. Across
        windows, each field takes the first non-null value found; list-valued
        fields are concatenated, with elements the overlap duplicated removed, so
        a field's matches spread across windows are gathered without repeats.
        Conflicting scalar values resolve to the earliest window.
        """
        if len(extractions) == 1:
            return extractions[0]
        merged: dict = {}
        for extraction in extractions:
            for key, value in extraction.items():
                current = merged.get(key)
                if current is None or current == []:
                    merged[key] = value
                elif isinstance(current, list) and isinstance(value, list):
                    merged[key] = _concat_unique(current, value)
        return merged

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

    def _parse_single_item(self, doc: Document, content: str, raw_result: dict) -> dict | None:
        """Parse one window's response into a single extraction dict (no injection)."""
        try:
            parsed = json.loads(content)
            jsonschema.validate(instance=parsed, schema=self._extraction_schema)
            return parsed
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
            return extraction

    def _parse_list_items(self, doc: Document, content: str, raw_result: dict) -> list[dict] | None:
        """Parse one window's response into its list of extraction items (no injection)."""
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
        return items
