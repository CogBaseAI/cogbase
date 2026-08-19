"""Extractor that reads nothing: one record built entirely from ``fields:``.

Some structured collections hold facts about the *document* rather than findings
in its text — which framework, part and version a regulation is, what jurisdiction
a policy belongs to, which contract a clause set came from. Those values are known
exactly at ingest, on ``doc.metadata``. Asking an LLM to read them back out of the
document is strictly worse: it costs a call, it can only lose fidelity, and the
failure is silent, because a retyped scoping key does not error — it just stops
matching, and whatever queries that collection quietly returns nothing.

Before ``fields:`` existed the only route from document metadata into an extracted
record was through the text, so configs wrote the values into a header block and
prompted a model to copy them back. ``FieldsExtractor`` removes the round trip: the
step declares the templates, the pipeline renders them against the document, and
one record is written per document with no model involved.

Use it by omitting ``extractor`` from an ``extract-structured`` step that declares
``fields:``::

    - tool: extract-structured
      collection: regulation_metadata
      fields:
        framework: "{{ doc.metadata.framework }}"
        part:      "{{ doc.metadata.part }}"

Always one record per document — there is no text to find several of anything in,
so ``record_mode`` has nothing to vary. ``doc_id`` is injected as it is everywhere
else, so the record joins to the document's other collections unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.templating import jinja_available, render_defined

logger = logging.getLogger(__name__)


def _declares_string(field_schema: dict) -> bool:
    """Whether *field_schema* says this column holds a string (nullable or not).

    Handles the three shapes a JSON Schema writes a nullable string in: ``"string"``,
    ``["string", "null"]``, and ``anyOf`` of the two.
    """
    declared = field_schema.get("type")
    if isinstance(declared, list):
        return "string" in declared
    if declared == "string":
        return True
    return any(
        variant.get("type") == "string" for variant in field_schema.get("anyOf", [])
    )


class FieldsExtractor(ExtractorBase):
    """Emit one record per document from rendered ``fields:`` templates.

    Args:
        fields:        ``{name: value}``. A ``str`` value is a Jinja2 template
                       rendered with the document exposed as ``doc``; anything else
                       is a constant, stored as written. Must be non-empty — a step
                       with neither an extractor nor fields would write empty
                       records forever.
        record_schema: JSON Schema of the stored record, used to check every field
                       name is a column the collection actually has.
        app_id:        Owning application, for log attribution.

    Raises:
        ValueError: if ``fields`` is empty, if jinja2 is not installed, or if a
            field name is absent from *record_schema*.
    """

    def __init__(
        self,
        fields: dict[str, Any],
        *,
        record_schema: dict,
        app_id: str = "",
    ) -> None:
        # No retries: rendering is deterministic, so a second attempt produces the
        # same result. A failure here is a config error, not a transient one.
        super().__init__(max_retries=0, app_id=app_id)
        if not fields:
            raise ValueError(
                "extract-structured with no 'extractor' requires 'fields': the step "
                "would otherwise write a record containing only doc_id"
            )
        if any(isinstance(value, str) for value in fields.values()) and not jinja_available():
            raise ValueError(
                "extract-structured 'fields:' requires jinja2: pip install 'cogbase[api]'"
            )

        record_fields = set(record_schema.get("properties", {}).keys())
        if "doc_id" not in record_fields:
            raise ValueError("record schema must include 'doc_id'")
        # Same contract the LLM extractor applies to its own overlay, and for the
        # same reason: ``save`` drops fields the collection has no column for, so a
        # name that is not in the record schema vanishes without an error.
        unknown = sorted(set(fields) - record_fields)
        if unknown:
            raise ValueError(
                f"fields: {unknown} are not in the collection's record schema; "
                "they would be dropped on save"
            )

        self._fields = dict(fields)
        self._string_fields = {
            name
            for name in fields
            if _declares_string(record_schema["properties"].get(name, {}))
        }

    async def _extract_once(self, doc: Document) -> list[dict[str, Any]] | None:
        record: dict[str, Any] = {}
        for name, declared in self._fields.items():
            # A non-string is a constant, and has to stay one: written as a template
            # instead, ``{{ false }}`` is constant-folded to a literal at compile
            # time and renders as the *string* "False", which a boolean column
            # stores as True — silently, a non-empty string being truthy.
            if not isinstance(declared, str):
                record[name] = declared
                continue
            value = render_defined(declared, {"doc": doc}, what=f"fields.{name}")
            # Templates render through a NativeEnvironment, so a bare ``{{ x }}``
            # returns the literal-eval'd Python object: metadata "211" arrives as
            # int 211. A column the schema declares as a string then holds a number,
            # and every equality filter against it stops matching — silently, since
            # a query that finds nothing is not an error. Coerce back to what the
            # collection says it stores.
            if name in self._string_fields and value is not None and not isinstance(value, str):
                value = str(value)
            record[name] = value
        record["doc_id"] = doc.doc_id
        logger.info(
            "fields_extractor.extract app_id=%s doc_id=%s fields=%d",
            self._app_id, doc.doc_id, len(self._fields),
        )
        return [record]

    async def extract(self, doc: Document) -> list[dict[str, Any]] | None:
        """Render regardless of whether *doc* has text.

        ``ExtractorBase.extract`` returns ``None`` for a blank document, which is
        right for an extractor that reads text and wrong here: the values come from
        metadata, and a document whose text failed to parse still has an identity
        worth recording. Without this a scanned PDF would produce no metadata row,
        and every query scoped through that row would come back empty rather than
        erroring.
        """
        return await self._extract_once(doc)
