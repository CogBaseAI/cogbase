"""Tests for FieldsExtractor — one record per document, no LLM.

The failures worth catching here are all silent ones. A field the record schema
does not declare is dropped by ``save``; a template naming something the document
lacks would otherwise be written as an ``Undefined``; and a document whose text
failed to parse must still get its identity row, or every query scoped through
that row returns empty rather than erroring.
"""

from __future__ import annotations

import pytest

from cogbase.config.config import ExtractStructuredStepConfig
from cogbase.core.models import Document
from cogbase.pipeline.extraction.fields import FieldsExtractor

_RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "framework": {"type": "string"},
        "part": {"type": "string"},
        "subpart": {"type": ["string", "null"]},
    },
}

_FIELDS = {
    "framework": "{{ doc.metadata.framework }}",
    "part": "{{ doc.metadata.part }}",
}


def _doc(text: str = "§ 211.100 Written procedures; deviations.", **metadata: str) -> Document:
    # Document is frozen, so each variation is a new instance rather than a mutation.
    return Document(
        doc_id="21-cfr-211-subpart-f",
        text=text,
        metadata=metadata or {"framework": "21 CFR", "part": "211"},
    )


async def test_renders_one_record_from_document_metadata() -> None:
    extractor = FieldsExtractor(_FIELDS, record_schema=_RECORD_SCHEMA)
    records = await extractor.extract(_doc())
    assert records == [
        {"framework": "21 CFR", "part": "211", "doc_id": "21-cfr-211-subpart-f"}
    ]


async def test_a_string_column_stays_a_string() -> None:
    """Templates render through a NativeEnvironment, so a bare ``{{ x }}`` returns
    the literal-eval'd object and metadata "211" arrives as int 211. Phase-2-style
    equality filters against a string column then match nothing, and a query that
    finds nothing is not an error — this is the silent join failure."""
    records = await FieldsExtractor(_FIELDS, record_schema=_RECORD_SCHEMA).extract(_doc())
    assert records is not None
    assert records[0]["part"] == "211"
    assert isinstance(records[0]["part"], str)


async def test_doc_id_is_injected_not_declared() -> None:
    """It joins this record to the document's other collections, so it is the
    pipeline's to set rather than a template's to get right."""
    extractor = FieldsExtractor(_FIELDS, record_schema=_RECORD_SCHEMA)
    records = await extractor.extract(_doc())
    assert records is not None
    assert records[0]["doc_id"] == "21-cfr-211-subpart-f"


async def test_one_record_per_document_however_long_the_text() -> None:
    """No text is read, so there is nothing for a second record to be about."""
    extractor = FieldsExtractor(_FIELDS, record_schema=_RECORD_SCHEMA)
    records = await extractor.extract(_doc(text="§ 211.100 ...\n" * 5_000))
    assert records is not None and len(records) == 1


async def test_a_document_with_no_text_still_gets_its_record() -> None:
    """``ExtractorBase.extract`` returns None for blank text, which is right for an
    extractor that reads text and wrong here. A scanned PDF that yielded nothing
    still has an identity, and without the row every scoped query comes back empty
    instead of erroring."""
    records = await FieldsExtractor(_FIELDS, record_schema=_RECORD_SCHEMA).extract(
        _doc(text="")
    )
    assert records is not None and len(records) == 1


async def test_a_template_naming_absent_metadata_raises() -> None:
    """Not an Undefined written into the record — that is the silent wrong value
    the strict renderer exists to prevent."""
    extractor = FieldsExtractor(
        {"framework": "{{ doc.metadata.nope }}"}, record_schema=_RECORD_SCHEMA
    )
    with pytest.raises(ValueError, match="fields.framework"):
        await extractor.extract(_doc())


async def test_an_optional_field_can_default() -> None:
    extractor = FieldsExtractor(
        {"subpart": "{{ doc.metadata.subpart | default(None, true) }}"},
        record_schema=_RECORD_SCHEMA,
    )
    records = await extractor.extract(_doc())
    assert records is not None and records[0]["subpart"] is None


def test_a_field_absent_from_the_record_schema_is_rejected() -> None:
    """``save`` drops columns the collection does not have, so this would vanish
    rather than error — caught at app build instead."""
    with pytest.raises(ValueError, match="jurisdiction"):
        FieldsExtractor(
            {"jurisdiction": "{{ doc.metadata.jurisdiction }}"},
            record_schema=_RECORD_SCHEMA,
        )


def test_empty_fields_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires 'fields'"):
        FieldsExtractor({}, record_schema=_RECORD_SCHEMA)


def test_record_schema_without_doc_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        FieldsExtractor(_FIELDS, record_schema={"type": "object", "properties": {}})


def test_step_config_accepts_fields_without_an_extractor() -> None:
    step = ExtractStructuredStepConfig(collection="regulation_metadata", fields=_FIELDS)
    assert step.extractor is None
    assert step.fields == _FIELDS


def test_step_config_rejects_neither_extractor_nor_fields() -> None:
    """Such a step writes rows containing only doc_id — useless, and nothing
    downstream reports it as wrong."""
    with pytest.raises(ValueError, match="neither 'extractor' nor 'fields'"):
        ExtractStructuredStepConfig(collection="regulation_metadata")
