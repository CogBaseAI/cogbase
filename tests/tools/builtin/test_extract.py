"""Unit tests for ExtractTool."""

import json

import pytest
from pydantic import BaseModel

from cogbase.core.models import Document
from cogbase.tools.builtin.extract import ExtractTool


def _doc(text: str = "Some content.") -> Document:
    return Document(doc_id="doc-42", text=text)


# ---------------------------------------------------------------------------
# Stub dependencies
# ---------------------------------------------------------------------------

class _Record(BaseModel):
    doc_id: str
    value: str


class StubExtractor:
    def __init__(self, record):
        self._record = record

    async def extract(self, doc: Document):
        return [self._record] if self._record is not None else None


class StubStructuredStore:
    def __init__(self):
        self.saved: list[tuple[str, list]] = []

    async def save(self, collection: str, records: list):
        self.saved.append((collection, records))


def _make_tool(record, collection_name: str = "test-col") -> ExtractTool:
    return ExtractTool(
        extractor=StubExtractor(record),
        structured_store=StubStructuredStore(),
        collection_name=collection_name,
    )


# ---------------------------------------------------------------------------
# Happy path — record returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_extracted_true():
    record = _Record(doc_id="doc-42", value="v")
    tool = ExtractTool(
        extractor=StubExtractor(record),
        structured_store=StubStructuredStore(),
        collection_name="test-col",
    )
    result = json.loads(await tool.handler({"document": _doc()}))
    assert result == {"doc_id": "doc-42", "extracted": True}


@pytest.mark.asyncio
async def test_run_saves_to_correct_collection():
    record = _Record(doc_id="doc-42", value="v")
    store = StubStructuredStore()
    tool = ExtractTool(
        extractor=StubExtractor(record),
        structured_store=store,
        collection_name="test-col",
    )
    await tool.handler({"document": _doc()})
    assert len(store.saved) == 1
    col, records = store.saved[0]
    assert col == "test-col"
    assert records == [record]


# ---------------------------------------------------------------------------
# No record (extractor returns None)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_extracted_false_when_none():
    store = StubStructuredStore()
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=store,
        collection_name="test-col",
    )
    result = json.loads(await tool.handler({"document": _doc()}))
    assert result == {"doc_id": "doc-42", "extracted": False}


@pytest.mark.asyncio
async def test_run_does_not_save_when_no_record():
    store = StubStructuredStore()
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=store,
        collection_name="test-col",
    )
    await tool.handler({"document": _doc()})
    assert store.saved == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_wrong_type_raises():
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=StubStructuredStore(),
        collection_name="test-col",
    )
    with pytest.raises(TypeError, match="Document"):
        await tool.handler({"document": 123})


@pytest.mark.asyncio
async def test_run_missing_key_raises():
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=StubStructuredStore(),
        collection_name="test-col",
    )
    with pytest.raises(KeyError):
        await tool.handler({})
