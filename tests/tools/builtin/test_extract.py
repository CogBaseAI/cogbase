"""Unit tests for ExtractTool."""

import pytest
from pydantic import BaseModel

from cogbase.core.models import Document
from cogbase.core.session import Session
from cogbase.tools.builtin.extract import ExtractTool


def _doc(text: str = "Some content.") -> Document:
    return Document(doc_id="doc-42", text=text)


def _session() -> Session:
    return Session()


# ---------------------------------------------------------------------------
# Stub dependencies
# ---------------------------------------------------------------------------

class _Record(BaseModel):
    doc_id: str
    value: str


class StubExtractor:
    def __init__(self, record):
        self._record = record
        self.collection = "test-col"

    async def extract(self, doc: Document):
        return self._record


class StubStructuredStore:
    def __init__(self):
        self.saved: list[tuple[str, list]] = []

    async def save(self, collection: str, records: list):
        self.saved.append((collection, records))


# ---------------------------------------------------------------------------
# Happy path — record returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_extracted_true():
    record = _Record(doc_id="doc-42", value="v")
    store = StubStructuredStore()
    tool = ExtractTool(extractor=StubExtractor(record), structured_store=store)
    result = await tool.run({"document": _doc()}, _session())
    assert result == {"doc_id": "doc-42", "extracted": True}


@pytest.mark.asyncio
async def test_run_saves_to_correct_collection():
    record = _Record(doc_id="doc-42", value="v")
    store = StubStructuredStore()
    tool = ExtractTool(extractor=StubExtractor(record), structured_store=store)
    await tool.run({"document": _doc()}, _session())
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
    tool = ExtractTool(extractor=StubExtractor(None), structured_store=store)
    result = await tool.run({"document": _doc()}, _session())
    assert result == {"doc_id": "doc-42", "extracted": False}


@pytest.mark.asyncio
async def test_run_does_not_save_when_no_record():
    store = StubStructuredStore()
    tool = ExtractTool(extractor=StubExtractor(None), structured_store=store)
    await tool.run({"document": _doc()}, _session())
    assert store.saved == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_wrong_type_raises():
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=StubStructuredStore(),
    )
    with pytest.raises(TypeError, match="Document"):
        await tool.run({"document": 123}, _session())


@pytest.mark.asyncio
async def test_run_missing_key_raises():
    tool = ExtractTool(
        extractor=StubExtractor(None),
        structured_store=StubStructuredStore(),
    )
    with pytest.raises(KeyError):
        await tool.run({}, _session())
