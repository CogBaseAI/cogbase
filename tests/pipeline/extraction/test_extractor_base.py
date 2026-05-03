"""Tests for ExtractorBase."""

import pytest
from pydantic import BaseModel

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.stores import CollectionSchema, FieldSchema, FieldType


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

class NounRecord(BaseModel):
    noun_id: str
    doc_id: str
    text: str


_NOUN_SCHEMA = CollectionSchema(
    name="nouns",
    description="Extracted nouns with source document reference.",
    primary_fields=["noun_id"],
    fields={
        "noun_id": FieldSchema(type=FieldType.STRING),
        "doc_id":  FieldSchema(type=FieldType.STRING, index=True),
        "text":    FieldSchema(type=FieldType.STRING),
    },
)


class StubExtractor(ExtractorBase):
    """Returns a single NounRecord containing the full document text."""

    def __init__(self) -> None:
        super().__init__()
        self._calls: list[tuple[str, str]] = []

    @property
    def collection(self) -> str:
        return "nouns"

    @property
    def schema(self) -> CollectionSchema:
        return _NOUN_SCHEMA

    async def _extract_once(self, doc: Document) -> list[BaseModel] | None:
        self._calls.append((doc.text, doc.doc_id))
        return [NounRecord(noun_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, text=doc.text)]


class NullExtractor(ExtractorBase):
    """Always returns None — simulates a parse failure on every attempt."""

    def __init__(self, max_retries: int = 0) -> None:
        super().__init__(max_retries=max_retries)
        self.call_count = 0

    @property
    def collection(self) -> str:
        return "nouns"

    @property
    def schema(self) -> CollectionSchema:
        return _NOUN_SCHEMA

    async def _extract_once(self, doc: Document) -> None:
        self.call_count += 1
        return None


# ---------------------------------------------------------------------------
# ExtractorBase contract
# ---------------------------------------------------------------------------

class TestExtractorBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ExtractorBase()  # type: ignore[abstract]

    def test_concrete_subclass_requires_all_methods(self):
        """Subclass missing _extract_once() must still be abstract."""

        class Incomplete(ExtractorBase):
            @property
            def collection(self) -> str:
                return "x"

            @property
            def schema(self) -> CollectionSchema:
                return _NOUN_SCHEMA

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_stub_extractor_properties(self):
        e = StubExtractor()
        assert e.collection == "nouns"
        assert e.schema.name == "nouns"
        assert e.schema.primary_fields == ["noun_id"]


# ---------------------------------------------------------------------------
# ExtractorBase.extract() behaviour
# ---------------------------------------------------------------------------

class TestExtract:
    @pytest.mark.asyncio
    async def test_extract_calls_extract_once_with_doc(self):
        extractor = StubExtractor()
        doc = Document(doc_id="d1", text="hello world")
        await extractor.extract(doc)
        assert extractor._calls == [("hello world", "d1")]

    @pytest.mark.asyncio
    async def test_extract_returns_record_on_success(self):
        extractor = StubExtractor()
        doc = Document(doc_id="d2", text="some text")
        result = await extractor.extract(doc)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], NounRecord)
        assert result[0].doc_id == "d2"
        assert result[0].text == "some text"

    @pytest.mark.asyncio
    async def test_extract_returns_none_for_blank_text(self):
        extractor = StubExtractor()
        result = await extractor.extract(Document(doc_id="d3", text="   "))
        assert result is None
        assert extractor._calls == []

    @pytest.mark.asyncio
    async def test_extract_returns_none_after_all_retries_exhausted(self):
        extractor = NullExtractor(max_retries=0)
        result = await extractor.extract(Document(doc_id="d4", text="text"))
        assert result is None
        assert extractor.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_retries_on_none(self):
        extractor = NullExtractor(max_retries=2)
        await extractor.extract(Document(doc_id="d5", text="text"))
        assert extractor.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_extract_stops_at_first_success(self):
        """Returns immediately on first non-None result without exhausting retries."""
        extractor = StubExtractor()
        doc = Document(doc_id="d6", text="stop early")
        result = await extractor.extract(doc)
        assert result is not None and len(result) == 1
        assert len(extractor._calls) == 1
