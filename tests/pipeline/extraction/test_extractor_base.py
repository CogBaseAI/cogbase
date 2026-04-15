"""Tests for ExtractorBase and the extractor path in ingest()."""

import pytest
from pydantic import BaseModel

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.pipeline.ingestion.pipeline import ingest, setup_extraction
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

class NounRecord(BaseModel):
    noun_id: str
    doc_id: str
    text: str


_NOUN_SCHEMA = CollectionSchema(
    name="nouns",
    id_field="noun_id",
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

    async def _extract_once(self, doc: Document) -> NounRecord:
        self._calls.append((doc.text, doc.doc_id))
        return NounRecord(noun_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, text=doc.text)


class EmptyExtractor(ExtractorBase):
    """Always returns None — represents an extractor that finds nothing parseable."""

    def __init__(self) -> None:
        super().__init__(max_retries=0)

    @property
    def collection(self) -> str:
        return "nouns"

    @property
    def schema(self) -> CollectionSchema:
        return _NOUN_SCHEMA

    async def _extract_once(self, doc: Document) -> None:
        return None


class StubEmbedder(EmbedderBase):
    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        return [c.model_copy(update={"embedding": [1.0, 0.0]}) for c in chunks]


# ---------------------------------------------------------------------------
# ExtractorBase contract
# ---------------------------------------------------------------------------

class TestExtractorBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ExtractorBase()  # type: ignore[abstract]

    def test_concrete_subclass_requires_all_methods(self):
        """Subclass missing extract() must still be abstract."""

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
        assert e.schema.id_field == "noun_id"


# ---------------------------------------------------------------------------
# ingest() — extractor path
# ---------------------------------------------------------------------------

@pytest.fixture
def vector_store():
    return FAISSVectorStore(dim=2)


@pytest.fixture
def structured_store():
    return InMemoryStructuredStore()


@pytest.fixture
def chunker():
    return FixedSizeChunker(chunk_size=50, overlap=0)


@pytest.fixture
def embedder():
    return StubEmbedder()


class TestIngestWithExtractors:
    @pytest.mark.asyncio
    async def test_extractor_called_with_full_text_and_doc_id(
        self, chunker, embedder, vector_store, structured_store
    ):
        extractor = StubExtractor()
        await setup_extraction([extractor], structured_store)
        text = "hello world foo bar"
        await ingest(
            Document(doc_id="doc-1", text=text),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
            extractors=[extractor],
            structured_store=structured_store,
        )
        assert extractor._calls == [(text, "doc-1")]

    @pytest.mark.asyncio
    async def test_records_saved_to_structured_store(
        self, chunker, embedder, vector_store, structured_store
    ):
        extractor = StubExtractor()
        await setup_extraction([extractor], structured_store)
        text = "alpha beta gamma"
        await ingest(
            Document(doc_id="doc-2", text=text),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
            extractors=[extractor],
            structured_store=structured_store,
        )
        rows = await structured_store.query("nouns")
        assert len(rows) == 1
        assert rows[0]["text"] == text

    @pytest.mark.asyncio
    async def test_empty_extractor_saves_nothing(
        self, chunker, embedder, vector_store, structured_store
    ):
        extractor = EmptyExtractor()
        await setup_extraction([extractor], structured_store)
        await ingest(
            Document(doc_id="doc-3", text="some text"),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
            extractors=[extractor],
            structured_store=structured_store,
        )
        rows = await structured_store.query("nouns")
        assert rows == []

    @pytest.mark.asyncio
    async def test_multiple_extractors_each_write_to_own_collection(
        self, chunker, embedder, vector_store, structured_store
    ):
        class TagRecord(BaseModel):
            tag_id: str
            doc_id: str
            label: str

        _tag_schema = CollectionSchema(
            name="tags",
            id_field="tag_id",
            fields={
                "tag_id": FieldSchema(type=FieldType.STRING),
                "doc_id": FieldSchema(type=FieldType.STRING),
                "label":  FieldSchema(type=FieldType.STRING),
            },
        )

        class TagExtractor(ExtractorBase):
            @property
            def collection(self) -> str:
                return "tags"

            @property
            def schema(self) -> CollectionSchema:
                return _tag_schema

            async def _extract_once(self, doc: Document) -> TagRecord:
                return TagRecord(tag_id=f"{doc.doc_id}-t0", doc_id=doc.doc_id, label="test-tag")

        extractors = [StubExtractor(), TagExtractor()]
        await setup_extraction(extractors, structured_store)
        await ingest(
            Document(doc_id="doc-4", text="hello world"),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
            extractors=extractors,
            structured_store=structured_store,
        )

        nouns = await structured_store.query("nouns")
        tags = await structured_store.query("tags")
        assert len(nouns) == 1
        assert len(tags) == 1
        assert tags[0]["label"] == "test-tag"

    @pytest.mark.asyncio
    async def test_no_extractors_skips_structured_store(
        self, chunker, embedder, vector_store
    ):
        """ingest() without extractors must not touch a structured store."""
        result = await ingest(
            Document(doc_id="doc-5", text="text without extraction"),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
        )
        assert len(result) > 0  # vector path still works

    @pytest.mark.asyncio
    async def test_missing_collection_raises(
        self, chunker, embedder, vector_store, structured_store
    ):
        """ingest() does not create collections — caller must call setup_extraction first."""
        with pytest.raises(KeyError):
            await ingest(
                Document(doc_id="doc-6", text="one two three"),
                chunker=chunker, embedder=embedder,
                vector_store=vector_store,
                extractors=[StubExtractor()],
                structured_store=structured_store,
            )

    @pytest.mark.asyncio
    async def test_setup_extraction_creates_collections(self, structured_store):
        from cogbase.pipeline.ingestion.pipeline import setup_extraction
        extractors = [StubExtractor()]
        await setup_extraction(extractors, structured_store)
        # Collection exists — query should not raise
        rows = await structured_store.query("nouns")
        assert rows == []

    @pytest.mark.asyncio
    async def test_setup_then_ingest(
        self, chunker, embedder, vector_store, structured_store
    ):
        from cogbase.pipeline.ingestion.pipeline import setup_extraction
        extractors = [StubExtractor()]
        await setup_extraction(extractors, structured_store)
        await ingest(
            Document(doc_id="doc-7", text="one two three"),
            chunker=chunker, embedder=embedder,
            vector_store=vector_store,
            extractors=extractors,
            structured_store=structured_store,
        )
        rows = await structured_store.query("nouns")
        assert len(rows) == 1
