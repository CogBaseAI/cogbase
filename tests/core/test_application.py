"""Tests for Application, VectorCollection, and StructuredCollection."""

import pytest
from pydantic import BaseModel

from cogbase.core.application import Application, StructuredCollection, VectorCollection
from cogbase.core.models import Chunk
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.stores.filters import Filter


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubEmbedder(EmbedderBase):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            c.model_copy(update={"embedding": [1.0] * self._dim})
            for c in chunks
        ]


class TagRecord(BaseModel):
    tag_id: str
    doc_id: str
    value: str


class StubExtractor(ExtractorBase):
    _collection = "tags"
    _schema = CollectionSchema(
        name="tags",
        id_field="tag_id",
        fields={
            "tag_id": FieldSchema(type=FieldType.STRING),
            "doc_id": FieldSchema(type=FieldType.STRING),
            "value":  FieldSchema(type=FieldType.STRING),
        },
    )

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def schema(self) -> CollectionSchema:
        return self._schema

    async def extract(self, text: str, doc_id: str) -> list[BaseModel]:
        if not text.strip():
            return []
        return [TagRecord(tag_id=f"{doc_id}-0", doc_id=doc_id, value=text[:10])]


# ---------------------------------------------------------------------------
# StructuredCollection
# ---------------------------------------------------------------------------


class TestStructuredCollection:
    def _make(self, collection_name: str = "tags") -> StructuredCollection:
        extractor = StubExtractor()
        schema = extractor.schema if collection_name == "tags" else CollectionSchema(
            name=collection_name,
            id_field="tag_id",
            fields={"tag_id": FieldSchema(type=FieldType.STRING)},
        )
        return StructuredCollection(
            schema=extractor.schema,
            store=InMemoryStructuredStore(),
            extractor=extractor,
        )

    def test_name_from_schema(self):
        sc = self._make()
        assert sc.name == "tags"

    def test_mismatched_extractor_raises(self):
        wrong_schema = CollectionSchema(
            name="other",
            id_field="tag_id",
            fields={"tag_id": FieldSchema(type=FieldType.STRING)},
        )
        with pytest.raises(ValueError, match="does not match schema.name"):
            StructuredCollection(
                schema=wrong_schema,
                store=InMemoryStructuredStore(),
                extractor=StubExtractor(),
            )


# ---------------------------------------------------------------------------
# Application construction
# ---------------------------------------------------------------------------


class TestApplicationConstruction:
    def test_empty_application(self):
        app = Application(name="empty")
        assert app.name == "empty"
        assert app.vector_collections == []
        assert app.structured_collections == []
        assert app.structured_schemas == []

    def test_vector_only(self):
        vc = VectorCollection(
            name="docs",
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedder(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        app = Application(name="app", vector_collections=[vc])
        assert len(app.vector_collections) == 1
        assert app.structured_collections == []

    def test_structured_only(self):
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=InMemoryStructuredStore(),
            extractor=StubExtractor(),
        )
        app = Application(name="app", structured_collections=[sc])
        assert app.vector_collections == []
        assert len(app.structured_collections) == 1

    def test_structured_schemas_property(self):
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=InMemoryStructuredStore(),
            extractor=StubExtractor(),
        )
        app = Application(name="app", structured_collections=[sc])
        schemas = app.structured_schemas
        assert len(schemas) == 1
        assert schemas[0].name == "tags"

    def test_collections_views_are_copies(self):
        """Mutating the returned list does not affect the application."""
        app = Application(name="app")
        app.vector_collections.append(object())  # type: ignore[arg-type]
        assert app.vector_collections == []


# ---------------------------------------------------------------------------
# Application.setup()
# ---------------------------------------------------------------------------


class TestApplicationSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_structured_collections(self):
        store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=store,
            extractor=StubExtractor(),
        )
        app = Application(name="app", structured_collections=[sc])
        await app.setup()

        # Collection now exists — querying it must not raise
        rows = await store.query("tags")
        assert rows == []

    @pytest.mark.asyncio
    async def test_setup_is_idempotent(self):
        store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=store,
            extractor=StubExtractor(),
        )
        app = Application(name="app", structured_collections=[sc])
        await app.setup()
        await app.setup()  # must not raise


# ---------------------------------------------------------------------------
# Application.ingest()
# ---------------------------------------------------------------------------


class TestApplicationIngest:
    def _make_app(self) -> tuple[Application, FAISSVectorStore, InMemoryStructuredStore]:
        vector_store = FAISSVectorStore(dim=4)
        structured_store = InMemoryStructuredStore()
        vc = VectorCollection(
            name="docs",
            store=vector_store,
            embedder=StubEmbedder(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=StubExtractor(),
        )
        app = Application(name="app", vector_collections=[vc], structured_collections=[sc])
        return app, vector_store, structured_store

    @pytest.mark.asyncio
    async def test_ingest_populates_vector_store(self):
        app, vector_store, _ = self._make_app()
        await app.setup()
        await app.ingest("word " * 30, "doc-1")
        assert vector_store.ntotal > 0

    @pytest.mark.asyncio
    async def test_ingest_populates_structured_store(self):
        app, _, structured_store = self._make_app()
        await app.setup()
        await app.ingest("hello world contract clause", "doc-1")
        rows = await structured_store.query("tags")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_ingest_empty_text_is_noop_for_vector(self):
        app, vector_store, _ = self._make_app()
        await app.setup()
        await app.ingest("", "doc-empty")
        assert vector_store.ntotal == 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        app, vector_store, structured_store = self._make_app()
        await app.setup()
        await app.ingest("alpha beta gamma delta epsilon " * 3, "doc-a")
        await app.ingest("one two three four five six seven " * 3, "doc-b")
        assert vector_store.ntotal > 0
        rows = await structured_store.query("tags")
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_vector_only_app_ingest(self):
        vector_store = FAISSVectorStore(dim=4)
        vc = VectorCollection(
            name="docs",
            store=vector_store,
            embedder=StubEmbedder(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        app = Application(name="app", vector_collections=[vc])
        await app.setup()
        await app.ingest("word " * 30, "doc-1")
        assert vector_store.ntotal > 0

    @pytest.mark.asyncio
    async def test_structured_only_app_ingest(self):
        structured_store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=StubExtractor(),
        )
        app = Application(name="app", structured_collections=[sc])
        await app.setup()
        await app.ingest("important clause about termination", "doc-1")
        rows = await structured_store.query("tags")
        assert len(rows) == 1
