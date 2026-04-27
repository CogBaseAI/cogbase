"""Tests for IngestionPipeline, VectorCollection, and StructuredCollection."""

import pytest
from pydantic import ValidationError
from pydantic import BaseModel

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult, StructuredCollection, VectorCollection
from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores.base import StructuredStoreBase, VectorCollectionSchema, VectorStoreBase
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from cogbase.stores.filters import Filter


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubEmbedding(EmbeddingBase):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self._dim for _ in texts]


class TagRecord(BaseModel):
    tag_id: str
    doc_id: str
    value: str


class StubExtractor(ExtractorBase):
    _collection = "tags"
    _schema = CollectionSchema(
        name="tags",
        primary_fields=["tag_id"],
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

    async def _extract_once(self, doc: Document) -> TagRecord | None:
        if not doc.text.strip():
            return None
        return TagRecord(tag_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, value=doc.text[:10])


# ---------------------------------------------------------------------------
# StructuredCollection
# ---------------------------------------------------------------------------


class TestStructuredCollection:
    def _make(self, collection_name: str = "tags") -> StructuredCollection:
        extractor = StubExtractor()
        schema = extractor.schema if collection_name == "tags" else CollectionSchema(
            name=collection_name,
            primary_fields=["tag_id"],
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
            primary_fields=["tag_id"],
            fields={"tag_id": FieldSchema(type=FieldType.STRING)},
        )
        with pytest.raises(ValueError, match="does not match schema.name"):
            StructuredCollection(
                schema=wrong_schema,
                store=InMemoryStructuredStore(),
                extractor=StubExtractor(),
            )


def test_field_schema_rejects_json_schema_on_non_json_type():
    with pytest.raises(ValidationError, match="json_schema is only valid for FieldType.JSON fields"):
        FieldSchema(type=FieldType.STRING, json_schema='{"status": "string"}')


# ---------------------------------------------------------------------------
# IngestionPipeline construction
# ---------------------------------------------------------------------------


class TestIngestionPipelineConstruction:
    def test_empty_application(self):
        app = IngestionPipeline(name="empty")
        assert app.name == "empty"
        assert app._vector_by_name == {}
        assert app._structured_by_name == {}
        assert app.structured_schemas == []

    def test_vector_only(self):
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="docs", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        app = IngestionPipeline(name="app", vector_collections=[vc])
        assert app._vector_by_name
        assert app._structured_by_name == {}

    def test_structured_only(self):
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=InMemoryStructuredStore(),
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
        assert app._vector_by_name == {}
        assert app._structured_by_name

    def test_structured_schemas_property(self):
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=InMemoryStructuredStore(),
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
        schemas = app.structured_schemas
        assert len(schemas) == 1
        assert schemas[0].name == "tags"


# ---------------------------------------------------------------------------
# IngestionPipeline.setup()
# ---------------------------------------------------------------------------


class TestIngestionPipelineSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_structured_collections(self):
        store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=store,
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
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
        app = IngestionPipeline(name="app", structured_collections=[sc])
        await app.setup()
        await app.setup()  # must not raise


# ---------------------------------------------------------------------------
# IngestionPipeline.ingest()
# ---------------------------------------------------------------------------


class TestIngestionPipelineIngest:
    def _make_app(self) -> tuple[IngestionPipeline, FAISSVectorStore, InMemoryStructuredStore]:
        vector_store = FAISSVectorStore(dim=4)
        structured_store = InMemoryStructuredStore()
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="docs", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", vector_collections=[vc], structured_collections=[sc])
        return app, vector_store, structured_store

    @pytest.mark.asyncio
    async def test_ingest_populates_vector_store(self):
        app, vector_store, _ = self._make_app()
        await app.setup()
        await app._ingest(Document(doc_id="doc-1", text="word " * 30))
        assert vector_store.ntotal > 0

    @pytest.mark.asyncio
    async def test_ingest_populates_structured_store(self):
        app, _, structured_store = self._make_app()
        await app.setup()
        await app._ingest(Document(doc_id="doc-1", text="hello world contract clause"))
        rows = await structured_store.query("tags")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_ingest_returns_record_count(self):
        app, _, _ = self._make_app()
        await app.setup()
        count = await app._ingest(Document(doc_id="doc-1", text="hello world contract clause"))
        assert count == 1

    @pytest.mark.asyncio
    async def test_ingest_empty_text_returns_zero(self):
        app, vector_store, _ = self._make_app()
        await app.setup()
        count = await app._ingest(Document(doc_id="doc-empty", text=""))
        assert count == 0
        assert vector_store.ntotal == 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        app, vector_store, structured_store = self._make_app()
        await app.setup()
        await app._ingest(Document(doc_id="doc-a", text="alpha beta gamma delta epsilon " * 3))
        await app._ingest(Document(doc_id="doc-b", text="one two three four five six seven " * 3))
        assert vector_store.ntotal > 0
        rows = await structured_store.query("tags")
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_vector_only_app_ingest_returns_zero(self):
        vector_store = FAISSVectorStore(dim=4)
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="docs", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        app = IngestionPipeline(name="app", vector_collections=[vc])
        await app.setup()
        count = await app._ingest(Document(doc_id="doc-1", text="word " * 30))
        assert vector_store.ntotal > 0
        assert count == 0  # no structured collection

    @pytest.mark.asyncio
    async def test_structured_only_app_ingest(self):
        structured_store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
        await app.setup()
        await app._ingest(Document(doc_id="doc-1", text="important clause about termination"))
        rows = await structured_store.query("tags")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# IngestionPipeline.ingest_documents()
# ---------------------------------------------------------------------------


class TestIngestionPipelineIngestMany:
    import asyncio as _asyncio

    def _make_app(self) -> tuple[IngestionPipeline, InMemoryStructuredStore]:
        structured_store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=StubExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
        return app, structured_store

    @pytest.mark.asyncio
    async def test_returns_one_result_per_document(self):
        from cogbase.core.models import Document
        app, _ = self._make_app()
        await app.setup()
        docs = [Document(doc_id=f"d-{i}", text=f"text {i}") for i in range(3)]
        results = await app.ingest_documents(docs)
        assert len(results) == 3
        assert all(isinstance(r, IngestResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_in_input_order(self):
        from cogbase.core.models import Document
        app, _ = self._make_app()
        await app.setup()
        doc_ids = [f"d-{i:03d}" for i in range(8)]
        docs = [Document(doc_id=d, text=f"text for {d}") for d in doc_ids]
        results = await app.ingest_documents(docs, concurrency=3)
        assert [r.doc_id for r in results] == doc_ids

    @pytest.mark.asyncio
    async def test_success_and_records_extracted(self):
        from cogbase.core.models import Document
        app, _ = self._make_app()
        await app.setup()
        results = await app.ingest_documents([Document(doc_id="d-001", text="some text")])
        assert results[0].success is True
        assert results[0].records_extracted == 1
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_each_result_reflects_own_records(self):
        from cogbase.core.models import Document
        app, _ = self._make_app()
        await app.setup()
        results = await app.ingest_documents(
            [
                Document(doc_id="d-001", text="first"),
                Document(doc_id="d-002", text="second"),
            ],
            concurrency=1,
        )
        assert results[0].records_extracted == 1
        assert results[1].records_extracted == 1

    @pytest.mark.asyncio
    async def test_failure_captured_not_raised(self):
        """A failing extractor on one doc does not abort the batch."""
        import asyncio

        from cogbase.core.models import Document

        call_count = 0

        class FailFirstExtractor(ExtractorBase):
            _collection = "tags"

            @property
            def collection(self) -> str:
                return self._collection

            @property
            def schema(self) -> CollectionSchema:
                return StubExtractor._schema

            async def _extract_once(self, doc: Document) -> TagRecord:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("extractor failed")
                return TagRecord(tag_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, value=doc.text[:10])

        structured_store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=structured_store,
            extractor=FailFirstExtractor(),
        )
        app = IngestionPipeline(name="app", structured_collections=[sc])
        await app.setup()

        results = await app.ingest_documents(
            [
                Document(doc_id="d-fail", text="will fail"),
                Document(doc_id="d-ok",   text="will succeed"),
            ],
            concurrency=1,
        )

        failed    = [r for r in results if not r.success]
        succeeded = [r for r in results if r.success]
        assert len(failed) == 1
        assert failed[0].doc_id == "d-fail"
        assert isinstance(failed[0].error, RuntimeError)
        assert len(succeeded) == 1
        assert succeeded[0].records_extracted == 1

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        app, _ = self._make_app()
        await app.setup()
        assert await app.ingest_documents([]) == []

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        app, _ = self._make_app()
        with pytest.raises(ValueError, match="concurrency"):
            await app.ingest_documents([], concurrency=0)
