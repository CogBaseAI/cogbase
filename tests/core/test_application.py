"""Tests for IngestionPipeline, VectorCollection, and StructuredCollection."""

import pytest
from pydantic import ValidationError
from pydantic import BaseModel

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult, StructuredCollection, VectorCollection, PipelineStep
from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.chunking.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.pipeline.chunking.fixed import FixedSizeChunker
from cogbase.stores import CollectionSchema, FieldSchema, FieldType, Filter, StructuredStoreBase, VectorCollectionSchema, VectorStoreBase
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


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
    schema = CollectionSchema(
        name="tags",
        description="Extracted document tags: tag identifier, source document, and tag value.",
        primary_fields=["tag_id"],
        fields={
            "tag_id": FieldSchema(type=FieldType.STRING),
            "doc_id": FieldSchema(type=FieldType.STRING),
            "value":  FieldSchema(type=FieldType.STRING),
        },
    )

    async def _extract_once(self, doc: Document) -> list[TagRecord] | None:
        if not doc.text.strip():
            return None
        return [TagRecord(tag_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, value=doc.text[:10])]


# ---------------------------------------------------------------------------
# StructuredCollection
# ---------------------------------------------------------------------------


class TestStructuredCollection:
    def _make(self, collection_name: str = "tags") -> StructuredCollection:
        schema = StubExtractor().schema if collection_name == "tags" else CollectionSchema(
            name=collection_name,
            description="Test collection.",
            primary_fields=["tag_id"],
            fields={"tag_id": FieldSchema(type=FieldType.STRING)},
        )
        return StructuredCollection(schema=schema, store=InMemoryStructuredStore())

    def test_name_from_schema(self):
        sc = self._make()
        assert sc.name == "tags"


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

    def test_vector_only(self):
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="docs", dimensions=4, description="Test document chunks"),
            store=FAISSVectorStore(),
            embedder=StubEmbedding(dim=4),
        )
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        app = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="docs", chunker=chunker)],
            vector_collections=[vc],
        )
        assert app._vector_by_name
        assert app._structured_by_name == {}

    def test_structured_only(self):
        sc = StructuredCollection(schema=StubExtractor().schema, store=InMemoryStructuredStore())
        app = IngestionPipeline(name="app", structured_collections=[sc])
        assert app._vector_by_name == {}
        assert app._structured_by_name


# ---------------------------------------------------------------------------
# IngestionPipeline.ingest()
# ---------------------------------------------------------------------------


class TestIngestionPipelineIngest:
    async def _make_app(self) -> tuple[IngestionPipeline, FAISSVectorStore, InMemoryStructuredStore]:
        vector_store = FAISSVectorStore()
        structured_store = InMemoryStructuredStore()
        vc_schema = VectorCollectionSchema(name="docs", dimensions=4, description="Test document chunks")
        sc_schema = StubExtractor().schema
        await vector_store.create_collection(vc_schema)
        await structured_store.create_collection(sc_schema)
        vc = VectorCollection(
            schema=vc_schema,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        sc = StructuredCollection(schema=sc_schema, store=structured_store)
        app = IngestionPipeline(
            name="app",
            steps=[
                PipelineStep(tool="chunk-embed-upsert", collection="docs", chunker=FixedSizeChunker(chunk_size=50, overlap=0)),
                PipelineStep(tool="extract-structured", collection="tags", extractor=StubExtractor()),
            ],
            vector_collections=[vc],
            structured_collections=[sc],
        )
        return app, vector_store, structured_store

    @pytest.mark.asyncio
    async def test_ingest_populates_vector_store(self):
        app, vector_store, _ = await self._make_app()
        await app._ingest(Document(doc_id="doc-1", text="word " * 30))
        assert vector_store.ntotal("docs") > 0

    @pytest.mark.asyncio
    async def test_ingest_populates_structured_store(self):
        app, _, structured_store = await self._make_app()
        await app._ingest(Document(doc_id="doc-1", text="hello world contract clause"))
        rows = await structured_store.query("tags")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_ingest_returns_record_count(self):
        app, _, _ = await self._make_app()
        count = await app._ingest(Document(doc_id="doc-1", text="hello world contract clause"))
        assert count == 1

    @pytest.mark.asyncio
    async def test_ingest_empty_text_returns_zero(self):
        app, vector_store, _ = await self._make_app()
        count = await app._ingest(Document(doc_id="doc-empty", text=""))
        assert count == 0
        assert vector_store.ntotal("docs") == 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        app, vector_store, structured_store = await self._make_app()
        await app._ingest(Document(doc_id="doc-a", text="alpha beta gamma delta epsilon " * 3))
        await app._ingest(Document(doc_id="doc-b", text="one two three four five six seven " * 3))
        assert vector_store.ntotal("docs") > 0
        rows = await structured_store.query("tags")
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_vector_only_app_ingest_returns_zero(self):
        vector_store = FAISSVectorStore()
        vc_schema = VectorCollectionSchema(name="docs", dimensions=4, description="Test document chunks")
        await vector_store.create_collection(vc_schema)
        vc = VectorCollection(
            schema=vc_schema,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        app = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="docs", chunker=FixedSizeChunker(chunk_size=50, overlap=0))],
            vector_collections=[vc],
        )
        count = await app._ingest(Document(doc_id="doc-1", text="word " * 30))
        assert vector_store.ntotal("docs") > 0
        assert count == 0  # no structured collection

    @pytest.mark.asyncio
    async def test_structured_only_app_ingest(self):
        structured_store = InMemoryStructuredStore()
        sc_schema = StubExtractor().schema
        await structured_store.create_collection(sc_schema)
        sc = StructuredCollection(schema=sc_schema, store=structured_store)
        app = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="extract-structured", collection="tags", extractor=StubExtractor())],
            structured_collections=[sc],
        )
        await app._ingest(Document(doc_id="doc-1", text="important clause about termination"))
        rows = await structured_store.query("tags")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# IngestionPipeline.ingest_documents()
# ---------------------------------------------------------------------------


class TestIngestionPipelineIngestMany:
    import asyncio as _asyncio

    async def _make_app(self) -> tuple[IngestionPipeline, InMemoryStructuredStore]:
        structured_store = InMemoryStructuredStore()
        sc_schema = StubExtractor().schema
        await structured_store.create_collection(sc_schema)
        sc = StructuredCollection(schema=sc_schema, store=structured_store)
        app = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="extract-structured", collection="tags", extractor=StubExtractor())],
            structured_collections=[sc],
        )
        return app, structured_store

    @pytest.mark.asyncio
    async def test_returns_one_result_per_document(self):
        from cogbase.core.models import Document
        app, _ = await self._make_app()
        docs = [Document(doc_id=f"d-{i}", text=f"text {i}") for i in range(3)]
        results = await app.ingest_documents(docs)
        assert len(results) == 3
        assert all(isinstance(r, IngestResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_in_input_order(self):
        from cogbase.core.models import Document
        app, _ = await self._make_app()
        doc_ids = [f"d-{i:03d}" for i in range(8)]
        docs = [Document(doc_id=d, text=f"text for {d}") for d in doc_ids]
        results = await app.ingest_documents(docs, concurrency=3)
        assert [r.doc_id for r in results] == doc_ids

    @pytest.mark.asyncio
    async def test_success_and_records_extracted(self):
        from cogbase.core.models import Document
        app, _ = await self._make_app()
        results = await app.ingest_documents([Document(doc_id="d-001", text="some text")])
        assert results[0].success is True
        assert results[0].records_extracted == 1
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_each_result_reflects_own_records(self):
        from cogbase.core.models import Document
        app, _ = await self._make_app()
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

            async def _extract_once(self, doc: Document) -> list[TagRecord]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("extractor failed")
                return [TagRecord(tag_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, value=doc.text[:10])]

        structured_store = InMemoryStructuredStore()
        sc_schema = StubExtractor().schema
        await structured_store.create_collection(sc_schema)
        sc = StructuredCollection(schema=sc_schema, store=structured_store)
        app = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="extract-structured", collection="tags", extractor=FailFirstExtractor())],
            structured_collections=[sc],
        )

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
        app, _ = await self._make_app()
        assert await app.ingest_documents([]) == []

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        app, _ = await self._make_app()
        with pytest.raises(ValueError, match="concurrency"):
            await app.ingest_documents([], concurrency=0)
