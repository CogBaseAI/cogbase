"""Tests for multi-collection IngestionPipeline (steps, DocumentCollection)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from cogbase.core.models import Document
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    DocumentCollection,
    ChunkCollection,
)
from cogbase.stores import CollectionSchema, FieldSchema, FieldType, VectorCollectionSchema
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class StubEmbedding(EmbeddingBase):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


class TagRecord(BaseModel):
    tag_id: str
    doc_id: str
    value: str


class StubExtractor(ExtractorBase):
    _collection = "tags"
    _schema = CollectionSchema(
        name="tags",
        description="Extracted document tags: tag identifier, source document, and tag value.",
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


def _make_llm(summary: str = "A short summary.") -> MagicMock:
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value={"content": summary, "tool_calls": None})
    return llm


# ---------------------------------------------------------------------------
# DocumentCollection dataclass
# ---------------------------------------------------------------------------

class TestDocumentCollection:
    def test_construction(self):
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="doc_summary", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
        )
        assert dc.name == "doc_summary"
        assert dc.max_tokens == 1024
        assert "sentence" in dc.prompt.lower()

    def test_custom_prompt_and_tokens(self):
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="s", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
            prompt="One sentence only.",
            max_tokens=64,
        )
        assert dc.prompt == "One sentence only."
        assert dc.max_tokens == 64

    def test_no_llm_defaults(self):
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="s", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
        )
        assert dc.llm is None
        assert dc.metadata_fields == []

    def test_metadata_fields(self):
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="s", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            metadata_fields=["customer_id", "deal_stage"],
        )
        assert dc.metadata_fields == ["customer_id", "deal_stage"]


# ---------------------------------------------------------------------------
# Multi-collection IngestionPipeline construction
# ---------------------------------------------------------------------------

class TestMultiCollectionPipelineConstruction:
    def _make_vc(self, name: str = "chunks") -> ChunkCollection:
        return ChunkCollection(
            schema=VectorCollectionSchema(name=name, dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )

    def _make_sc(self) -> StructuredCollection:
        return StructuredCollection(
            schema=StubExtractor().schema,
            store=InMemoryStructuredStore(),
            extractor=StubExtractor(),
        )

    def _make_dc(self, name: str = "summaries") -> DocumentCollection:
        return DocumentCollection(
            schema=VectorCollectionSchema(name=name, dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
        )

    def test_explicit_steps_with_all_three_types(self):
        vc = self._make_vc("document_chunks")
        sc = self._make_sc()
        dc = self._make_dc("document_summary")

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert",    "document_chunks"),
                ("extract-structured",    "tags"),
                ("document-embed-upsert", "document_summary"),
            ],
            chunk_collections=[vc],
            structured_collections=[sc],
            document_collections=[dc],
        )

        assert ("chunk-embed-upsert", "document_chunks") in pipeline._steps
        assert ("document-embed-upsert", "document_summary") in pipeline._steps
        assert "tags" in pipeline._structured_by_name

    def test_auto_steps_generation_from_collections(self):
        vc = self._make_vc()
        sc = self._make_sc()
        dc = self._make_dc()

        pipeline = IngestionPipeline(
            name="app",
            chunk_collections=[vc],
            structured_collections=[sc],
            document_collections=[dc],
        )

        assert ("chunk-embed-upsert", "chunks") in pipeline._steps
        assert ("extract-structured", "tags") in pipeline._steps
        assert ("document-embed-upsert", "summaries") in pipeline._steps

    def test_two_vector_collections(self):
        vc1 = self._make_vc("col_a")
        vc2 = self._make_vc("col_b")

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert", "col_a"),
                ("chunk-embed-upsert", "col_b"),
            ],
            chunk_collections=[vc1, vc2],
        )

        assert "col_a" in pipeline._chunk_by_name
        assert "col_b" in pipeline._chunk_by_name


# ---------------------------------------------------------------------------
# document-embed-upsert ingestion
# ---------------------------------------------------------------------------

class TestDocumentEmbedUpsert:
    def _make_pipeline_with_llm(self, summary_text: str) -> tuple[IngestionPipeline, FAISSVectorStore]:
        vector_store = FAISSVectorStore(dim=4)
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(summary=summary_text),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("document-embed-upsert", "summaries")],
            document_collections=[dc],
        )
        return pipeline, vector_store

    @pytest.mark.asyncio
    async def test_summary_chunk_upserted(self):
        pipeline, vector_store = self._make_pipeline_with_llm("Contract summary.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="Long contract text here..."))
        assert vector_store.ntotal == 1

    @pytest.mark.asyncio
    async def test_chunk_id_is_doc_id_with_document_suffix(self):
        pipeline, vector_store = self._make_pipeline_with_llm("Summary text.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="doc-42", text="Some text."))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-42__document"
        assert chunks[0].doc_id == "doc-42"

    @pytest.mark.asyncio
    async def test_summary_text_stored_in_chunk(self):
        pipeline, vector_store = self._make_pipeline_with_llm("The parties agree to NDA terms.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="contract text"))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert chunks[0].text == "The parties agree to NDA terms."

    @pytest.mark.asyncio
    async def test_no_llm_embeds_doc_text_directly(self):
        vector_store = FAISSVectorStore(dim=4)
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("document-embed-upsert", "summaries")],
            document_collections=[dc],
        )
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="raw document text"))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert len(chunks) == 1
        assert chunks[0].text == "raw document text"

    @pytest.mark.asyncio
    async def test_metadata_fields_projected_into_chunk(self):
        vector_store = FAISSVectorStore(dim=4)
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            metadata_fields=["customer_id", "deal_stage"],
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("document-embed-upsert", "summaries")],
            document_collections=[dc],
        )
        await pipeline.setup()
        await pipeline._ingest(Document(
            doc_id="d-001",
            text="transcript",
            metadata={"customer_id": "acme", "deal_stage": "negotiation", "internal": "skip"},
        ))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert chunks[0].metadata == {"customer_id": "acme", "deal_stage": "negotiation"}

    @pytest.mark.asyncio
    async def test_empty_llm_response_skips_upsert(self):
        vector_store = FAISSVectorStore(dim=4)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(return_value={"content": None, "tool_calls": None})
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=llm,
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("document-embed-upsert", "summaries")],
            document_collections=[dc],
        )
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="text"))
        assert vector_store.ntotal == 0

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_raise(self):
        vector_store = FAISSVectorStore(dim=4)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=llm,
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("document-embed-upsert", "summaries")],
            document_collections=[dc],
        )
        await pipeline.setup()
        count = await pipeline._ingest(Document(doc_id="d-001", text="text"))
        assert count == 0
        assert vector_store.ntotal == 0


# ---------------------------------------------------------------------------
# Full 3-step pipeline: chunk + extract + summarize
# ---------------------------------------------------------------------------

class TestThreeStepPipeline:
    @pytest.mark.asyncio
    async def test_all_three_steps_execute(self):
        chunk_store = FAISSVectorStore(dim=4)
        summary_store = FAISSVectorStore(dim=4)
        struct_store = InMemoryStructuredStore()

        vc = ChunkCollection(
            schema=VectorCollectionSchema(name="chunks", dimensions=4),
            store=chunk_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=struct_store,
            extractor=StubExtractor(),
        )
        dc = DocumentCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=summary_store,
            embedder=StubEmbedding(dim=4),
            llm=_make_llm("Short summary."),
        )

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert",    "chunks"),
                ("extract-structured",    "tags"),
                ("document-embed-upsert", "summaries"),
            ],
            chunk_collections=[vc],
            structured_collections=[sc],
            document_collections=[dc],
        )

        await pipeline.setup()
        count = await pipeline._ingest(Document(doc_id="d-001", text="word " * 20))

        assert chunk_store.ntotal > 0, "chunk-embed-upsert did not populate vector store"
        assert count == 1, "extract-structured did not produce a record"
        assert summary_store.ntotal == 1, "document-embed-upsert did not upsert summary"

    @pytest.mark.asyncio
    async def test_setup_creates_all_structured_collections(self):
        struct_store_a = InMemoryStructuredStore()
        struct_store_b = InMemoryStructuredStore()

        _fields = {"tag_id": FieldSchema(type=FieldType.STRING), "doc_id": FieldSchema(type=FieldType.STRING), "value": FieldSchema(type=FieldType.STRING)}

        class ExtA(ExtractorBase):
            @property
            def collection(self): return "col_a"
            @property
            def schema(self): return CollectionSchema(name="col_a", description="Test collection A.", primary_fields=["tag_id"], fields=_fields)
            async def _extract_once(self, doc): return None

        class ExtB(ExtractorBase):
            @property
            def collection(self): return "col_b"
            @property
            def schema(self): return CollectionSchema(name="col_b", description="Test collection B.", primary_fields=["tag_id"], fields=_fields)
            async def _extract_once(self, doc): return None

        sc_a = StructuredCollection(schema=ExtA().schema, store=struct_store_a, extractor=ExtA())
        sc_b = StructuredCollection(schema=ExtB().schema, store=struct_store_b, extractor=ExtB())

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("extract-structured", "col_a"),
                ("extract-structured", "col_b"),
            ],
            structured_collections=[sc_a, sc_b],
        )

        await pipeline.setup()

        # Both collections must exist after setup
        rows_a = await struct_store_a.query("col_a")
        rows_b = await struct_store_b.query("col_b")
        assert rows_a == []
        assert rows_b == []


# ---------------------------------------------------------------------------
# Config: DocumentCollectionConfig and new step type
# ---------------------------------------------------------------------------

class TestDocumentCollectionConfig:
    def test_parse_document_collection(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: test-app
            llm:
              provider: openai
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            document_collections:
              - name: document_summary
                prompt: "Summarize in one sentence."
                max_tokens: 128
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: document_summary
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.document_collections) == 1
        assert cfg.document_collections[0].name == "document_summary"
        assert cfg.document_collections[0].prompt == "Summarize in one sentence."
        assert cfg.document_collections[0].max_tokens == 128

    def test_document_collection_requires_embedding(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            document_collections:
              - name: document_summary
        """)
        with pytest.raises(Exception, match="embedding is required when document_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_unknown_document_collection_in_step_raises(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            document_collections:
              - name: document_summary
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown document collection"):
            AppConfig.from_yaml(yaml_text)
