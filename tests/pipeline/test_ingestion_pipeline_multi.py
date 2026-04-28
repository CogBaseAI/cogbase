"""Tests for multi-collection IngestionPipeline (steps, SummarizeCollection)."""

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
    SummarizeCollection,
    VectorCollection,
)
from cogbase.stores.vector.base import VectorCollectionSchema
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
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
# SummarizeCollection dataclass
# ---------------------------------------------------------------------------

class TestSummarizeCollection:
    def test_construction(self):
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="doc_summary", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
        )
        assert smc.name == "doc_summary"
        assert smc.max_tokens == 1024
        assert "2" in smc.prompt or "sentence" in smc.prompt.lower()

    def test_custom_prompt_and_tokens(self):
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="s", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
            prompt="One sentence only.",
            max_tokens=64,
        )
        assert smc.prompt == "One sentence only."
        assert smc.max_tokens == 64


# ---------------------------------------------------------------------------
# Multi-collection IngestionPipeline construction
# ---------------------------------------------------------------------------

class TestMultiCollectionPipelineConstruction:
    def _make_vc(self, name: str = "chunks") -> VectorCollection:
        return VectorCollection(
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

    def _make_smc(self, name: str = "summaries") -> SummarizeCollection:
        return SummarizeCollection(
            schema=VectorCollectionSchema(name=name, dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
        )

    def test_explicit_steps_with_all_three_types(self):
        vc = self._make_vc("document_chunks")
        sc = self._make_sc()
        smc = self._make_smc("document_summary")

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert",     "document_chunks"),
                ("extract-structured",     "tags"),
                ("summarize-embed-upsert", "document_summary"),
            ],
            vector_collections=[vc],
            structured_collections=[sc],
            summarize_collections=[smc],
        )

        assert pipeline.vector_collection_names == ["document_chunks", "document_summary"]
        assert len(pipeline.structured_schemas) == 1
        assert pipeline.structured_schemas[0].name == "tags"

    def test_auto_steps_generation_from_collections(self):
        vc = self._make_vc()
        sc = self._make_sc()
        smc = self._make_smc()

        pipeline = IngestionPipeline(
            name="app",
            vector_collections=[vc],
            structured_collections=[sc],
            summarize_collections=[smc],
        )

        # Steps auto-generated: vc first, then sc, then smc
        assert ("chunk-embed-upsert", "chunks") in pipeline._steps
        assert ("extract-structured", "tags") in pipeline._steps
        assert ("summarize-embed-upsert", "summaries") in pipeline._steps

    def test_two_vector_collections(self):
        vc1 = self._make_vc("col_a")
        vc2 = self._make_vc("col_b")

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert", "col_a"),
                ("chunk-embed-upsert", "col_b"),
            ],
            vector_collections=[vc1, vc2],
        )

        assert pipeline.vector_collection_names == ["col_a", "col_b"]

    def test_vector_collection_names_respects_step_order(self):
        vc = self._make_vc("chunks")
        smc = self._make_smc("summaries")

        # Summary step comes before chunk step
        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("summarize-embed-upsert", "summaries"),
                ("chunk-embed-upsert",     "chunks"),
            ],
            vector_collections=[vc],
            summarize_collections=[smc],
        )

        assert pipeline.vector_collection_names == ["summaries", "chunks"]


# ---------------------------------------------------------------------------
# runner_resources() helper
# ---------------------------------------------------------------------------

class TestRunnerResources:
    def test_returns_first_chunk_embed_store(self):
        store = FAISSVectorStore(dim=4)
        emb = StubEmbedding(dim=4)
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="chunks", dimensions=4),
            store=store,
            embedder=emb,
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
        )
        pipeline = IngestionPipeline(name="app", vector_collections=[vc])

        ss, vs, embedder, default = pipeline.runner_resources()
        assert vs is store
        assert embedder is emb
        assert default == "chunks"
        assert ss is None

    def test_falls_back_to_summarize_store(self):
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("summarize-embed-upsert", "summaries")],
            summarize_collections=[smc],
        )
        _, vs, _, default = pipeline.runner_resources()
        assert vs is smc.store
        assert default == "summaries"

    def test_structured_store_returned(self):
        store = InMemoryStructuredStore()
        sc = StructuredCollection(
            schema=StubExtractor().schema,
            store=store,
            extractor=StubExtractor(),
        )
        pipeline = IngestionPipeline(name="app", structured_collections=[sc])
        ss, _, _, _ = pipeline.runner_resources()
        assert ss is store

    def test_empty_pipeline_returns_all_none(self):
        pipeline = IngestionPipeline(name="empty")
        ss, vs, emb, default = pipeline.runner_resources()
        assert all(v is None for v in (ss, vs, emb, default))


# ---------------------------------------------------------------------------
# summarize-embed-upsert ingestion
# ---------------------------------------------------------------------------

class TestSummarizeEmbedUpsert:
    def _make_pipeline_with_summary(self, summary_text: str) -> tuple[IngestionPipeline, FAISSVectorStore]:
        vector_store = FAISSVectorStore(dim=4)
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=_make_llm(summary=summary_text),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("summarize-embed-upsert", "summaries")],
            summarize_collections=[smc],
        )
        return pipeline, vector_store

    @pytest.mark.asyncio
    async def test_summary_chunk_upserted(self):
        pipeline, vector_store = self._make_pipeline_with_summary("Contract summary.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="Long contract text here..."))
        assert vector_store.ntotal == 1

    @pytest.mark.asyncio
    async def test_summary_chunk_id_is_doc_id_with_suffix(self):
        pipeline, vector_store = self._make_pipeline_with_summary("Summary text.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="doc-42", text="Some text."))
        chunks = await vector_store.search("summaries", [0.1] * 4, top_k=1)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-42__summary"
        assert chunks[0].doc_id == "doc-42"

    @pytest.mark.asyncio
    async def test_summary_text_stored_in_chunk(self):
        pipeline, vector_store = self._make_pipeline_with_summary("The parties agree to NDA terms.")
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="contract text"))
        chunks = await vector_store.search("summaries", [0.1] * 4, top_k=1)
        assert chunks[0].text == "The parties agree to NDA terms."

    @pytest.mark.asyncio
    async def test_empty_llm_response_skips_upsert(self):
        vector_store = FAISSVectorStore(dim=4)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(return_value={"content": None, "tool_calls": None})
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=llm,
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("summarize-embed-upsert", "summaries")],
            summarize_collections=[smc],
        )
        await pipeline.setup()
        await pipeline._ingest(Document(doc_id="d-001", text="text"))
        assert vector_store.ntotal == 0

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_raise(self):
        vector_store = FAISSVectorStore(dim=4)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=vector_store,
            embedder=StubEmbedding(dim=4),
            llm=llm,
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[("summarize-embed-upsert", "summaries")],
            summarize_collections=[smc],
        )
        await pipeline.setup()
        # Must not raise even though LLM failed
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

        vc = VectorCollection(
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
        smc = SummarizeCollection(
            schema=VectorCollectionSchema(name="summaries", dimensions=4),
            store=summary_store,
            embedder=StubEmbedding(dim=4),
            llm=_make_llm("Short summary."),
        )

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                ("chunk-embed-upsert",     "chunks"),
                ("extract-structured",     "tags"),
                ("summarize-embed-upsert", "summaries"),
            ],
            vector_collections=[vc],
            structured_collections=[sc],
            summarize_collections=[smc],
        )

        await pipeline.setup()
        count = await pipeline._ingest(Document(doc_id="d-001", text="word " * 20))

        assert chunk_store.ntotal > 0, "chunk-embed-upsert did not populate vector store"
        assert count == 1, "extract-structured did not produce a record"
        assert summary_store.ntotal == 1, "summarize-embed-upsert did not upsert summary"

    @pytest.mark.asyncio
    async def test_setup_creates_all_structured_collections(self):
        struct_store_a = InMemoryStructuredStore()
        struct_store_b = InMemoryStructuredStore()

        _fields = {"tag_id": FieldSchema(type=FieldType.STRING), "doc_id": FieldSchema(type=FieldType.STRING), "value": FieldSchema(type=FieldType.STRING)}

        class ExtA(ExtractorBase):
            @property
            def collection(self): return "col_a"
            @property
            def schema(self): return CollectionSchema(name="col_a", primary_fields=["tag_id"], fields=_fields)
            async def _extract_once(self, doc): return None

        class ExtB(ExtractorBase):
            @property
            def collection(self): return "col_b"
            @property
            def schema(self): return CollectionSchema(name="col_b", primary_fields=["tag_id"], fields=_fields)
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
# Config: SummarizeCollectionConfig and new step type
# ---------------------------------------------------------------------------

class TestSummarizeCollectionConfig:
    def test_parse_summarize_collection(self):
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
            summarize_collections:
              - name: document_summary
                prompt: "Summarize in one sentence."
                max_tokens: 128
            pipeline:
              steps:
                - tool: summarize-embed-upsert
                  collection: document_summary
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.summarize_collections) == 1
        assert cfg.summarize_collections[0].name == "document_summary"
        assert cfg.summarize_collections[0].prompt == "Summarize in one sentence."
        assert cfg.summarize_collections[0].max_tokens == 128

    def test_summarize_collection_requires_embedding(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            summarize_collections:
              - name: document_summary
        """)
        with pytest.raises(Exception, match="embedding is required when summarize_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_unknown_summarize_collection_in_step_raises(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            summarize_collections:
              - name: document_summary
            pipeline:
              steps:
                - tool: summarize-embed-upsert
                  collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown summarize collection"):
            AppConfig.from_yaml(yaml_text)
