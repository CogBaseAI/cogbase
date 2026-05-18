"""Tests for multi-collection IngestionPipeline (steps, VectorCollection)."""

from __future__ import annotations

from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from cogbase.core.app import CogBaseApp
from cogbase.core.models import Document
from cogbase.core.query_runner import QueryRunner
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.chunking.fixed import FixedSizeChunker
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    StructuredCollection,
    VectorCollection,
    PipelineStep,
)
from cogbase.stores import CollectionSchema, FieldSchema, FieldType, VectorCollectionSchema
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.vector.base import VectorStoreBase
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

    async def _extract_once(self, doc: Document) -> list[TagRecord] | None:
        if not doc.text.strip():
            return None
        return [TagRecord(tag_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, value=doc.text[:10])]


def _make_llm(summary: str = "A short summary.") -> MagicMock:
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value={"content": summary, "tool_calls": None})
    return llm


# ---------------------------------------------------------------------------
# VectorCollection dataclass
# ---------------------------------------------------------------------------

class TestVectorCollection:
    def test_construction(self):
        vc = VectorCollection(
            schema=VectorCollectionSchema(name="doc_summary", dimensions=4, description="Test summaries"),
            store=FAISSVectorStore(),
            embedder=StubEmbedding(dim=4),
        )
        assert vc.name == "doc_summary"
        assert vc.description == "Test summaries"


# ---------------------------------------------------------------------------
# Multi-collection IngestionPipeline construction
# ---------------------------------------------------------------------------

class TestMultiCollectionPipelineConstruction:
    def _make_vc(self, make_vector_store: Callable[[], VectorStoreBase], name: str = "chunks") -> VectorCollection:
        return VectorCollection(
            schema=VectorCollectionSchema(name=name, dimensions=4, description="Test chunks"),
            store=make_vector_store(),
            embedder=StubEmbedding(dim=4),
        )

    def _make_sc(self, make_structured_store: Callable[[], StructuredStoreBase]) -> StructuredCollection:
        return StructuredCollection(schema=StubExtractor().schema, store=make_structured_store())

    def test_explicit_steps_with_all_three_types(self, make_vector_store, make_structured_store):
        vc_chunks = self._make_vc(make_vector_store, "document_chunks")
        vc_summary = self._make_vc(make_vector_store, "document_summary")
        sc = self._make_sc(make_structured_store)

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                PipelineStep(tool="chunk-embed-upsert",    collection="document_chunks",  chunker=FixedSizeChunker()),
                PipelineStep(tool="extract-structured",    collection="tags"),
                PipelineStep(tool="document-embed-upsert", collection="document_summary"),
            ],
            vector_collections=[vc_chunks, vc_summary],
            structured_collections=[sc],
        )

        tools = [s.tool for s in pipeline._steps]
        assert "chunk-embed-upsert" in tools
        assert "document-embed-upsert" in tools
        assert "tags" in pipeline._structured_by_name

    def test_two_vector_collections(self, make_vector_store):
        vc1 = self._make_vc(make_vector_store, "col_a")
        vc2 = self._make_vc(make_vector_store, "col_b")

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                PipelineStep(tool="chunk-embed-upsert", collection="col_a", chunker=FixedSizeChunker()),
                PipelineStep(tool="chunk-embed-upsert", collection="col_b", chunker=FixedSizeChunker()),
            ],
            vector_collections=[vc1, vc2],
        )

        assert "col_a" in pipeline._vector_by_name
        assert "col_b" in pipeline._vector_by_name


# ---------------------------------------------------------------------------
# document-embed-upsert ingestion
# ---------------------------------------------------------------------------

class TestDocumentEmbedUpsert:
    _SUMMARIES_SCHEMA = VectorCollectionSchema(name="summaries", dimensions=4, description="Test document summaries")

    async def _make_pipeline_with_llm(
        self,
        summary_text: str,
        make_vector_store: Callable[[], VectorStoreBase],
    ) -> tuple[IngestionPipeline, VectorStoreBase]:
        vector_store = make_vector_store()
        await vector_store.create_collection(self._SUMMARIES_SCHEMA)
        vc = VectorCollection(
            schema=self._SUMMARIES_SCHEMA,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="document-embed-upsert", collection="summaries", llm=_make_llm(summary=summary_text))],
            vector_collections=[vc],
        )
        return pipeline, vector_store

    @pytest.mark.asyncio
    async def test_summary_chunk_upserted(self, make_vector_store):
        pipeline, vector_store = await self._make_pipeline_with_llm("Contract summary.", make_vector_store)
        await pipeline._ingest(Document(doc_id="d-001", text="Long contract text here..."))
        assert vector_store.ntotal("summaries") == 1

    @pytest.mark.asyncio
    async def test_chunk_id_is_doc_id_with_document_suffix(self, make_vector_store):
        pipeline, vector_store = await self._make_pipeline_with_llm("Summary text.", make_vector_store)
        await pipeline._ingest(Document(doc_id="doc-42", text="Some text."))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-42__document"
        assert chunks[0].doc_id == "doc-42"

    @pytest.mark.asyncio
    async def test_summary_text_stored_in_chunk(self, make_vector_store):
        pipeline, vector_store = await self._make_pipeline_with_llm("The parties agree to NDA terms.", make_vector_store)
        await pipeline._ingest(Document(doc_id="d-001", text="contract text"))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert chunks[0].text == "The parties agree to NDA terms."

    @pytest.mark.asyncio
    async def test_no_llm_embeds_doc_text_directly(self, make_vector_store):
        vector_store = make_vector_store()
        await vector_store.create_collection(self._SUMMARIES_SCHEMA)
        vc = VectorCollection(
            schema=self._SUMMARIES_SCHEMA,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="document-embed-upsert", collection="summaries")],
            vector_collections=[vc],
        )
        await pipeline._ingest(Document(doc_id="d-001", text="raw document text"))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert len(chunks) == 1
        assert chunks[0].text == "raw document text"

    @pytest.mark.asyncio
    async def test_metadata_fields_projected_into_chunk(self, make_vector_store):
        vector_store = make_vector_store()
        await vector_store.create_collection(self._SUMMARIES_SCHEMA)
        schema = self._SUMMARIES_SCHEMA.model_copy(update={"metadata_fields": ["customer_id", "deal_stage"]})
        vc = VectorCollection(
            schema=schema,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(
                tool="document-embed-upsert",
                collection="summaries",
            )],
            vector_collections=[vc],
        )
        await pipeline._ingest(Document(
            doc_id="d-001",
            text="transcript",
            metadata={"customer_id": "acme", "deal_stage": "negotiation", "internal": "skip"},
        ))
        chunks = await vector_store.search("summaries", "", [0.1] * 4, top_k=1)
        assert chunks[0].metadata == {"customer_id": "acme", "deal_stage": "negotiation"}

    @pytest.mark.asyncio
    async def test_empty_llm_response_skips_upsert(self, make_vector_store):
        vector_store = make_vector_store()
        await vector_store.create_collection(self._SUMMARIES_SCHEMA)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(return_value={"content": None, "tool_calls": None})
        vc = VectorCollection(
            schema=self._SUMMARIES_SCHEMA,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="document-embed-upsert", collection="summaries", llm=llm)],
            vector_collections=[vc],
        )
        await pipeline._ingest(Document(doc_id="d-001", text="text"))
        assert vector_store.ntotal("summaries") == 0

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_raise(self, make_vector_store):
        vector_store = make_vector_store()
        await vector_store.create_collection(self._SUMMARIES_SCHEMA)
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        vc = VectorCollection(
            schema=self._SUMMARIES_SCHEMA,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="document-embed-upsert", collection="summaries", llm=llm)],
            vector_collections=[vc],
        )
        count = await pipeline._ingest(Document(doc_id="d-001", text="text"))
        assert count == 0
        assert vector_store.ntotal("summaries") == 0


# ---------------------------------------------------------------------------
# Full 3-step pipeline: chunk + extract + summarize
# ---------------------------------------------------------------------------

class TestThreeStepPipeline:
    @pytest.mark.asyncio
    async def test_all_three_steps_execute(self, make_vector_store, make_structured_store):
        chunk_store = make_vector_store()
        summary_store = make_vector_store()
        struct_store = make_structured_store()

        vc_schema = VectorCollectionSchema(name="chunks", dimensions=4, description="Test chunks")
        dc_schema = VectorCollectionSchema(name="summaries", dimensions=4, description="Test document summaries")
        sc_schema = StubExtractor().schema
        await chunk_store.create_collection(vc_schema)
        await summary_store.create_collection(dc_schema)
        await struct_store.create_collection(sc_schema)

        vc_chunks = VectorCollection(schema=vc_schema, store=chunk_store, embedder=StubEmbedding(dim=4))
        vc_summary = VectorCollection(schema=dc_schema, store=summary_store, embedder=StubEmbedding(dim=4))
        sc = StructuredCollection(schema=sc_schema, store=struct_store)

        pipeline = IngestionPipeline(
            name="app",
            steps=[
                PipelineStep(tool="chunk-embed-upsert",    collection="chunks",    chunker=FixedSizeChunker(chunk_size=20, overlap=0)),
                PipelineStep(tool="extract-structured",    collection="tags",      extractor=StubExtractor()),
                PipelineStep(tool="document-embed-upsert", collection="summaries", llm=_make_llm("Short summary.")),
            ],
            vector_collections=[vc_chunks, vc_summary],
            structured_collections=[sc],
        )

        count = await pipeline._ingest(Document(doc_id="d-001", text="word " * 20))

        assert chunk_store.ntotal("chunks") > 0, "chunk-embed-upsert did not populate vector store"
        assert count == 1, "extract-structured did not produce a record"
        assert summary_store.ntotal("summaries") == 1, "document-embed-upsert did not upsert summary"

    @pytest.mark.asyncio
    async def test_structured_collections_queryable_after_creation(self, make_structured_store):
        struct_store_a = make_structured_store()
        struct_store_b = make_structured_store()

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

        sc_a = StructuredCollection(schema=ExtA().schema, store=struct_store_a)
        sc_b = StructuredCollection(schema=ExtB().schema, store=struct_store_b)

        await struct_store_a.create_collection(ExtA().schema)
        await struct_store_b.create_collection(ExtB().schema)

        rows_a = await struct_store_a.query("col_a")
        rows_b = await struct_store_b.query("col_b")
        assert rows_a == []
        assert rows_b == []


# ---------------------------------------------------------------------------
# Config: VectorCollectionConfig and step-level prompt
# ---------------------------------------------------------------------------

class TestVectorCollectionConfig:
    def test_parse_vector_collection_with_step_prompt(self):
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
            vector_collections:
              - name: document_summary
                description: One summary vector per document for topic-level search.
            pipelines:
              - name: main
                routing_description: Documents to embed as per-document summaries.
                steps:
                  - tool: document-embed-upsert
                    collection: document_summary
                    doc_prompt: "Summarize in one sentence."
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.vector_collections) == 1
        assert cfg.vector_collections[0].name == "document_summary"
        step = cfg.pipelines[0].steps[0]
        assert step.doc_prompt == "Summarize in one sentence."

    def test_vector_collection_without_embedding_is_valid(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            vector_collections:
              - name: document_summary
                description: One summary vector per document for topic-level search.
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.embedding is None
        assert len(cfg.vector_collections) == 1

    def test_unknown_vector_collection_in_step_raises(self):
        import textwrap
        from cogbase.config.config import AppConfig

        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: document_summary
                description: One summary vector per document for topic-level search.
            pipelines:
              - name: main
                routing_description: Documents to embed.
                steps:
                  - tool: document-embed-upsert
                    collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown vector collection"):
            AppConfig.from_yaml(yaml_text)


# ---------------------------------------------------------------------------
# metadata.doc_type routing via pipeline.match
# ---------------------------------------------------------------------------

def _make_vector_collection(
    store: VectorStoreBase, name: str, dim: int = 4
) -> VectorCollection:
    return VectorCollection(
        schema=VectorCollectionSchema(name=name, dimensions=dim, description=f"{name} chunks"),
        store=store,
        embedder=StubEmbedding(dim=dim),
    )


class TestWhenConditionRouting:
    @staticmethod
    def _make_app(pipelines: list[IngestionPipeline]) -> CogBaseApp:
        store = InMemoryStructuredStore()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value={"content": "ok", "tool_calls": None})
        runner = QueryRunner(llm=llm, structured_store=store)
        return CogBaseApp("app", pipelines, runner)

    @pytest.mark.asyncio
    async def test_matching_doc_type_runs_step(self, make_vector_store):
        """A rules document goes to rule_chunks (doc_type matches)."""
        rule_store = make_vector_store()
        rule_schema = VectorCollectionSchema(name="rule_chunks", dimensions=4, description="rule chunks")
        await rule_store.create_collection(rule_schema)

        vc = _make_vector_collection(rule_store, "rule_chunks")
        pipeline = IngestionPipeline(
            name="rules",
            match={"doc_type": "rules"},
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="rule_chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        app = self._make_app([pipeline])

        await app.ingest_documents([Document(
            doc_id="rules-001",
            text="Vendors must comply with ISO 27001.",
            metadata={"doc_type": "rules"},
        )])

        assert rule_store.ntotal("rule_chunks") > 0

    @pytest.mark.asyncio
    async def test_non_matching_doc_type_skips_step(self, make_vector_store):
        """A contract document is skipped by the rules step (doc_type mismatch)."""
        rule_store = make_vector_store()
        rule_schema = VectorCollectionSchema(name="rule_chunks", dimensions=4, description="rule chunks")
        await rule_store.create_collection(rule_schema)

        vc = _make_vector_collection(rule_store, "rule_chunks")
        pipeline = IngestionPipeline(
            name="rules",
            match={"doc_type": "rules"},
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="rule_chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        app = self._make_app([pipeline])

        results = await app.ingest_documents([Document(
            doc_id="contract-001",
            text="This agreement is entered into by the parties.",
            metadata={"doc_type": "contract"},
        )])

        assert results[0].success is False
        assert rule_store.ntotal("rule_chunks") == 0

    @pytest.mark.asyncio
    async def test_step_without_when_runs_for_all_docs(self, make_vector_store):
        """A step with no when condition runs regardless of doc metadata."""
        store = make_vector_store()
        schema = VectorCollectionSchema(name="all_chunks", dimensions=4, description="all chunks")
        await store.create_collection(schema)

        vc = _make_vector_collection(store, "all_chunks")
        pipeline = IngestionPipeline(
            name="app",
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="all_chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        app = self._make_app([pipeline])

        await app.ingest_documents([Document(doc_id="d1", text="rules text", metadata={"doc_type": "rules"})])
        await app.ingest_documents([Document(doc_id="d2", text="contract text", metadata={"doc_type": "contract"})])

        assert store.ntotal("all_chunks") > 0

    @pytest.mark.asyncio
    async def test_rules_and_contract_steps_route_independently(self, make_vector_store):
        """rules doc → rule_chunks only; contract doc → contract_chunks only."""
        rule_store = make_vector_store()
        contract_store = make_vector_store()

        rule_schema = VectorCollectionSchema(name="rule_chunks", dimensions=4, description="rule chunks")
        contract_schema = VectorCollectionSchema(name="contract_chunks", dimensions=4, description="contract chunks")
        await rule_store.create_collection(rule_schema)
        await contract_store.create_collection(contract_schema)

        rule_vc = _make_vector_collection(rule_store, "rule_chunks")
        contract_vc = _make_vector_collection(contract_store, "contract_chunks")

        rules_pipeline = IngestionPipeline(
            name="rules",
            match={"doc_type": "rules"},
            steps=[
                PipelineStep(tool="chunk-embed-upsert", collection="rule_chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0)),
            ],
            vector_collections=[rule_vc],
        )
        contract_pipeline = IngestionPipeline(
            name="contracts",
            match={"doc_type": "contract"},
            steps=[
                PipelineStep(tool="chunk-embed-upsert", collection="contract_chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0)),
            ],
            vector_collections=[contract_vc],
        )
        app = self._make_app([rules_pipeline, contract_pipeline])

        await app.ingest_documents([Document(
            doc_id="rules-001",
            text="ISO 27001 compliance required.",
            metadata={"doc_type": "rules"},
        )])
        await app.ingest_documents([Document(
            doc_id="contract-001",
            text="This agreement is between vendor and buyer.",
            metadata={"doc_type": "contract"},
        )])

        assert rule_store.ntotal("rule_chunks") > 0, "rules doc did not land in rule_chunks"
        assert contract_store.ntotal("contract_chunks") > 0, "contract doc did not land in contract_chunks"

    @pytest.mark.asyncio
    async def test_partial_metadata_match_skips_step(self, make_vector_store):
        """All when keys must match; partial match still skips the step."""
        store = make_vector_store()
        schema = VectorCollectionSchema(name="chunks", dimensions=4, description="chunks")
        await store.create_collection(schema)

        vc = _make_vector_collection(store, "chunks")
        pipeline = IngestionPipeline(
            name="app",
            match={"doc_type": "rules", "region": "us"},
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        app = self._make_app([pipeline])

        results = await app.ingest_documents([Document(
            doc_id="d1",
            text="Some rules.",
            metadata={"doc_type": "rules", "region": "eu"},
        )])

        assert results[0].success is False
        assert store.ntotal("chunks") == 0

    @pytest.mark.asyncio
    async def test_doc_missing_metadata_key_skips_step(self, make_vector_store):
        """A document that lacks the when key entirely is skipped."""
        store = make_vector_store()
        schema = VectorCollectionSchema(name="chunks", dimensions=4, description="chunks")
        await store.create_collection(schema)

        vc = _make_vector_collection(store, "chunks")
        pipeline = IngestionPipeline(
            name="app",
            match={"doc_type": "rules"},
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="chunks", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        app = self._make_app([pipeline])

        results = await app.ingest_documents([Document(doc_id="d1", text="Some text.", metadata={})])

        assert results[0].success is False
        assert store.ntotal("chunks") == 0
