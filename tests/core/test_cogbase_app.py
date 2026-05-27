"""Integration tests for CogBaseApp."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.config.config import RoutingStrategy
from cogbase.core.app import CogBaseApp
from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.stores.document.base import DocumentStoreBase
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    IngestResult,
    StructuredCollection,
    VectorCollection,
    PipelineStep,
)
from cogbase.core.models import Document
from cogbase.config.config import ExtractorConfig
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import LLMBase
from api.factory import _json_schema_to_collection_fields
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.stores import CollectionSchema
from cogbase.pipeline.chunking.fixed import FixedSizeChunker
from cogbase.stores import VectorCollectionSchema
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from examples.contract_analyst_demo.demo import _CONTRACTS_COLLECTION
from examples.contract_analyst_demo.schema import ContractExtraction


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_llm(content: str) -> MagicMock:
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value={"content": content})

    async def _stream(*args, **kwargs):
        yield content

    llm.complete_stream = _stream
    return llm


def _mock_task_store():
    m = MagicMock()
    m.create_workflow_task = AsyncMock(return_value=None)
    m.complete_workflow_task = AsyncMock()
    return m


async def _drain_query(app: CogBaseApp, text: str) -> QueryResult:
    async for item in app.query_stream(text):
        if not isinstance(item, str):
            return item
    raise AssertionError("query_stream did not yield a QueryResult")


def _contract_payload(**overrides) -> str:
    data = {
        "contract_type": "NDA",
        "purpose": "Test non-disclosure agreement.",
        "effective_date": None,
        "expiry_date": None,
        "parties": [
            {"name": "Acme Corp", "role": "discloser", "jurisdiction": None},
            {"name": "Supplier Ltd", "role": "recipient", "jurisdiction": None},
        ],
        "contract_value": None,
        "currency": None,
        "payment_terms": None,
        "termination": None,
        "liability": None,
        "governing_law": None,
        "confidentiality": None,
        "indemnification": None,
        "dispute_resolution": None,
        "notice_period_days": None,
        "liability_cap": None,
        "key_terms": [],
        "special_conditions": [],
    }
    data.update(overrides)
    return json.dumps(data)


class StubEmbedding(EmbeddingBase):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


_EXTRACTION_SCHEMA = ContractExtraction.model_json_schema()
_RECORD_SCHEMA = {
    **_EXTRACTION_SCHEMA,
    "properties": {**_EXTRACTION_SCHEMA.get("properties", {}), "doc_id": {"type": "string"}},
}

_CONTRACTS_SCHEMA = CollectionSchema(
    name=_CONTRACTS_COLLECTION,
    description="Extracted contract metadata: parties, dates, and governing law.",
    primary_fields=["doc_id"],
    fields=_json_schema_to_collection_fields(_RECORD_SCHEMA),
)


def _make_extractor(llm: MagicMock) -> LLMExtractor:
    return LLMExtractor(
        llm,
        extraction_schema=_EXTRACTION_SCHEMA,
        config=ExtractorConfig(
            extraction_schema='{"type":"object","properties":{"value":{"type":"string"}}}',
            prompt="Extract the relevant fields from the document.",
        ),
        record_schema=_RECORD_SCHEMA,
    )


async def _make_pipeline(
    llm: MagicMock,
    store: InMemoryStructuredStore,
    *,
    vector_store: FAISSVectorStore | None = None,
    embedder: StubEmbedding | None = None,
    chunker=None,
    name: str = "legal",
) -> IngestionPipeline:
    extractor = _make_extractor(llm)
    sc_schema = _CONTRACTS_SCHEMA
    await store.create_collection(sc_schema)
    sc = StructuredCollection(schema=sc_schema, store=store)

    steps = [PipelineStep(tool="extract-structured", collection=sc.name, extractor=extractor)]
    vector_collections = None
    if vector_store is not None:
        assert embedder is not None and chunker is not None
        vc_schema = VectorCollectionSchema(name=name, dimensions=4, description="Test chunks")
        await vector_store.create_collection(vc_schema)
        vc = VectorCollection(schema=vc_schema, store=vector_store, embedder=embedder)
        vector_collections = [vc]
        steps.insert(0, PipelineStep(tool="chunk-embed-upsert", collection=name, chunker=chunker))

    return IngestionPipeline(
        name=name,
        steps=steps,
        vector_collections=vector_collections,
        structured_collections=[sc],
    )


async def _make_app(
    llm: MagicMock,
    store: InMemoryStructuredStore,
    *,
    vector_store: FAISSVectorStore | None = None,
    embedder: StubEmbedding | None = None,
    chunker=None,
    name: str = "legal",
) -> CogBaseApp:
    pipeline = await _make_pipeline(llm, store, vector_store=vector_store, embedder=embedder, chunker=chunker, name=name)
    runner = QueryRunner(
        llm=llm,
        structured_store=store,
        vector_store=vector_store,
        embedder=embedder,
        vector_schemas=[c.schema for c in pipeline._vector_by_name.values()] or None,
        structured_schemas=[sc.schema for sc in pipeline._structured_by_name.values()] or None,
    )
    return CogBaseApp(
        name, [pipeline], runner,
        document_store=InMemoryDocumentStore(),
        structured_store=store,
        workflow_runners={},
        llm=llm,
        task_store=_mock_task_store(),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCogBaseAppConstruction:
    async def test_structured_only_builds(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert app._pipelines[0].name == "legal"
        assert app._pipelines[0]._structured_by_name
        assert _CONTRACTS_COLLECTION in app._pipelines[0]._structured_by_name
        assert app._pipelines[0]._vector_by_name == {}

    async def test_full_mode_builds(self):
        app = await _make_app(
            _make_llm("{}"),
            InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        assert app._pipelines[0]._vector_by_name
        assert "legal" in app._pipelines[0]._vector_by_name

    async def test_pipeline_wired_to_app(self):
        store = InMemoryStructuredStore()
        pipeline = await _make_pipeline(_make_llm("{}"), store)
        runner = QueryRunner(llm=_make_llm("{}"), structured_store=store, structured_schemas=[sc.schema for sc in pipeline._structured_by_name.values()] or None)
        llm = _make_llm("{}")
        app = CogBaseApp(
            "test", [pipeline], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=store,
            workflow_runners={},
            llm=llm,
            task_store=_mock_task_store(),
        )
        assert app._pipelines[0] is pipeline

    async def test_custom_name(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore(), name="my-legal-app")
        assert app._pipelines[0].name == "my-legal-app"

    async def test_ingestion_pipeline_and_query_runner_accessible(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert isinstance(app.ingestion_pipelines[0], IngestionPipeline)
        assert isinstance(app.query_runner, QueryRunner)


# ---------------------------------------------------------------------------
# ingest_documents()
# ---------------------------------------------------------------------------

class TestCogBaseAppLifecycle:
    @pytest.mark.asyncio
    async def test_ingest_extracts_record(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload(contract_type="SaaS")), store)
        await app.ingest_documents([Document(doc_id="c-001", text="Some contract text.")])
        rows = await store.query(_CONTRACTS_COLLECTION)
        assert len(rows) == 1
        assert rows[0]["contract_type"] == "SaaS"

    @pytest.mark.asyncio
    async def test_ingest_empty_text_is_noop(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm("{}"), store)
        await app.ingest_documents([Document(doc_id="c-empty", text="")])
        rows = await store.query(_CONTRACTS_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)
        await app.ingest_documents([
            Document(doc_id="c-001", text="contract one text"),
            Document(doc_id="c-002", text="contract two text"),
        ])
        rows = await store.query(_CONTRACTS_COLLECTION)
        assert len(rows) == 2
        doc_ids = {r["doc_id"] for r in rows}
        assert doc_ids == {"c-001", "c-002"}

    @pytest.mark.asyncio
    async def test_ingest_full_mode_populates_vector_store(self):
        store = InMemoryStructuredStore()
        vector_store = FAISSVectorStore()
        app = await _make_app(
            _make_llm("{}"),
            store,
            vector_store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        await app.ingest_documents([Document(doc_id="c-001", text="word " * 20)])
        assert vector_store.ntotal("legal") > 0


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestCogBaseAppQuery:
    async def _make_app_with_runner_response(
        self,
        runner_response: dict,
        *,
        extractor_json: str = "{}",
    ) -> tuple[CogBaseApp, InMemoryStructuredStore]:
        store = InMemoryStructuredStore()
        llm = MagicMock(spec=LLMBase)

        async def _complete(messages, **kwargs):
            system_content = messages[0].get("content", "") if messages else ""
            if "extract structured" in system_content.lower():
                return extractor_json
            return runner_response

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(messages, **kwargs):
            yield runner_response.get("content") or ""

        llm.complete_stream = _stream
        app = await _make_app(llm, store)
        return app, store

    @pytest.mark.asyncio
    async def test_direct_answer_returns_query_result(self):
        app, store = await self._make_app_with_runner_response(
            {"content": "The termination notice period is 60 days.", "tool_calls": None},
        )
        result = await _drain_query(app, "what is the termination notice period?")
        assert isinstance(result, QueryResult)
        assert "60 days" in result.answer

    @pytest.mark.asyncio
    async def test_structured_lookup_populates_records(self):
        store = InMemoryStructuredStore()
        llm = MagicMock(spec=LLMBase)
        stream_call_count = 0

        async def _stream(messages, **kwargs):
            nonlocal stream_call_count
            stream_call_count += 1
            if stream_call_count == 1:
                yield {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "name": "structured_lookup",
                        "arguments": json.dumps({"collection": _CONTRACTS_COLLECTION, "filters": []}),
                    }],
                }
            else:
                yield "Found NDA contracts: Acme Corp and Supplier Ltd."

        llm.complete_stream = _stream
        app = await _make_app(llm, store)

        record = {
            "contract_id": "c-001_abc",
            "doc_id": "c-001",
            "contract_type": "NDA",
            "parties": [
                {"name": "Acme Corp", "role": "discloser"},
                {"name": "Supplier Ltd", "role": "recipient"},
            ],
            "payment_terms": {"schedule": "net-30", "verbatim": "Payment is due within 30 days."},
        }
        await store.save(_CONTRACTS_COLLECTION, [record])

        result = await _drain_query(app, "list NDA contracts")
        assert isinstance(result, QueryResult)
        assert len(result.structured_records) > 0
        assert result.passthrough is False

    @pytest.mark.asyncio
    async def test_answer_content_captured(self):
        app, store = await self._make_app_with_runner_response(
            {"content": "Both parties have broad indemnification obligations.", "tool_calls": None},
        )
        result = await _drain_query(app, "summarise indemnification clauses")
        assert isinstance(result, QueryResult)
        assert "indemnification" in result.answer.lower()


# ---------------------------------------------------------------------------
# Tool availability — structured-only vs full mode
# ---------------------------------------------------------------------------

class TestQueryRunnerToolAvailability:
    def _tool_names(self, app: CogBaseApp) -> list[str]:
        return [t["name"] for t in app.query_runner._tool_defs]

    async def test_structured_only_has_no_vector_search(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert "vector_search" not in self._tool_names(app)

    async def test_structured_only_has_structured_lookup(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert "structured_lookup" in self._tool_names(app)

    async def test_full_mode_has_both_tools(self):
        app = await _make_app(
            _make_llm("{}"),
            InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        names = self._tool_names(app)
        assert "structured_lookup" in names
        assert "vector_search" in names

    @pytest.mark.asyncio
    async def test_structured_only_query_returns_result(self):
        llm = MagicMock(spec=LLMBase)

        async def _complete(messages, **kwargs):
            return {"content": "The answer.", "tool_calls": None}

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(messages, **kwargs):
            yield "The answer."

        llm.complete_stream = _stream

        app = await _make_app(llm, InMemoryStructuredStore())
        result = await _drain_query(app, "what is the termination clause?")
        assert isinstance(result, QueryResult)


# ---------------------------------------------------------------------------
# Vector-only mode (no structured collection)
# ---------------------------------------------------------------------------

class TestVectorOnlyMode:
    """CogBaseApp with no structured collection skips extraction entirely."""

    async def _make_vector_only_app(self, llm: MagicMock) -> CogBaseApp:
        vc_store = FAISSVectorStore()
        vc_embedder = StubEmbedding(dim=4)
        vc_schema = VectorCollectionSchema(name="vector_only", dimensions=4, description="Test vector-only chunks")
        await vc_store.create_collection(vc_schema)
        vc = VectorCollection(
            schema=vc_schema,
            store=vc_store,
            embedder=vc_embedder,
        )
        pipeline = IngestionPipeline(
            name="vector-only",
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="vector_only", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        runner = QueryRunner(
            llm=llm,
            vector_store=vc_store,
            embedder=vc_embedder,
            vector_schemas=[c.schema for c in pipeline._vector_by_name.values()] or None,
        )
        return CogBaseApp(
            "vector-only", [pipeline], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=llm,
            task_store=_mock_task_store(),
        )

    def _tool_names(self, app: CogBaseApp) -> list[str]:
        return [t["name"] for t in app.query_runner._tool_defs]

    async def test_no_structured_collection(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert app._pipelines[0]._structured_by_name == {}

    async def test_vector_collection_present(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert app._pipelines[0]._vector_by_name

    async def test_no_structured_lookup_tool(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert "structured_lookup" not in self._tool_names(app)

    async def test_has_vector_search_tool(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert "vector_search" in self._tool_names(app)

    @pytest.mark.asyncio
    async def test_ingest_populates_vector_store_not_structured(self):
        vector_store = FAISSVectorStore()
        vc_schema = VectorCollectionSchema(name="testapp", dimensions=4, description="Test chunks")
        await vector_store.create_collection(vc_schema)
        vc = VectorCollection(
            schema=vc_schema,
            store=vector_store,
            embedder=StubEmbedding(dim=4),
        )
        pipeline = IngestionPipeline(
            name="testapp",
            steps=[PipelineStep(tool="chunk-embed-upsert", collection="testapp", chunker=FixedSizeChunker(chunk_size=20, overlap=0))],
            vector_collections=[vc],
        )
        runner = QueryRunner(llm=_make_llm("{}"), vector_store=vector_store, embedder=StubEmbedding(dim=4), vector_schemas=[c.schema for c in pipeline._vector_by_name.values()] or None)
        app = CogBaseApp(
            "testapp", [pipeline], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=_make_llm("{}"),
            task_store=_mock_task_store(),
        )
        results = await app.ingest_documents([Document(doc_id="d-001", text="word " * 20)])
        assert results[0].success is True
        assert results[0].records_extracted == 0
        assert vector_store.ntotal("testapp") > 0

    @pytest.mark.asyncio
    async def test_ingest_records_extracted_is_zero(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        results = await app.ingest_documents([Document(doc_id="d-001", text="some text " * 5)])
        assert results[0].records_extracted == 0


# ---------------------------------------------------------------------------
# ingest_documents()
# ---------------------------------------------------------------------------

class TestIngestMany:
    @pytest.mark.asyncio
    async def test_returns_one_result_per_document(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        documents = [
            Document(doc_id="c-001", text="contract one"),
            Document(doc_id="c-002", text="contract two"),
            Document(doc_id="c-003", text="contract three"),
        ]
        results = await app.ingest_documents(documents)

        assert len(results) == 3
        assert all(isinstance(r, IngestResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_in_input_order(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm("{}"), store)

        doc_ids = [f"c-{i:03d}" for i in range(8)]
        documents = [Document(doc_id=d, text=f"text for {d}") for d in doc_ids]
        results = await app.ingest_documents(documents)

        assert [r.doc_id for r in results] == doc_ids

    @pytest.mark.asyncio
    async def test_success_flag_set(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        results = await app.ingest_documents([Document(doc_id="c-001", text="some text")])

        assert results[0].success is True
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_records_extracted_count(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        results = await app.ingest_documents([Document(doc_id="c-001", text="contract text")])

        assert results[0].records_extracted == 1

    @pytest.mark.asyncio
    async def test_failure_captured_not_raised(self):
        store = InMemoryStructuredStore()
        call_n = 0

        llm = MagicMock(spec=LLMBase)

        async def _complete(messages, **kwargs):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise RuntimeError("LLM unavailable")
            return {"content": _contract_payload(), "tool_calls": None}

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(*args, **kwargs):
            yield ""

        llm.complete_stream = _stream

        app = await _make_app(llm, store)

        results = await app.ingest_documents(
            [
                Document(doc_id="c-fail", text="will fail"),
                Document(doc_id="c-ok",   text="will succeed"),
            ],
        )

        failed = [r for r in results if not r.success]
        succeeded = [r for r in results if r.success]

        assert len(failed) == 1
        assert failed[0].doc_id == "c-fail"
        assert isinstance(failed[0].error, RuntimeError)

        assert len(succeeded) == 1
        assert succeeded[0].doc_id == "c-ok"
        assert succeeded[0].records_extracted == 1

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm("{}"), store)

        results = await app.ingest_documents([])
        assert results == []

    @pytest.mark.asyncio
    async def test_document_store_save_failure_recorded(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        doc_store = MagicMock(spec=DocumentStoreBase)
        doc_store.save = AsyncMock(side_effect=IOError("disk full"))
        app._document_store = doc_store

        results = await app.ingest_documents([Document(doc_id="c-001", text="text")])

        assert len(results) == 1
        assert results[0].success is False
        assert isinstance(results[0].error, IOError)

    @pytest.mark.asyncio
    async def test_document_store_partial_failure_skips_failed_doc(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        save_calls: list[str] = []

        async def _save(collection: str, doc_id: str, content: str) -> None:
            if doc_id == "c-fail":
                raise IOError("disk full")
            save_calls.append(doc_id)

        doc_store = MagicMock(spec=DocumentStoreBase)
        doc_store.save = AsyncMock(side_effect=_save)
        app._document_store = doc_store

        results = await app.ingest_documents([
            Document(doc_id="c-fail", text="will fail"),
            Document(doc_id="c-ok",   text="will succeed"),
        ])

        assert len(results) == 2
        failed = [r for r in results if not r.success]
        succeeded = [r for r in results if r.success]
        assert len(failed) == 1
        assert failed[0].doc_id == "c-fail"
        assert isinstance(failed[0].error, IOError)
        assert len(succeeded) == 1
        assert succeeded[0].doc_id == "c-ok"
        assert "c-fail" not in save_calls

    @pytest.mark.asyncio
    async def test_document_store_order_preserved_with_failure(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm(_contract_payload()), store)

        async def _save(collection: str, doc_id: str, content: str) -> None:
            if doc_id == "c-002":
                raise IOError("disk full")

        doc_store = MagicMock(spec=DocumentStoreBase)
        doc_store.save = AsyncMock(side_effect=_save)
        app._document_store = doc_store

        results = await app.ingest_documents([
            Document(doc_id="c-001", text="text 1"),
            Document(doc_id="c-002", text="text 2"),
            Document(doc_id="c-003", text="text 3"),
        ])

        assert [r.doc_id for r in results] == ["c-001", "c-002", "c-003"]
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# ---------------------------------------------------------------------------
# RoutingStrategy.AUTO — metadata back-fill after LLM fallback
# ---------------------------------------------------------------------------

class TestRoutingStrategyAuto:
    """RoutingStrategy.AUTO: metadata match skips LLM; LLM fallback back-fills doc.metadata."""

    @staticmethod
    def _pipeline_stub(name: str, match: dict[str, str] | None) -> MagicMock:
        p = MagicMock(spec=IngestionPipeline)
        p.name = name
        p.match = match
        p.description = name

        async def _ingest(docs):
            return [IngestResult(doc_id=d.doc_id, success=True) for d in docs]

        p.ingest_documents = AsyncMock(side_effect=_ingest)
        return p

    @staticmethod
    def _make_app(llm_response: str, *pipelines: MagicMock) -> CogBaseApp:
        llm = MagicMock(spec=LLMBase)
        llm.complete = AsyncMock(return_value={"content": llm_response})
        return CogBaseApp(
            "test",
            list(pipelines),
            MagicMock(spec=QueryRunner),
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=llm,
            routing_strategy=RoutingStrategy.AUTO,
            task_store=_mock_task_store(),
        )

    @pytest.mark.asyncio
    async def test_metadata_match_routes_without_llm(self):
        legal = self._pipeline_stub("legal", {"doc_type": "legal"})
        finance = self._pipeline_stub("finance", {"doc_type": "finance"})
        app = self._make_app("finance", legal, finance)

        doc = Document(doc_id="d-001", text="text", metadata={"doc_type": "legal"})
        await app.ingest_documents([doc])

        app._llm.complete.assert_not_called()
        legal.ingest_documents.assert_called_once()
        finance.ingest_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_fallback_called_when_metadata_misses(self):
        legal = self._pipeline_stub("legal", {"doc_type": "legal"})
        finance = self._pipeline_stub("finance", {"doc_type": "finance"})
        app = self._make_app("finance", legal, finance)

        doc = Document(doc_id="d-001", text="quarterly earnings report")
        await app.ingest_documents([doc])

        app._llm.complete.assert_called_once()
        finance.ingest_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_backfills_match_conditions_into_metadata(self):
        legal = self._pipeline_stub("legal", {"doc_type": "legal"})
        finance = self._pipeline_stub("finance", {"doc_type": "finance"})
        app = self._make_app("finance", legal, finance)

        doc = Document(doc_id="d-001", text="quarterly earnings report")
        await app.ingest_documents([doc])

        assert doc.metadata.get("doc_type") == "finance"

    @pytest.mark.asyncio
    async def test_backfill_does_not_overwrite_existing_metadata(self):
        legal = self._pipeline_stub("legal", {"doc_type": "legal"})
        finance = self._pipeline_stub("finance", {"doc_type": "finance"})
        app = self._make_app("finance", legal, finance)

        doc = Document(doc_id="d-001", text="text", metadata={"doc_type": "custom"})
        await app.ingest_documents([doc])

        assert doc.metadata["doc_type"] == "custom"

    @pytest.mark.asyncio
    async def test_match_none_pipeline_does_not_short_circuit_metadata_routing(self):
        """A pipeline with match=None must not be returned by metadata routing.

        Before the fix, _find_pipeline_by_metadata returned the first pipeline
        whose match was None, preventing LLM fallback from ever running in
        multi-pipeline apps where no pipeline declared match conditions.
        """
        no_match = self._pipeline_stub("no_match", None)
        specific = self._pipeline_stub("specific", {"doc_type": "legal"})
        app = self._make_app("specific", no_match, specific)

        doc = Document(doc_id="d-001", text="legal document text")
        await app.ingest_documents([doc])

        # Metadata routing skips match=None; LLM picks "specific".
        app._llm.complete.assert_called_once()
        specific.ingest_documents.assert_called_once()
        no_match.ingest_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_match_none_pipelines_use_llm_routing(self):
        """When every pipeline has match=None, LLM routing decides."""
        alpha = self._pipeline_stub("alpha", None)
        beta = self._pipeline_stub("beta", None)
        app = self._make_app("beta", alpha, beta)

        doc = Document(doc_id="d-001", text="some document text")
        await app.ingest_documents([doc])

        app._llm.complete.assert_called_once()
        beta.ingest_documents.assert_called_once()
        alpha.ingest_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_unmatched_error_names_pipelines_and_suggests_action(self):
        """Unmatched documents get an error message listing tried pipelines and next steps."""
        legal = self._pipeline_stub("legal", {"doc_type": "legal"})
        finance = self._pipeline_stub("finance", {"doc_type": "finance"})
        # LLM returns an unknown name so both routing strategies fail.
        app = self._make_app("unknown_pipeline", legal, finance)

        doc = Document(doc_id="d-001", text="unclassifiable text")
        results = await app.ingest_documents([doc])

        assert len(results) == 1
        result = results[0]
        assert not result.success
        msg = str(result.error)
        assert "d-001" in msg
        assert "legal" in msg and "finance" in msg
        assert "routing_description" in msg


# ---------------------------------------------------------------------------
# query_prompt — stored and forwarded to runner.run as base_prompt
# ---------------------------------------------------------------------------

class TestQueryPrompt:
    def _make_tracking_runner(self) -> tuple[MagicMock, list[dict]]:
        """Return a runner mock that records every run() call's kwargs."""
        calls: list[dict] = []
        runner = MagicMock(spec=QueryRunner)

        async def _run(text, **kwargs):
            calls.append(kwargs)
            yield QueryResult(answer="ok")

        runner.run = _run
        return runner, calls

    def _make_app_with_prompt(self, query_prompt: str | None) -> CogBaseApp:
        runner, _ = self._make_tracking_runner()
        return CogBaseApp(
            "test", [], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=_make_llm(""),
            task_store=_mock_task_store(),
            query_prompt=query_prompt,
        )

    def test_query_prompt_stored_on_app(self):
        app = self._make_app_with_prompt("Be concise.")
        assert app._query_prompt == "Be concise."

    def test_no_query_prompt_stored_as_none(self):
        app = self._make_app_with_prompt(None)
        assert app._query_prompt is None

    @pytest.mark.asyncio
    async def test_custom_prompt_passed_as_base_prompt(self):
        calls: list[dict] = []
        runner = MagicMock(spec=QueryRunner)

        async def _run(text, **kwargs):
            calls.append(kwargs)
            yield QueryResult(answer="ok")

        runner.run = _run
        app = CogBaseApp(
            "test", [], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=_make_llm(""),
            task_store=_mock_task_store(),
            query_prompt="Answer in one sentence.",
        )
        await _drain_query(app, "what is X?")
        assert calls[0].get("base_prompt") == "Answer in one sentence."

    @pytest.mark.asyncio
    async def test_no_prompt_omits_base_prompt_kwarg(self):
        calls: list[dict] = []
        runner = MagicMock(spec=QueryRunner)

        async def _run(text, **kwargs):
            calls.append(kwargs)
            yield QueryResult(answer="ok")

        runner.run = _run
        app = CogBaseApp(
            "test", [], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=_make_llm(""),
            task_store=_mock_task_store(),
            query_prompt=None,
        )
        await _drain_query(app, "what is X?")
        assert "base_prompt" not in calls[0]

    @pytest.mark.asyncio
    async def test_history_forwarded_alongside_custom_prompt(self):
        calls: list[dict] = []
        runner = MagicMock(spec=QueryRunner)

        async def _run(text, **kwargs):
            calls.append(kwargs)
            yield QueryResult(answer="ok")

        runner.run = _run
        app = CogBaseApp(
            "test", [], runner,
            document_store=InMemoryDocumentStore(),
            structured_store=InMemoryStructuredStore(),
            workflow_runners={},
            llm=_make_llm(""),
            task_store=_mock_task_store(),
            query_prompt="Be precise.",
        )
        history = [{"role": "user", "content": "prior turn"}]
        async for _ in app.query_stream("follow-up?", history=history):
            pass
        assert calls[0].get("base_prompt") == "Be precise."
        assert calls[0].get("history") == history
