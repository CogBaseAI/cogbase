"""Integration tests for CogBaseApp."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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
from cogbase.core.basemodel_to_schema import cls_generate_schema
from cogbase.pipeline.extraction.llm import LLMExtractor, _build_record_model
from cogbase.stores import CollectionSchema
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores import VectorCollectionSchema
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from examples.contract_analyst_demo.demo import _CONTRACTS_COLLECTION
from examples.contract_analyst_demo.schema import (
    ContractExtraction,
    Party,
    PaymentTerms,
)


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


_CONTRACTS_SCHEMA = CollectionSchema(
    name=_CONTRACTS_COLLECTION,
    description="Extracted contract metadata: parties, dates, and governing law.",
    primary_fields=["doc_id"],
    fields=cls_generate_schema(_build_record_model(ContractExtraction)),
)


def _make_extractor(llm: MagicMock) -> LLMExtractor:
    return LLMExtractor(
        llm,
        extraction_model=ContractExtraction,
        config=ExtractorConfig(extraction_schema='{"type":"object","properties":{"value":{"type":"string"}}}'),
        record_model=_build_record_model(ContractExtraction),
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
    return CogBaseApp(name, pipeline, runner)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCogBaseAppConstruction:
    async def test_structured_only_builds(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert app._ingest_pipeline.name == "legal"
        assert app._ingest_pipeline._structured_by_name
        assert _CONTRACTS_COLLECTION in app._ingest_pipeline._structured_by_name
        assert app._ingest_pipeline._vector_by_name == {}

    async def test_full_mode_builds(self):
        app = await _make_app(
            _make_llm("{}"),
            InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        assert app._ingest_pipeline._vector_by_name
        assert "legal" in app._ingest_pipeline._vector_by_name

    async def test_pipeline_wired_to_app(self):
        store = InMemoryStructuredStore()
        pipeline = await _make_pipeline(_make_llm("{}"), store)
        runner = QueryRunner(llm=_make_llm("{}"), structured_store=store, structured_schemas=[sc.schema for sc in pipeline._structured_by_name.values()] or None)
        app = CogBaseApp("test", pipeline, runner)
        assert app._ingest_pipeline is pipeline

    async def test_custom_name(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore(), name="my-legal-app")
        assert app._ingest_pipeline.name == "my-legal-app"

    async def test_ingestion_pipeline_and_query_runner_accessible(self):
        app = await _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert isinstance(app.ingestion_pipeline, IngestionPipeline)
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
        call_count = 0

        async def _complete(messages, **kwargs):
            nonlocal call_count
            system_content = messages[0].get("content", "") if messages else ""
            if "extract structured" in system_content.lower():
                return "{}"
            call_count += 1
            if call_count == 1:
                return {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "name": "structured_lookup",
                        "arguments": json.dumps({"collection": _CONTRACTS_COLLECTION, "filters": []}),
                    }],
                }
            return {"content": "Found NDA contracts: Acme Corp and Supplier Ltd.", "tool_calls": None}

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(messages, **kwargs):
            yield "Found NDA contracts: Acme Corp and Supplier Ltd."

        llm.complete_stream = _stream
        app = await _make_app(llm, store)

        extractor = _make_extractor(_make_llm("{}"))
        record_model = extractor._record_model
        record = record_model(
            contract_id="c-001_abc",
            doc_id="c-001",
            contract_type="NDA",
            parties=[Party(name="Acme Corp", role="discloser"), Party(name="Supplier Ltd", role="recipient")],
            payment_terms=PaymentTerms(schedule="net-30", verbatim="Payment is due within 30 days."),
        )
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
        return CogBaseApp("vector-only", pipeline, runner)

    def _tool_names(self, app: CogBaseApp) -> list[str]:
        return [t["name"] for t in app.query_runner._tool_defs]

    async def test_no_structured_collection(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert app._ingest_pipeline._structured_by_name == {}

    async def test_vector_collection_present(self):
        app = await self._make_vector_only_app(_make_llm("{}"))
        assert app._ingest_pipeline._vector_by_name

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
        app = CogBaseApp("testapp", pipeline, runner)
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
        results = await app.ingest_documents(documents, concurrency=3)

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
            concurrency=1,
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

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        store = InMemoryStructuredStore()
        app = await _make_app(_make_llm("{}"), store)

        with pytest.raises(ValueError, match="concurrency"):
            await app.ingest_documents([], concurrency=0)

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        store = InMemoryStructuredStore()
        active = 0
        peak = 0
        lock = asyncio.Lock()

        llm = MagicMock(spec=LLMBase)

        async def _complete(messages, **kwargs):
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0)
            async with lock:
                active -= 1
            return "{}"

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(*args, **kwargs):
            yield ""

        llm.complete_stream = _stream

        app = await _make_app(llm, store)

        documents = [Document(doc_id=f"c-{i}", text="text") for i in range(10)]
        await app.ingest_documents(documents, concurrency=3)

        assert peak <= 3
