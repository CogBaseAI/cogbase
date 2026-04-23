"""Integration tests for CogBaseApp."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.app import CogBaseApp
from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import Document
from cogbase.embeddings import EmbeddingBase
from cogbase.engine.query_runner import QueryResult, QueryRunner
from cogbase.llms.base import LLMBase
from cogbase.pipeline.extraction.llm import LLMExtractor
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from examples.contract_analyst_demo.schema import (
    CONTRACTS_COLLECTION,
    ContractExtraction,
    Party,
    PaymentTerms,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_llm(content: str) -> MagicMock:
    """Build a mock LLMBase returning *content* for complete() (extractor usage)."""
    llm = MagicMock(spec=LLMBase)
    llm.complete = AsyncMock(return_value=content)

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


def _make_extractor(llm: MagicMock) -> LLMExtractor:
    return LLMExtractor(
        llm,
        extraction_model=ContractExtraction,
        collection_name=CONTRACTS_COLLECTION,
    )


def _make_app(
    llm: MagicMock,
    store: InMemoryStructuredStore,
    *,
    vector_store: FAISSVectorStore | None = None,
    embedder: StubEmbedding | None = None,
    chunker=None,
    name: str = "legal",
) -> CogBaseApp:
    extractor = _make_extractor(llm)
    return CogBaseApp(
        name=name,
        llm=llm,
        extractor=extractor,
        structured_store=store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCogBaseAppConstruction:
    def test_structured_only_builds(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert app._ingest_pipeline.name == "legal"
        assert app._ingest_pipeline.structured_collection is not None
        assert app._ingest_pipeline.structured_collection.name == CONTRACTS_COLLECTION
        assert app._ingest_pipeline.vector_collection is None

    def test_full_mode_builds(self):
        app = _make_app(
            _make_llm("{}"),
            InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        assert app._ingest_pipeline.vector_collection is not None
        assert app._ingest_pipeline.vector_collection.name == "legal"

    def test_partial_vector_params_raises(self):
        llm = _make_llm("{}")
        extractor = _make_extractor(llm)
        with pytest.raises(ValueError, match="all be provided together"):
            CogBaseApp(
                name='testapp',
                llm=llm,
                extractor=extractor,
                structured_store=InMemoryStructuredStore(),
                vector_store=FAISSVectorStore(dim=4),
                # embedder and chunker missing
            )

    def test_custom_name(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore(), name="my-legal-app")
        assert app._ingest_pipeline.name == "my-legal-app"

    def test_structured_schemas_exposed(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore())
        schemas = app.structured_schemas
        assert len(schemas) == 1
        assert schemas[0].name == CONTRACTS_COLLECTION

    def test_ingestion_pipeline_and_query_runner_accessible(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert isinstance(app.ingestion_pipeline, IngestionPipeline)
        assert isinstance(app.query_runner, QueryRunner)


# ---------------------------------------------------------------------------
# setup() / ingest_documents()
# ---------------------------------------------------------------------------

class TestCogBaseAppLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_collection(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm("{}"), store)
        await app.setup()
        rows = await store.query(CONTRACTS_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_setup_idempotent(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm("{}"), store)
        await app.setup()
        await app.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_ingest_extracts_record(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm(_contract_payload(contract_type="SaaS")), store)
        await app.setup()
        await app.ingest_documents([Document(doc_id="c-001", text="Some contract text.")])
        rows = await store.query(CONTRACTS_COLLECTION)
        assert len(rows) == 1
        assert rows[0]["contract_type"] == "SaaS"

    @pytest.mark.asyncio
    async def test_ingest_empty_text_is_noop(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm("{}"), store)
        await app.setup()
        await app.ingest_documents([Document(doc_id="c-empty", text="")])
        rows = await store.query(CONTRACTS_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm(_contract_payload()), store)
        await app.setup()
        await app.ingest_documents([
            Document(doc_id="c-001", text="contract one text"),
            Document(doc_id="c-002", text="contract two text"),
        ])
        rows = await store.query(CONTRACTS_COLLECTION)
        assert len(rows) == 2
        doc_ids = {r["doc_id"] for r in rows}
        assert doc_ids == {"c-001", "c-002"}

    @pytest.mark.asyncio
    async def test_ingest_full_mode_populates_vector_store(self):
        store = InMemoryStructuredStore()
        vector_store = FAISSVectorStore(dim=4)
        app = _make_app(
            _make_llm("{}"),
            store,
            vector_store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        await app.setup()
        await app.ingest_documents([Document(doc_id="c-001", text="word " * 20)])
        assert vector_store.ntotal > 0


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestCogBaseAppQuery:
    def _make_app_with_runner_response(
        self,
        runner_response: dict,
        *,
        extractor_json: str = "{}",
    ) -> tuple[CogBaseApp, InMemoryStructuredStore]:
        """Build app where QueryRunner LLM always returns *runner_response* (a dict)."""
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
        app = _make_app(llm, store)
        return app, store

    @pytest.mark.asyncio
    async def test_direct_answer_returns_query_result(self):
        app, store = self._make_app_with_runner_response(
            {"content": "The termination notice period is 60 days.", "tool_calls": None},
        )
        await app.setup()
        result = await _drain_query(app, "what is the termination notice period?")
        assert isinstance(result, QueryResult)
        assert "60 days" in result.answer

    @pytest.mark.asyncio
    async def test_structured_lookup_populates_records(self):
        """LLM calls structured_lookup; records from store appear in QueryResult."""
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
                        "arguments": json.dumps({"collection": CONTRACTS_COLLECTION, "filters": []}),
                    }],
                }
            return {"content": "Found NDA contracts: Acme Corp and Supplier Ltd.", "tool_calls": None}

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(messages, **kwargs):
            yield "Found NDA contracts: Acme Corp and Supplier Ltd."

        llm.complete_stream = _stream
        app = _make_app(llm, store)
        await app.setup()

        extractor = _make_extractor(_make_llm("{}"))
        record_model = extractor._record_model
        record = record_model(
            contract_id="c-001_abc",
            doc_id="c-001",
            contract_type="NDA",
            parties=[Party(name="Acme Corp", role="discloser"), Party(name="Supplier Ltd", role="recipient")],
            payment_terms=PaymentTerms(schedule="net-30", verbatim="Payment is due within 30 days."),
        )
        await store.save(CONTRACTS_COLLECTION, [record])

        result = await _drain_query(app, "list NDA contracts")
        assert isinstance(result, QueryResult)
        assert len(result.structured_records) > 0
        assert result.passthrough is False

    @pytest.mark.asyncio
    async def test_answer_content_captured(self):
        app, store = self._make_app_with_runner_response(
            {"content": "Both parties have broad indemnification obligations.", "tool_calls": None},
        )
        await app.setup()
        result = await _drain_query(app, "summarise indemnification clauses")
        assert isinstance(result, QueryResult)
        assert "indemnification" in result.answer.lower()


# ---------------------------------------------------------------------------
# Tool availability — structured-only vs full mode
# ---------------------------------------------------------------------------

class TestQueryRunnerToolAvailability:
    def _tool_names(self, app: CogBaseApp) -> list[str]:
        return [t["name"] for t in app.query_runner._tool_defs]

    def test_structured_only_has_no_vector_search(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert "vector_search" not in self._tool_names(app)

    def test_structured_only_has_structured_lookup(self):
        app = _make_app(_make_llm("{}"), InMemoryStructuredStore())
        assert "structured_lookup" in self._tool_names(app)

    def test_full_mode_has_both_tools(self):
        app = _make_app(
            _make_llm("{}"),
            InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(dim=4),
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

        app = _make_app(llm, InMemoryStructuredStore())
        await app.setup()
        result = await _drain_query(app, "what is the termination clause?")
        assert isinstance(result, QueryResult)


# ---------------------------------------------------------------------------
# Vector-only mode (structured_store=None)
# ---------------------------------------------------------------------------

class TestVectorOnlyMode:
    """CogBaseApp with structured_store=None skips extraction entirely."""

    def _make_vector_only_app(self, llm: MagicMock) -> CogBaseApp:
        return CogBaseApp(
            name="vector-only",
            llm=llm,
            extractor=None,
            structured_store=None,
            vector_store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )

    def _tool_names(self, app: CogBaseApp) -> list[str]:
        return [t["name"] for t in app.query_runner._tool_defs]

    def test_no_structured_collection(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        assert app._ingest_pipeline.structured_collection is None

    def test_vector_collection_present(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        assert app._ingest_pipeline.vector_collection is not None

    def test_structured_schemas_empty(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        assert app.structured_schemas == []

    def test_no_structured_lookup_tool(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        assert "structured_lookup" not in self._tool_names(app)

    def test_has_vector_search_tool(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        assert "vector_search" in self._tool_names(app)

    @pytest.mark.asyncio
    async def test_setup_is_noop(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        await app.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_ingest_populates_vector_store_not_structured(self):
        vector_store = FAISSVectorStore(dim=4)
        app = CogBaseApp(
            name='testapp',
            llm=_make_llm("{}"),
            extractor=None,
            structured_store=None,
            vector_store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        await app.setup()
        results = await app.ingest_documents([Document(doc_id="d-001", text="word " * 20)])
        assert results[0].success is True
        assert results[0].records_extracted == 0
        assert vector_store.ntotal > 0

    @pytest.mark.asyncio
    async def test_ingest_records_extracted_is_zero(self):
        app = self._make_vector_only_app(_make_llm("{}"))
        await app.setup()
        results = await app.ingest_documents([Document(doc_id="d-001", text="some text " * 5)])
        assert results[0].records_extracted == 0


# ---------------------------------------------------------------------------
# ingest_documents()
# ---------------------------------------------------------------------------

class TestIngestMany:
    @pytest.mark.asyncio
    async def test_returns_one_result_per_document(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm(_contract_payload()), store)
        await app.setup()

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
        app = _make_app(_make_llm("{}"), store)
        await app.setup()

        doc_ids = [f"c-{i:03d}" for i in range(8)]
        documents = [Document(doc_id=d, text=f"text for {d}") for d in doc_ids]
        results = await app.ingest_documents(documents, concurrency=3)

        assert [r.doc_id for r in results] == doc_ids

    @pytest.mark.asyncio
    async def test_success_flag_set(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm(_contract_payload()), store)
        await app.setup()

        results = await app.ingest_documents([Document(doc_id="c-001", text="some text")])

        assert results[0].success is True
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_records_extracted_count(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm(_contract_payload()), store)
        await app.setup()

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
            return _contract_payload()

        llm.complete = AsyncMock(side_effect=_complete)

        async def _stream(*args, **kwargs):
            yield ""

        llm.complete_stream = _stream

        app = _make_app(llm, store)
        await app.setup()

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
        app = _make_app(_make_llm("{}"), store)
        await app.setup()

        results = await app.ingest_documents([])
        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        store = InMemoryStructuredStore()
        app = _make_app(_make_llm("{}"), store)
        await app.setup()

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

        app = _make_app(llm, store)
        await app.setup()

        documents = [Document(doc_id=f"c-{i}", text="text") for i in range(10)]
        await app.ingest_documents(documents, concurrency=3)

        assert peak <= 3
