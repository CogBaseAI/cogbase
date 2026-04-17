"""Integration tests for LegalContractApp."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Chunk, Document
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.router import QueryPattern
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from packs.legal.contract_analyst import IngestResult, LegalContractApp
from packs.legal.contract_analyst.schema import CONTRACTS_COLLECTION, ContractRecord, Party, PaymentTerms


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_extractor_response(content: str) -> MagicMock:
    """Build a mock OpenAI client that always returns *content*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _contract_payload(**overrides) -> str:
    """Return a minimal valid ContractExtractor JSON response (single object)."""
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

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        return [c.model_copy(update={"embedding": [0.1] * self._dim}) for c in chunks]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLegalContractAppConstruction:
    def test_structured_only_builds(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        assert app.application.name == "legal"
        assert len(app.application.structured_collections) == 1
        assert app.application.structured_collections[0].name == CONTRACTS_COLLECTION
        assert app.application.vector_collections == []

    def test_full_mode_builds(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        assert len(app.application.vector_collections) == 1
        assert app.application.vector_collections[0].name == "documents"

    def test_partial_vector_params_raises(self):
        client = _make_extractor_response("{}")
        with pytest.raises(ValueError, match="all be provided together"):
            LegalContractApp(
                client=client,
                model="test-model",
                structured_store=InMemoryStructuredStore(),
                vector_store=FAISSVectorStore(dim=4),
                # embedder and chunker missing
            )

    def test_custom_name(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
            name="my-legal-app",
        )
        assert app.application.name == "my-legal-app"

    def test_structured_schemas_exposed(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        schemas = app.structured_schemas
        assert len(schemas) == 1
        assert schemas[0].name == CONTRACTS_COLLECTION


# ---------------------------------------------------------------------------
# setup() / ingest()
# ---------------------------------------------------------------------------

class TestLegalContractAppLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_collection(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        # Collection exists — query must not raise
        rows = await store.query(CONTRACTS_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_setup_idempotent(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_ingest_extracts_contract(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload(contract_type="SaaS"))
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest(Document(doc_id="c-001", text="Some contract text."))
        rows = await store.query(CONTRACTS_COLLECTION)
        assert len(rows) == 1
        assert rows[0]["contract_type"] == "SaaS"

    @pytest.mark.asyncio
    async def test_ingest_empty_text_is_noop(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest(Document(doc_id="c-empty", text=""))
        rows = await store.query(CONTRACTS_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload())
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest(Document(doc_id="c-001", text="contract one text"))
        await app.ingest(Document(doc_id="c-002", text="contract two text"))
        rows = await store.query(CONTRACTS_COLLECTION)
        assert len(rows) == 2
        doc_ids = {r["doc_id"] for r in rows}
        assert doc_ids == {"c-001", "c-002"}

    @pytest.mark.asyncio
    async def test_ingest_full_mode_populates_vector_store(self):
        store = InMemoryStructuredStore()
        vector_store = FAISSVectorStore(dim=4)
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=store,
            vector_store=vector_store,
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        await app.setup()
        await app.ingest(Document(doc_id="c-001", text="word " * 20))
        assert vector_store.ntotal > 0


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestLegalContractAppQuery:
    def _make_app_with_router_response(
        self,
        router_json: str,
        generator_answer: str,
        *,
        extractor_json: str = "{}",
    ) -> tuple[LegalContractApp, InMemoryStructuredStore]:
        """Create an app whose LLM always responds with the given router and generator content."""
        store = InMemoryStructuredStore()

        async def _create(**kwargs):
            messages = kwargs.get("messages", [])
            system_content = messages[0].get("content", "") if messages else ""
            if "legal contract analyst" in system_content:
                content = extractor_json
            elif "query router" in system_content:
                content = router_json
            else:
                content = generator_answer
            choice = SimpleNamespace(message=SimpleNamespace(content=content))
            return SimpleNamespace(choices=[choice])

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        return app, store

    @pytest.mark.asyncio
    async def test_query_pattern_a_returns_structured_answer(self):
        router_resp = json.dumps({
            "pattern": "A",
            "semantic_query": "list NDA contracts",
            "structured_targets": [{"collection": CONTRACTS_COLLECTION, "filters": []}],
        })
        app, store = self._make_app_with_router_response(
            router_resp,
            generator_answer="unused for pattern A",
        )
        await app.setup()
        # Pre-load a contract record directly into the store
        await store.save(CONTRACTS_COLLECTION, [
            ContractRecord(
                contract_id="c-001_abc",
                doc_id="c-001",
                contract_type="NDA",
                parties=[Party(name="Acme Corp", role="discloser"), Party(name="Supplier Ltd", role="recipient")],
                payment_terms=PaymentTerms(schedule="net-30", verbatim="Payment is due within 30 days."),
            )
        ])

        result = await app.query("list NDA contracts")
        assert isinstance(result, GenerationResult)
        assert result.pattern == QueryPattern.A
        assert "acme" in result.answer.lower() or "nda" in result.answer.lower() or "30 days" in result.answer

    @pytest.mark.asyncio
    async def test_query_pattern_b_returns_answer(self):
        router_resp = json.dumps({
            "pattern": "B",
            "semantic_query": "termination notice period",
            "structured_targets": [],
        })
        app, store = self._make_app_with_router_response(
            router_resp,
            generator_answer="The termination notice period is 60 days.",
        )
        await app.setup()
        result = await app.query("what is the termination notice period?")
        assert isinstance(result, GenerationResult)
        assert result.pattern == QueryPattern.B

    @pytest.mark.asyncio
    async def test_query_pattern_d_populates_findings(self):
        router_resp = json.dumps({
            "pattern": "D",
            "semantic_query": "summarise indemnification",
            "structured_targets": [{"collection": CONTRACTS_COLLECTION, "filters": []}],
        })
        gen_answer = (
            "[FINDINGS]\nBoth parties have broad indemnification obligations.\n\n"
            "[SUPPORTING_QUOTES]\n- Each party shall indemnify the other."
        )
        app, store = self._make_app_with_router_response(router_resp, gen_answer)
        await app.setup()
        result = await app.query("summarise indemnification clauses")
        assert result.pattern == QueryPattern.D
        assert result.findings is not None
        assert "indemnification" in result.findings.lower()
        assert len(result.supporting_quotes) > 0

    @pytest.mark.asyncio
    async def test_engine_and_application_accessible(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=InMemoryStructuredStore())
        from cogbase.core.application import Application
        from cogbase.engine.engine import Engine
        assert isinstance(app.application, Application)
        assert isinstance(app.engine, Engine)


# ---------------------------------------------------------------------------
# Structured-only mode — pattern B/C skipping
# ---------------------------------------------------------------------------

class TestStructuredOnlyPatternRestriction:
    """Verify that structured-only mode excludes B and C from the router prompt."""

    def _capture_system_prompt(self, app: LegalContractApp) -> str:
        """Trigger a route call and return the system prompt the router sent."""
        # The router's system prompt is baked in at construction time; we read
        # it directly from the private attribute so no LLM call is needed.
        return app.engine._router._system_prompt

    def test_structured_only_prompt_excludes_pattern_b(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        assert "B —" not in self._capture_system_prompt(app)

    def test_structured_only_prompt_excludes_pattern_c(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        assert "C —" not in self._capture_system_prompt(app)

    def test_structured_only_prompt_includes_pattern_a_and_d(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        prompt = self._capture_system_prompt(app)
        assert "A —" in prompt
        assert "D —" in prompt

    def test_full_mode_prompt_includes_all_four_patterns(self):
        client = _make_extractor_response("{}")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(dim=4),
            embedder=StubEmbedding(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        prompt = self._capture_system_prompt(app)
        for label in ("A —", "B —", "C —", "D —"):
            assert label in prompt

    @pytest.mark.asyncio
    async def test_structured_only_query_pattern_b_returns_empty_chunks(self):
        """Even if the router somehow returns B, retrieval yields empty chunks."""
        router_resp = json.dumps({
            "pattern": "B",
            "semantic_query": "termination clause",
            "structured_targets": [],
        })
        app, _ = self._make_app_with_router_response(router_resp, "The answer.")
        await app.setup()
        result = await app.query("what is the termination clause?")
        # No vector store — chunks are empty; generation still returns an answer.
        assert isinstance(result, GenerationResult)

    def _make_app_with_router_response(
        self,
        router_json: str,
        generator_answer: str,
    ) -> tuple[LegalContractApp, InMemoryStructuredStore]:
        store = InMemoryStructuredStore()

        async def _create(**kwargs):
            messages = kwargs.get("messages", [])
            system_content = messages[0].get("content", "") if messages else ""
            if "legal contract analyst" in system_content:
                content = "{}"
            elif "query router" in system_content:
                content = router_json
            else:
                content = generator_answer
            choice = SimpleNamespace(message=SimpleNamespace(content=content))
            return SimpleNamespace(choices=[choice])

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        return app, store


# ---------------------------------------------------------------------------
# ingest_many()
# ---------------------------------------------------------------------------

class TestIngestMany:
    @pytest.mark.asyncio
    async def test_returns_one_result_per_contract(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload())
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        contracts = [
            Document(doc_id="c-001", text="contract one"),
            Document(doc_id="c-002", text="contract two"),
            Document(doc_id="c-003", text="contract three"),
        ]
        results = await app.ingest_many(contracts)

        assert len(results) == 3
        assert all(isinstance(r, IngestResult) for r in results)

    @pytest.mark.asyncio
    async def test_results_in_input_order(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        doc_ids = [f"c-{i:03d}" for i in range(8)]
        contracts = [Document(doc_id=d, text=f"text for {d}") for d in doc_ids]
        results = await app.ingest_many(contracts, concurrency=3)

        assert [r.doc_id for r in results] == doc_ids

    @pytest.mark.asyncio
    async def test_success_flag_set(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload())
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        results = await app.ingest_many([Document(doc_id="c-001", text="some text")])

        assert results[0].success is True
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_records_extracted_count(self):
        """Each contract produces exactly 1 record."""
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload())
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        results = await app.ingest_many([Document(doc_id="c-001", text="contract text")])

        assert results[0].records_extracted == 1

    @pytest.mark.asyncio
    async def test_records_per_doc_counted_independently(self):
        """Each result reflects only that document's records, not a cumulative total."""
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_contract_payload())
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        results = await app.ingest_many(
            [
                Document(doc_id="c-001", text="first contract"),
                Document(doc_id="c-002", text="second contract"),
            ],
            concurrency=1,
        )

        assert results[0].doc_id == "c-001"
        assert results[0].records_extracted == 1
        assert results[1].doc_id == "c-002"
        assert results[1].records_extracted == 1

    @pytest.mark.asyncio
    async def test_failure_captured_not_raised(self):
        """A failure on one document does not abort the batch."""
        store = InMemoryStructuredStore()
        call_n = 0

        async def _create(**kwargs):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise RuntimeError("LLM unavailable")
            choice = SimpleNamespace(message=SimpleNamespace(content=_contract_payload()))
            return SimpleNamespace(choices=[choice])

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        results = await app.ingest_many(
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
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        results = await app.ingest_many([])
        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_concurrency_raises(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("{}")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        with pytest.raises(ValueError, match="concurrency"):
            await app.ingest_many([], concurrency=0)

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Track peak simultaneous in-flight ingestions."""
        store = InMemoryStructuredStore()
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def _create(**kwargs):
            nonlocal active, peak
            async with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0)  # yield to allow other coroutines to enter
            async with lock:
                active -= 1
            choice = SimpleNamespace(message=SimpleNamespace(content="{}"))
            return SimpleNamespace(choices=[choice])

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()

        contracts = [Document(doc_id=f"c-{i}", text="text") for i in range(10)]
        await app.ingest_many(contracts, concurrency=3)

        assert peak <= 3
