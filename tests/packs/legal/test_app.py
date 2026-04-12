"""Integration tests for LegalContractApp."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Chunk
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.router import QueryPattern
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from packs.legal import LegalContractApp
from packs.legal.schema import CLAUSES_COLLECTION


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


def _clauses_payload(*clause_types: str) -> str:
    """Return a JSON clause array with the given types."""
    return json.dumps([
        {"type": t, "text": f"Clause text for {t}.", "confidence": 0.9}
        for t in clause_types
    ])


class StubEmbedder(EmbedderBase):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        return [c.model_copy(update={"embedding": [0.1] * self._dim}) for c in chunks]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLegalContractAppConstruction:
    def test_structured_only_builds(self):
        client = _make_extractor_response("[]")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        assert app.application.name == "legal"
        assert len(app.application.structured_collections) == 1
        assert app.application.structured_collections[0].name == CLAUSES_COLLECTION
        assert app.application.vector_collections == []

    def test_full_mode_builds(self):
        client = _make_extractor_response("[]")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(dim=4),
            embedder=StubEmbedder(dim=4),
            chunker=FixedSizeChunker(chunk_size=64, overlap=0),
        )
        assert len(app.application.vector_collections) == 1
        assert app.application.vector_collections[0].name == "documents"

    def test_partial_vector_params_raises(self):
        client = _make_extractor_response("[]")
        with pytest.raises(ValueError, match="all be provided together"):
            LegalContractApp(
                client=client,
                model="test-model",
                structured_store=InMemoryStructuredStore(),
                vector_store=FAISSVectorStore(dim=4),
                # embedder and chunker missing
            )

    def test_custom_name(self):
        client = _make_extractor_response("[]")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
            name="my-legal-app",
        )
        assert app.application.name == "my-legal-app"

    def test_structured_schemas_exposed(self):
        client = _make_extractor_response("[]")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=InMemoryStructuredStore(),
        )
        schemas = app.structured_schemas
        assert len(schemas) == 1
        assert schemas[0].name == CLAUSES_COLLECTION


# ---------------------------------------------------------------------------
# setup() / ingest()
# ---------------------------------------------------------------------------

class TestLegalContractAppLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_collection(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("[]")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        # Collection exists — query must not raise
        rows = await store.query(CLAUSES_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_setup_idempotent(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response("[]")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_ingest_extracts_clauses(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_clauses_payload("payment", "termination"))
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest("Some contract text.", doc_id="c-001")
        rows = await store.query(CLAUSES_COLLECTION)
        assert len(rows) == 2
        types = {r["type"] for r in rows}
        assert types == {"payment", "termination"}

    @pytest.mark.asyncio
    async def test_ingest_empty_text_is_noop(self):
        store = InMemoryStructuredStore()
        # extractor won't be called for empty text, but client returns empty regardless
        client = _make_extractor_response("[]")
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest("", doc_id="c-empty")
        rows = await store.query(CLAUSES_COLLECTION)
        assert rows == []

    @pytest.mark.asyncio
    async def test_ingest_multiple_docs_accumulate(self):
        store = InMemoryStructuredStore()
        client = _make_extractor_response(_clauses_payload("payment"))
        app = LegalContractApp(client=client, model="test-model", structured_store=store)
        await app.setup()
        await app.ingest("contract one text", doc_id="c-001")
        await app.ingest("contract two text", doc_id="c-002")
        rows = await store.query(CLAUSES_COLLECTION)
        assert len(rows) == 2
        doc_ids = {r["doc_id"] for r in rows}
        assert doc_ids == {"c-001", "c-002"}

    @pytest.mark.asyncio
    async def test_ingest_full_mode_populates_vector_store(self):
        store = InMemoryStructuredStore()
        vector_store = FAISSVectorStore(dim=4)
        client = _make_extractor_response("[]")
        app = LegalContractApp(
            client=client,
            model="test-model",
            structured_store=store,
            vector_store=vector_store,
            embedder=StubEmbedder(dim=4),
            chunker=FixedSizeChunker(chunk_size=20, overlap=0),
        )
        await app.setup()
        await app.ingest("word " * 20, doc_id="c-001")
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
        extractor_json: str = "[]",
    ) -> tuple[LegalContractApp, InMemoryStructuredStore]:
        """Create an app whose LLM always responds with the given router and generator content."""
        store = InMemoryStructuredStore()
        call_count = 0

        async def _create(**kwargs):
            nonlocal call_count
            call_count += 1
            messages = kwargs.get("messages", [])
            user_content = messages[-1].get("content", "") if messages else ""
            # Extractor calls have the contract text; router/generator have query text
            # Distinguish by checking for the system prompt pattern
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
            "semantic_query": "list payment clauses",
            "structured_targets": [{"collection": CLAUSES_COLLECTION, "filters": []}],
        })
        app, store = self._make_app_with_router_response(
            router_resp,
            generator_answer="unused for pattern A",
        )
        await app.setup()
        # Pre-load a clause directly into the store
        from packs.legal.schema import Clause
        await store.save(CLAUSES_COLLECTION, [
            Clause(
                clause_id="c-001_payment_0_abc",
                doc_id="c-001",
                type="payment",
                text="Payment is due within 30 days.",
                confidence=0.95,
            )
        ])

        result = await app.query("list payment clauses")
        assert isinstance(result, GenerationResult)
        assert result.pattern == QueryPattern.A
        assert "payment" in result.answer.lower() or "30 days" in result.answer

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
            "structured_targets": [{"collection": CLAUSES_COLLECTION, "filters": []}],
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
        client = _make_extractor_response("[]")
        app = LegalContractApp(client=client, model="test-model", structured_store=InMemoryStructuredStore())
        # Both properties return the right types
        from cogbase.core.application import Application
        from cogbase.engine.engine import Engine
        assert isinstance(app.application, Application)
        assert isinstance(app.engine, Engine)
