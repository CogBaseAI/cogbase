"""Tests for cogbase.engine.retrieval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Chunk
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.retrieval.hybrid import HybridRetriever
from cogbase.engine.retrieval.structured import StructuredRetriever
from cogbase.engine.retrieval.vector import VectorRetriever
from cogbase.engine.router import CollectionTarget, QueryPattern, RouteResult
from cogbase.stores.filters import Col


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route(
    pattern: QueryPattern,
    semantic_query: str = "test query",
    collection: str | None = None,
    filters=None,
) -> RouteResult:
    """Build a ``RouteResult`` from the legacy (collection, filters) signature.

    Wraps the pair into a single ``CollectionTarget`` when *collection* is given,
    leaving ``structured_targets`` empty otherwise.  This keeps test call-sites
    unchanged while the public API uses the new multi-target model.
    """
    targets: list[CollectionTarget] = []
    if collection is not None:
        targets = [CollectionTarget(collection=collection, filters=filters or [])]
    return RouteResult(
        pattern=pattern,
        semantic_query=semantic_query,
        structured_targets=targets,
    )


def _mock_structured_store(records: list[dict]) -> MagicMock:
    store = MagicMock()
    store.query = AsyncMock(return_value=records)
    return store


def _mock_vector_store(chunks: list[Chunk]) -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=chunks)
    return store


def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    v = vector or [0.1, 0.2, 0.3]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[v])
    return embedder


def _make_chunk(text: str = "hello") -> Chunk:
    return Chunk(doc_id="doc-1", text=text)


# ---------------------------------------------------------------------------
# StructuredRetriever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_retriever_returns_records() -> None:
    records = [{"fact_id": "1", "type": "date", "value": "2024-01-01"}]
    store = _mock_structured_store(records)
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert isinstance(result, RetrievalResult)
    assert result.structured_records == records
    assert result.chunks == []


@pytest.mark.asyncio
async def test_structured_retriever_passes_filters_to_store() -> None:
    store = _mock_structured_store([])
    retriever = StructuredRetriever(store)
    filters = [Col("type") == "date"]
    route = _route(QueryPattern.A, collection="facts", filters=filters)

    await retriever.retrieve(route)

    store.query.assert_called_once_with("facts", filters)


@pytest.mark.asyncio
async def test_structured_retriever_empty_filters_queries_all() -> None:
    store = _mock_structured_store([])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    await retriever.retrieve(route)

    store.query.assert_called_once_with("facts", [])


@pytest.mark.asyncio
async def test_structured_retriever_raises_without_collection() -> None:
    store = _mock_structured_store([])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A)  # no collection → structured_targets=[]

    with pytest.raises(ValueError, match="collection"):
        await retriever.retrieve(route)


@pytest.mark.asyncio
async def test_structured_retriever_preserves_route() -> None:
    store = _mock_structured_store([])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert result.route is route


@pytest.mark.asyncio
async def test_structured_retriever_merges_multiple_targets() -> None:
    """Records from two collections are merged into a single list."""
    contract_records = [{"id": "c1", "type": "contract"}]
    fact_records = [{"id": "f1", "type": "date"}]

    store = MagicMock()
    store.query = AsyncMock(side_effect=[contract_records, fact_records])
    retriever = StructuredRetriever(store)

    route = RouteResult(
        pattern=QueryPattern.C,
        semantic_query="compare",
        structured_targets=[
            CollectionTarget(collection="contracts"),
            CollectionTarget(collection="facts"),
        ],
    )

    result = await retriever.retrieve(route)

    assert result.structured_records == contract_records + fact_records
    assert store.query.call_count == 2
    store.query.assert_any_call("contracts", [])
    store.query.assert_any_call("facts", [])


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_retriever_returns_chunks() -> None:
    chunk = _make_chunk("relevant passage")
    vector_store = _mock_vector_store([chunk])
    embedder = _mock_embedder()
    retriever = VectorRetriever(vector_store, embedder, top_k=5)
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert result.chunks == [chunk]
    assert result.structured_records == []


@pytest.mark.asyncio
async def test_vector_retriever_embeds_semantic_query() -> None:
    vector_store = _mock_vector_store([])
    embedder = _mock_embedder()
    retriever = VectorRetriever(vector_store, embedder)
    route = _route(QueryPattern.B, semantic_query="notice period length")

    await retriever.retrieve(route)

    call_args = embedder.embed.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0] == "notice period length"


@pytest.mark.asyncio
async def test_vector_retriever_passes_top_k_to_store() -> None:
    vector_store = _mock_vector_store([])
    embedder = _mock_embedder(vector=[0.1, 0.2])
    retriever = VectorRetriever(vector_store, embedder, top_k=7)
    route = _route(QueryPattern.B)

    await retriever.retrieve(route)

    _, call_top_k = vector_store.search.call_args[0]
    assert call_top_k == 7


@pytest.mark.asyncio
async def test_vector_retriever_passes_embedding_to_store() -> None:
    vec = [0.5, 0.6, 0.7]
    vector_store = _mock_vector_store([])
    embedder = _mock_embedder(vector=vec)
    retriever = VectorRetriever(vector_store, embedder)
    route = _route(QueryPattern.B)

    await retriever.retrieve(route)

    call_embedding, _ = vector_store.search.call_args[0]
    assert call_embedding == vec


@pytest.mark.asyncio
async def test_vector_retriever_raises_when_embedder_returns_no_vector() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[None])
    vector_store = _mock_vector_store([])
    retriever = VectorRetriever(vector_store, embedder)
    route = _route(QueryPattern.B)

    with pytest.raises(RuntimeError, match="embedding"):
        await retriever.retrieve(route)


# ---------------------------------------------------------------------------
# HybridRetriever — dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_pattern_a_only_queries_structured() -> None:
    s_store = _mock_structured_store([{"id": "1"}])
    v_store = _mock_vector_store([])
    embedder = _mock_embedder()
    retriever = HybridRetriever(s_store, v_store, embedder)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert result.structured_records == [{"id": "1"}]
    assert result.chunks == []
    v_store.search.assert_not_called()
    embedder.embed.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_pattern_b_only_queries_vector() -> None:
    chunk = _make_chunk("passage")
    s_store = _mock_structured_store([])
    v_store = _mock_vector_store([chunk])
    embedder = _mock_embedder()
    retriever = HybridRetriever(s_store, v_store, embedder)
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert result.chunks == [chunk]
    assert result.structured_records == []
    s_store.query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_pattern_cd_queries_both_stores(pattern: QueryPattern) -> None:
    records = [{"id": "r1"}]
    chunk = _make_chunk("passage")
    s_store = _mock_structured_store(records)
    v_store = _mock_vector_store([chunk])
    embedder = _mock_embedder()
    retriever = HybridRetriever(s_store, v_store, embedder)
    route = _route(pattern, collection="facts")

    result = await retriever.retrieve(route)

    assert result.structured_records == records
    assert result.chunks == [chunk]
    s_store.query.assert_called_once()
    v_store.search.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_pattern_cd_no_collection_skips_structured(pattern: QueryPattern) -> None:
    chunk = _make_chunk("passage")
    s_store = _mock_structured_store([])
    v_store = _mock_vector_store([chunk])
    embedder = _mock_embedder()
    retriever = HybridRetriever(s_store, v_store, embedder)
    route = _route(pattern)  # no collection → structured_targets=[]

    result = await retriever.retrieve(route)

    assert result.chunks == [chunk]
    assert result.structured_records == []
    s_store.query.assert_not_called()


# ---------------------------------------------------------------------------
# HybridRetriever — no vector store (structured-only mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_no_vector_pattern_b_returns_empty_chunks() -> None:
    """Pattern B with no vector store yields an empty RetrievalResult."""
    s_store = _mock_structured_store([])
    retriever = HybridRetriever(s_store)  # no vector_store / embedder
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert result.chunks == []
    assert result.structured_records == []
    s_store.query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_no_vector_pattern_cd_returns_structured_only(pattern: QueryPattern) -> None:
    """Without a vector store, C/D still return structured records but no chunks."""
    records = [{"id": "r1"}]
    s_store = _mock_structured_store(records)
    retriever = HybridRetriever(s_store)
    route = _route(pattern, collection="facts")

    result = await retriever.retrieve(route)

    assert result.structured_records == records
    assert result.chunks == []
    s_store.query.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_no_vector_pattern_a_unaffected() -> None:
    """Pattern A never touches the vector store — behaviour is unchanged."""
    records = [{"id": "r1"}]
    s_store = _mock_structured_store(records)
    retriever = HybridRetriever(s_store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert result.structured_records == records
    assert result.chunks == []
