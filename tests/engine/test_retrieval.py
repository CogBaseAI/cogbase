"""Integration tests for cogbase.engine.retrieval — uses real store implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Chunk
from cogbase.embeddings.base import EmbeddingBase
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.retrieval.hybrid import HybridRetriever
from cogbase.engine.retrieval.structured import StructuredRetriever
from cogbase.engine.retrieval.vector import VectorRetriever
from cogbase.engine.router import CollectionTarget, QueryPattern, RouteResult
from cogbase.stores.filters import Col
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _Fact(BaseModel):
    fact_id: str
    type: str
    value: str


_FACTS_SCHEMA = CollectionSchema(
    name="facts",
    primary_fields=["fact_id"],
    fields={
        "fact_id": FieldSchema(type=FieldType.STRING),
        "type": FieldSchema(type=FieldType.STRING),
        "value": FieldSchema(type=FieldType.STRING),
    },
)


class _FixedEmbedder(EmbeddingBase):
    """Deterministic 3-dim embedder — maps known texts to fixed unit vectors."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        default = [1.0, 0.0, 0.0]
        return [self._mapping.get(t, default) for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route(
    pattern: QueryPattern,
    semantic_query: str = "test query",
    collection: str | None = None,
    filters=None,
) -> RouteResult:
    targets: list[CollectionTarget] = []
    if collection is not None:
        targets = [CollectionTarget(collection=collection, filters=filters or [])]
    return RouteResult(
        pattern=pattern,
        semantic_query=semantic_query,
        structured_targets=targets,
    )


def _chunk(chunk_id: str, text: str = "hello", embedding: list[float] | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc-1", text=text, embedding=embedding or [1.0, 0.0, 0.0])


async def _structured_store(records: list[_Fact] = (), schema: CollectionSchema = _FACTS_SCHEMA) -> InMemoryStructuredStore:
    store = InMemoryStructuredStore()
    await store.create_collection(schema)
    if records:
        await store.save(schema.name, list(records))
    return store


async def _vector_store(chunks: list[Chunk], collection: str = "chunks") -> FAISSVectorStore:
    store = FAISSVectorStore(dim=3)
    if chunks:
        await store.upsert(collection, chunks)
    return store


# ---------------------------------------------------------------------------
# StructuredRetriever
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_structured_retriever_returns_records() -> None:
    store = await _structured_store([_Fact(fact_id="1", type="date", value="2024-01-01")])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert isinstance(result, RetrievalResult)
    assert len(result.structured_records) == 1
    assert result.structured_records[0]["fact_id"] == "1"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_structured_retriever_filters_by_column() -> None:
    store = await _structured_store([
        _Fact(fact_id="1", type="date", value="2024-01-01"),
        _Fact(fact_id="2", type="amount", value="500"),
    ])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts", filters=[Col("type") == "date"])

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 1
    assert result.structured_records[0]["type"] == "date"


@pytest.mark.asyncio
async def test_structured_retriever_empty_filters_queries_all() -> None:
    store = await _structured_store([
        _Fact(fact_id="1", type="date", value="2024-01-01"),
        _Fact(fact_id="2", type="amount", value="500"),
    ])
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 2


@pytest.mark.asyncio
async def test_structured_retriever_raises_without_collection() -> None:
    store = await _structured_store()
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A)  # no collection → structured_targets=[]

    with pytest.raises(ValueError, match="collection"):
        await retriever.retrieve(route)


@pytest.mark.asyncio
async def test_structured_retriever_preserves_route() -> None:
    store = await _structured_store()
    retriever = StructuredRetriever(store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert result.route is route


@pytest.mark.asyncio
async def test_structured_retriever_merges_multiple_targets() -> None:
    contracts_schema = CollectionSchema(
        name="contracts",
        primary_fields=["fact_id"],
        fields={
            "fact_id": FieldSchema(type=FieldType.STRING),
            "type": FieldSchema(type=FieldType.STRING),
            "value": FieldSchema(type=FieldType.STRING),
        },
    )
    store = InMemoryStructuredStore()
    await store.create_collection(_FACTS_SCHEMA)
    await store.create_collection(contracts_schema)
    await store.save("facts", [_Fact(fact_id="f1", type="date", value="2024")])
    await store.save("contracts", [_Fact(fact_id="c1", type="contract", value="foo")])

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

    assert len(result.structured_records) == 2
    types = {r["type"] for r in result.structured_records}
    assert types == {"date", "contract"}


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_retriever_returns_chunks() -> None:
    chunk = _chunk("doc-1_0", embedding=[1.0, 0.0, 0.0])
    store = await _vector_store([chunk])
    retriever = VectorRetriever("chunks", store, _FixedEmbedder(), top_k=5)
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == chunk.chunk_id
    assert result.structured_records == []


@pytest.mark.asyncio
async def test_vector_retriever_respects_top_k() -> None:
    chunks = [_chunk(f"doc-1_{i}", embedding=[1.0, 0.0, 0.0]) for i in range(5)]
    store = await _vector_store(chunks)
    retriever = VectorRetriever("chunks", store, _FixedEmbedder(), top_k=3)
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert len(result.chunks) == 3


@pytest.mark.asyncio
async def test_vector_retriever_returns_closest_match() -> None:
    """The chunk whose embedding is most similar to the query ranks first."""
    chunk_a = _chunk("doc-1_0", text="notice period", embedding=[1.0, 0.0, 0.0])
    chunk_b = _chunk("doc-1_1", text="payment terms", embedding=[0.0, 1.0, 0.0])
    store = await _vector_store([chunk_a, chunk_b])
    # Query aligned with chunk_a
    embedder = _FixedEmbedder({"notice period length": [1.0, 0.0, 0.0]})
    retriever = VectorRetriever("chunks", store, embedder, top_k=2)
    route = _route(QueryPattern.B, semantic_query="notice period length")

    result = await retriever.retrieve(route)

    assert result.chunks[0].chunk_id == chunk_a.chunk_id


@pytest.mark.asyncio
async def test_vector_retriever_raises_when_embedder_returns_no_vector() -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[None])
    store = FAISSVectorStore(dim=3)
    retriever = VectorRetriever("chunks", store, embedder)
    route = _route(QueryPattern.B)

    with pytest.raises(RuntimeError, match="embedding"):
        await retriever.retrieve(route)


# ---------------------------------------------------------------------------
# HybridRetriever — dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_pattern_a_only_queries_structured() -> None:
    s_store = await _structured_store([_Fact(fact_id="1", type="date", value="2024")])
    v_store = await _vector_store([])
    retriever = HybridRetriever("chunks", s_store, v_store, _FixedEmbedder())
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 1
    assert result.chunks == []


@pytest.mark.asyncio
async def test_hybrid_pattern_b_only_queries_vector() -> None:
    chunk = _chunk("doc-1_0", embedding=[1.0, 0.0, 0.0])
    s_store = await _structured_store()
    v_store = await _vector_store([chunk])
    retriever = HybridRetriever("chunks", s_store, v_store, _FixedEmbedder())
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert result.chunks[0].chunk_id == chunk.chunk_id
    assert result.structured_records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_pattern_cd_queries_both_stores(pattern: QueryPattern) -> None:
    chunk = _chunk("doc-1_0", embedding=[1.0, 0.0, 0.0])
    s_store = await _structured_store([_Fact(fact_id="1", type="date", value="2024")])
    v_store = await _vector_store([chunk])
    retriever = HybridRetriever("chunks", s_store, v_store, _FixedEmbedder())
    route = _route(pattern, collection="facts")

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 1
    assert len(result.chunks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_pattern_cd_no_collection_skips_structured(pattern: QueryPattern) -> None:
    chunk = _chunk("doc-1_0", embedding=[1.0, 0.0, 0.0])
    s_store = await _structured_store()
    v_store = await _vector_store([chunk])
    retriever = HybridRetriever("chunks", s_store, v_store, _FixedEmbedder())
    route = _route(pattern)  # no collection → structured_targets=[]

    result = await retriever.retrieve(route)

    assert result.chunks[0].chunk_id == chunk.chunk_id
    assert result.structured_records == []


# ---------------------------------------------------------------------------
# HybridRetriever — no vector store (structured-only mode)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_no_vector_pattern_b_returns_empty_chunks() -> None:
    s_store = await _structured_store()
    retriever = HybridRetriever("chunks", s_store)
    route = _route(QueryPattern.B)

    result = await retriever.retrieve(route)

    assert result.chunks == []
    assert result.structured_records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [QueryPattern.C, QueryPattern.D])
async def test_hybrid_no_vector_pattern_cd_returns_structured_only(pattern: QueryPattern) -> None:
    s_store = await _structured_store([_Fact(fact_id="1", type="date", value="2024")])
    retriever = HybridRetriever("chunks", s_store)
    route = _route(pattern, collection="facts")

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 1
    assert result.chunks == []


@pytest.mark.asyncio
async def test_hybrid_no_vector_pattern_a_unaffected() -> None:
    s_store = await _structured_store([_Fact(fact_id="1", type="date", value="2024")])
    retriever = HybridRetriever("chunks", s_store)
    route = _route(QueryPattern.A, collection="facts")

    result = await retriever.retrieve(route)

    assert len(result.structured_records) == 1
    assert result.chunks == []
