"""Tests for FAISSVectorStore."""

import math
from uuid import uuid4

import numpy as np
import pytest

from cogbase.core.models import Chunk
from cogbase.stores.filters import Col
from cogbase.stores.vector.faiss_store import FAISSVectorStore

COLLECTION = "chunks"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def unit(v: list[float]) -> list[float]:
    """Return the L2-normalised version of v."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def make_chunk(doc_id: str = "doc-1", embedding: list[float] | None = None, **kwargs) -> Chunk:
    chunk_id = kwargs.pop("chunk_id", f"{doc_id}_{uuid4().hex[:8]}")
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text="sample text", embedding=embedding, **kwargs)


# ------------------------------------------------------------------
# Basic upsert / search
# ------------------------------------------------------------------

async def test_empty_store_returns_no_results():
    store = FAISSVectorStore()
    assert await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5) == []


async def test_upsert_and_search_returns_chunk():
    store = FAISSVectorStore()
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id


async def test_search_returns_nearest_neighbour():
    store = FAISSVectorStore()
    # Three orthogonal unit vectors
    a = make_chunk(doc_id="doc-a", embedding=[1.0, 0.0, 0.0])
    b = make_chunk(doc_id="doc-b", embedding=[0.0, 1.0, 0.0])
    c = make_chunk(doc_id="doc-c", embedding=[0.0, 0.0, 1.0])
    await store.upsert(COLLECTION, [a, b, c])

    # Query close to vector b
    results = await store.search(COLLECTION, [0.1, 0.9, 0.1], top_k=1)
    assert results[0].chunk_id == b.chunk_id


async def test_search_top_k_limits_results():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=[float(i), 0.0, 0.0]) for i in range(1, 6)])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3


async def test_search_top_k_larger_than_index_returns_all():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0]), make_chunk(embedding=[0.0, 1.0])])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=100)
    assert len(results) == 2


async def test_cosine_order():
    """Verify results are ordered highest cosine similarity first."""
    store = FAISSVectorStore()
    # query = [1, 0]; a is closer, b is further
    a = make_chunk(doc_id="a", embedding=unit([0.9, 0.1]))   # ~84° from [0,1]
    b = make_chunk(doc_id="b", embedding=unit([0.1, 0.9]))   # ~6° from [0,1]
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=2)
    assert results[0].chunk_id == a.chunk_id
    assert results[1].chunk_id == b.chunk_id


# ------------------------------------------------------------------
# Chunks without embeddings
# ------------------------------------------------------------------

async def test_chunks_without_embedding_are_skipped():
    store = FAISSVectorStore()
    no_emb = make_chunk(embedding=None)
    with_emb = make_chunk(embedding=[1.0, 0.0])
    await store.upsert(COLLECTION, [no_emb, with_emb])
    assert store.ntotal == 1


async def test_all_chunks_without_embeddings_is_a_no_op():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=None)])
    assert store.ntotal == 0
    assert await store.search(COLLECTION, [1.0, 0.0], top_k=1) == []


# ------------------------------------------------------------------
# Upsert (update existing)
# ------------------------------------------------------------------

async def test_upsert_replaces_existing_chunk():
    store = FAISSVectorStore()
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])

    updated = Chunk(
        chunk_id=chunk.chunk_id,
        doc_id="doc-updated",
        text="updated text",
        embedding=[0.0, 1.0, 0.0],
    )
    await store.upsert(COLLECTION, [updated])

    assert store.ntotal == 1
    results = await store.search(COLLECTION, [0.0, 1.0, 0.0], top_k=1)
    assert results[0].doc_id == "doc-updated"


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------

async def test_delete_removes_doc_chunks():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[1.0, 0.0]),
        make_chunk(doc_id="doc-1", embedding=[0.9, 0.1]),
        make_chunk(doc_id="doc-2", embedding=[0.0, 1.0]),
    ])
    await store.delete(COLLECTION, "doc-1")
    assert store.ntotal == 1
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5)
    assert all(r.doc_id == "doc-2" for r in results)


async def test_delete_unknown_doc_is_a_no_op():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0])])
    await store.delete(COLLECTION, "nonexistent-doc")
    assert store.ntotal == 1


async def test_delete_all_chunks_leaves_empty_store():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])])
    await store.delete(COLLECTION, "doc-1")
    assert store.ntotal == 0
    assert await store.search(COLLECTION, [1.0, 0.0], top_k=5) == []


async def test_upsert_after_delete_works():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])])
    await store.delete(COLLECTION, "doc-1")
    new_chunk = make_chunk(doc_id="doc-2", embedding=[0.0, 1.0])
    await store.upsert(COLLECTION, [new_chunk])
    results = await store.search(COLLECTION, [0.0, 1.0], top_k=1)
    assert results[0].chunk_id == new_chunk.chunk_id


async def test_delete_collection_clears_store():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[1.0, 0.0]),
        make_chunk(doc_id="doc-2", embedding=[0.0, 1.0]),
    ])
    await store.delete_collection(COLLECTION)
    assert store.ntotal == 0
    assert await store.search(COLLECTION, [1.0, 0.0], top_k=5) == []


async def test_delete_collection_then_upsert_works():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])])
    await store.delete_collection(COLLECTION)
    new_chunk = make_chunk(doc_id="doc-2", embedding=[0.0, 1.0])
    await store.upsert(COLLECTION, [new_chunk])
    results = await store.search(COLLECTION, [0.0, 1.0], top_k=1)
    assert results[0].chunk_id == new_chunk.chunk_id


# ------------------------------------------------------------------
# Dimension mismatch
# ------------------------------------------------------------------

async def test_dimension_mismatch_raises():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0])])
    with pytest.raises(ValueError, match="dimension"):
        await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0, 0.0])])


async def test_explicit_dim_constructor():
    store = FAISSVectorStore(dim=3)
    assert store.ntotal == 0
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0, 0.0])])
    assert store.ntotal == 1


# ------------------------------------------------------------------
# Persistence (save / load)
# ------------------------------------------------------------------

async def test_save_and_load_roundtrip(tmp_path):
    store = FAISSVectorStore()
    chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])
    await store.save(tmp_path / "faiss_store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "faiss_store")
    results = await loaded.search(COLLECTION, [1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id
    assert results[0].doc_id == chunk.doc_id
    assert results[0].text == chunk.text


async def test_save_and_load_preserves_ntotal(tmp_path):
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[1.0, 0.0]),
        make_chunk(doc_id="doc-2", embedding=[0.0, 1.0]),
    ])
    await store.save(tmp_path / "store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "store")
    assert loaded.ntotal == 2


async def test_save_creates_nested_directory(tmp_path):
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0])])
    nested = tmp_path / "a" / "b" / "store"
    await store.save(nested)
    assert (nested / "index.faiss").exists()
    assert (nested / "meta.json").exists()


async def test_save_empty_store_raises(tmp_path):
    store = FAISSVectorStore()
    with pytest.raises(RuntimeError, match="empty"):
        await store.save(tmp_path / "store")


async def test_load_restores_search_order(tmp_path):
    store = FAISSVectorStore()
    a = make_chunk(doc_id="a", embedding=unit([0.9, 0.1]))
    b = make_chunk(doc_id="b", embedding=unit([0.1, 0.9]))
    await store.upsert(COLLECTION, [a, b])
    await store.save(tmp_path / "store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "store")
    results = await loaded.search(COLLECTION, [1.0, 0.0], top_k=2)
    assert results[0].doc_id == "a"
    assert results[1].doc_id == "b"


async def test_delete_after_load(tmp_path):
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[1.0, 0.0]),
        make_chunk(doc_id="doc-2", embedding=[0.0, 1.0]),
    ])
    await store.save(tmp_path / "store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "store")
    await loaded.delete(COLLECTION, "doc-1")
    assert loaded.ntotal == 1
    results = await loaded.search(COLLECTION, [0.0, 1.0], top_k=1)
    assert results[0].doc_id == "doc-2"


async def test_upsert_after_load(tmp_path):
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])])
    await store.save(tmp_path / "store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "store")
    new_chunk = make_chunk(doc_id="doc-2", embedding=[0.0, 1.0])
    await loaded.upsert(COLLECTION, [new_chunk])
    assert loaded.ntotal == 2
    results = await loaded.search(COLLECTION, [0.0, 1.0], top_k=1)
    assert results[0].chunk_id == new_chunk.chunk_id


async def test_save_preserves_chunk_metadata(tmp_path):
    store = FAISSVectorStore()
    chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"source": "upload", "page": 3})
    await store.upsert(COLLECTION, [chunk])
    await store.save(tmp_path / "store")

    loaded = FAISSVectorStore()
    await loaded.load(tmp_path / "store")
    results = await loaded.search(COLLECTION, [1.0, 0.0], top_k=1)
    assert results[0].metadata == {"source": "upload", "page": 3}


# ------------------------------------------------------------------
# Metadata filters
# ------------------------------------------------------------------

async def test_search_filter_by_doc_id():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0])
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1, 0.0])
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "doc-1"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


async def test_search_filter_by_metadata_eq():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1], metadata={"section": "body"})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("metadata.section") == "intro"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


async def test_search_filter_in_operator():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1])
    c = make_chunk(doc_id="doc-3", embedding=[0.1, 0.9])
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("doc_id").in_(["doc-1", "doc-2"])])
    assert len(results) == 2
    assert {r.doc_id for r in results} == {"doc-1", "doc-2"}


async def test_search_filter_metadata_numeric_gte():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"page": 1})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1], metadata={"page": 3})
    c = make_chunk(doc_id="doc-1", embedding=[0.8, 0.2], metadata={"page": 5})
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("metadata.page") >= 3])
    assert len(results) == 2
    assert all(r.metadata["page"] >= 3 for r in results)


async def test_search_filter_metadata_is_null():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"page": 1})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1], metadata={})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("metadata.page").is_null()])
    assert len(results) == 1
    assert results[0].chunk_id == b.chunk_id


async def test_search_filter_no_match_returns_empty():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0])])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "no-such-doc"])
    assert results == []


async def test_search_filter_respects_top_k():
    """top_k is applied after filtering."""
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[float(i), 0.0]) for i in range(1, 6)
    ])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=2,
                                 filters=[Col("doc_id") == "doc-1"])
    assert len(results) == 2


async def test_search_multiple_filters_are_anded():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1], metadata={"section": "body"})
    c = make_chunk(doc_id="doc-2", embedding=[0.8, 0.2], metadata={"section": "intro"})
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5, filters=[
        Col("doc_id") == "doc-1",
        Col("metadata.section") == "intro",
    ])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


# ------------------------------------------------------------------
# Field projection
# ------------------------------------------------------------------

async def test_search_fields_omits_embedding():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0],
                                               metadata={"k": "v"})])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=1,
                                 fields=["chunk_id", "doc_id", "text", "metadata"])
    assert results[0].embedding is None
    assert results[0].metadata == {"k": "v"}


async def test_search_fields_omits_metadata():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0],
                                               metadata={"k": "v"})])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=1,
                                 fields=["chunk_id", "doc_id", "text", "embedding"])
    assert results[0].metadata == {}
    assert results[0].embedding is not None


async def test_search_fields_none_returns_all():
    store = FAISSVectorStore()
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0],
                                               metadata={"k": "v"})])
    results = await store.search(COLLECTION, [1.0, 0.0], top_k=1, fields=None)
    assert results[0].embedding is not None
    assert results[0].metadata == {"k": "v"}


async def test_search_filters_and_fields_combined():
    store = FAISSVectorStore()
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1], metadata={"section": "body"})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "doc-1"],
                                 fields=["chunk_id", "doc_id", "text", "metadata"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id
    assert results[0].embedding is None
    assert results[0].metadata == {"section": "intro"}
