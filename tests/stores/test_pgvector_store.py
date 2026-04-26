"""Tests for PGVectorStore (requires Docker)."""

import math
import subprocess
import time
import uuid

import pytest

from cogbase.core.models import Chunk
from cogbase.stores.base import VectorCollectionSchema
from cogbase.stores.filters import Col
from cogbase.stores.vector.pgvector_store import PGVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4
COLLECTION = "chunks"


def unit(v: list[float]) -> list[float]:
    """Return the L2-normalised version of v."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def make_chunk(doc_id: str = "doc-1", embedding: list[float] | None = None, **kwargs) -> Chunk:
    chunk_id = kwargs.pop("chunk_id", f"{doc_id}_{uuid.uuid4().hex[:8]}")
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text="sample text", embedding=embedding, **kwargs)


# ---------------------------------------------------------------------------
# Session-scoped pgvector container
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pgvector_container():
    """Start a pgvector/pgvector:pg17 Docker container for the test session."""
    container_name = f"cogbase_test_pgvec_{uuid.uuid4().hex[:8]}"
    db_user = "test"
    db_password = "test"
    db_name = "test"

    subprocess.run(
        [
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "-e", f"POSTGRES_USER={db_user}",
            "-e", f"POSTGRES_PASSWORD={db_password}",
            "-e", f"POSTGRES_DB={db_name}",
            "-p", "0:5432",
            "pgvector/pgvector:pg17",
        ],
        check=True,
        capture_output=True,
    )

    port = subprocess.check_output(
        [
            "docker", "inspect", container_name,
            "--format", "{{(index (index .NetworkSettings.Ports \"5432/tcp\") 0).HostPort}}",
        ],
        text=True,
    ).strip()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", container_name, "pg_isready", "-U", db_user],
            capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        raise RuntimeError("pgvector container did not become ready within 30 s")

    dsn = f"postgresql://{db_user}:{db_password}@localhost:{port}/{db_name}"
    yield dsn

    subprocess.run(["docker", "stop", container_name], capture_output=True)


@pytest.fixture
async def store(pgvector_container):
    """PGVectorStore connected to the session container; collection is reset per test."""
    s = PGVectorStore(dsn=pgvector_container)
    await s.connect()
    await s.delete_collection(COLLECTION)
    await s.create_collection(VectorCollectionSchema(name=COLLECTION, dimensions=DIM))
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# Basic upsert / search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_store_returns_no_results(store):
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_upsert_and_search_returns_chunk(store):
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id


@pytest.mark.asyncio
async def test_search_returns_nearest_neighbour(store):
    a = make_chunk(doc_id="doc-a", embedding=[1.0, 0.0, 0.0, 0.0])
    b = make_chunk(doc_id="doc-b", embedding=[0.0, 1.0, 0.0, 0.0])
    c = make_chunk(doc_id="doc-c", embedding=[0.0, 0.0, 1.0, 0.0])
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [0.1, 0.9, 0.1, 0.0], top_k=1)
    assert results[0].chunk_id == b.chunk_id


@pytest.mark.asyncio
async def test_search_top_k_limits_results(store):
    chunks = [make_chunk(embedding=[float(i), 0.0, 0.0, 0.0]) for i in range(1, 6)]
    await store.upsert(COLLECTION, chunks)
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_top_k_larger_than_index_returns_all(store):
    await store.upsert(COLLECTION, [
        make_chunk(embedding=[1.0, 0.0, 0.0, 0.0]),
        make_chunk(embedding=[0.0, 1.0, 0.0, 0.0]),
    ])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=100)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_cosine_order(store):
    """Results must be ordered by cosine similarity, highest first."""
    a = make_chunk(doc_id="a", embedding=unit([0.9, 0.1, 0.0, 0.0]))
    b = make_chunk(doc_id="b", embedding=unit([0.1, 0.9, 0.0, 0.0]))
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert results[0].chunk_id == a.chunk_id
    assert results[1].chunk_id == b.chunk_id


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metadata_is_preserved(store):
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0], metadata={"source": "contract.pdf", "page": "3"})
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].metadata == {"source": "contract.pdf", "page": "3"}


# ---------------------------------------------------------------------------
# Chunks without embeddings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunks_without_embedding_are_skipped(store):
    no_emb = make_chunk(embedding=None)
    with_emb = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [no_emb, with_emb])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == with_emb.chunk_id


@pytest.mark.asyncio
async def test_all_chunks_without_embeddings_is_a_no_op(store):
    await store.upsert(COLLECTION, [make_chunk(embedding=None)])
    assert await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5) == []


# ---------------------------------------------------------------------------
# Upsert (update existing)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_replaces_existing_chunk(store):
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])

    updated = Chunk(
        chunk_id=chunk.chunk_id,
        doc_id="doc-updated",
        text="updated text",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )
    await store.upsert(COLLECTION, [updated])

    results = await store.search(COLLECTION, [0.0, 1.0, 0.0, 0.0], top_k=1)
    assert results[0].doc_id == "doc-updated"
    assert results[0].text == "updated text"


@pytest.mark.asyncio
async def test_upsert_does_not_duplicate(store):
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])
    await store.upsert(COLLECTION, [chunk])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_removes_doc_chunks(store):
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0]),
        make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0]),
        make_chunk(doc_id="doc-2", embedding=[0.0, 1.0, 0.0, 0.0]),
    ])
    await store.delete(COLLECTION, "doc-1")
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=10)
    assert all(r.doc_id == "doc-2" for r in results)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_delete_unknown_doc_is_a_no_op(store):
    chunk = make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [chunk])
    await store.delete(COLLECTION, "nonexistent-doc")
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_delete_all_chunks_leaves_empty_store(store):
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])])
    await store.delete(COLLECTION, "doc-1")
    assert await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5) == []


@pytest.mark.asyncio
async def test_upsert_after_delete_works(store):
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])])
    await store.delete(COLLECTION, "doc-1")
    new_chunk = make_chunk(doc_id="doc-2", embedding=[0.0, 1.0, 0.0, 0.0])
    await store.upsert(COLLECTION, [new_chunk])
    results = await store.search(COLLECTION, [0.0, 1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk_id == new_chunk.chunk_id


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_collection_drops_table(store):
    await store.upsert(COLLECTION, [make_chunk(embedding=[1.0, 0.0, 0.0, 0.0])])
    await store.delete_collection(COLLECTION)
    # Table is gone — recreate and verify it's empty
    await store.create_collection(VectorCollectionSchema(name=COLLECTION, dimensions=DIM))
    assert await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5) == []


@pytest.mark.asyncio
async def test_delete_collection_is_idempotent(store):
    await store.delete_collection(COLLECTION)
    await store.delete_collection(COLLECTION)  # second call must not raise


@pytest.mark.asyncio
async def test_delete_collection_leaves_other_collections_intact(store):
    other = "other_chunks"
    await store.create_collection(VectorCollectionSchema(name=other, dimensions=DIM))
    try:
        chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])
        await store.upsert(other, [chunk])
        await store.delete_collection(COLLECTION)
        results = await store.search(other, [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
    finally:
        await store.delete_collection(other)


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------

def test_missing_dsn_and_pool_raises():
    with pytest.raises(ValueError, match="dsn or pool"):
        PGVectorStore()


def test_both_dsn_and_pool_raises():
    with pytest.raises(ValueError, match="not both"):
        PGVectorStore(dsn="postgresql://localhost/test", pool=object())  # type: ignore[arg-type]


def test_get_pool_before_connect_raises():
    s = PGVectorStore(dsn="postgresql://localhost/test")
    with pytest.raises(RuntimeError, match="Not connected"):
        s._get_pool()


# ---------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_filter_by_doc_id(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1, 0.0, 0.0])
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "doc-1"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


@pytest.mark.asyncio
async def test_search_filter_by_metadata_eq(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0], metadata={"section": "body"})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("metadata.section") == "intro"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


@pytest.mark.asyncio
async def test_search_filter_in_operator(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1, 0.0, 0.0])
    c = make_chunk(doc_id="doc-3", embedding=[0.1, 0.9, 0.0, 0.0])
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("doc_id").in_(["doc-1", "doc-2"])])
    assert len(results) == 2
    assert {r.doc_id for r in results} == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_search_filter_metadata_like(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0],
                   metadata={"source": "contract_2024.pdf"})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0],
                   metadata={"source": "invoice.pdf"})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("metadata.source").like("contract%")])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


@pytest.mark.asyncio
async def test_search_filter_metadata_numeric_gte(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"page": 1})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0], metadata={"page": 3})
    c = make_chunk(doc_id="doc-1", embedding=[0.8, 0.2, 0.0, 0.0], metadata={"page": 5})
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("metadata.page") >= 3])
    assert len(results) == 2
    assert {r.metadata["page"] for r in results} == {3, 5}


@pytest.mark.asyncio
async def test_search_filter_metadata_is_null(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"page": 1})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0], metadata={})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("metadata.page").is_null()])
    assert len(results) == 1
    assert results[0].chunk_id == b.chunk_id


@pytest.mark.asyncio
async def test_search_filter_no_match_returns_empty(store):
    await store.upsert(COLLECTION, [make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0])])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "no-such-doc"])
    assert results == []


@pytest.mark.asyncio
async def test_search_filter_respects_top_k(store):
    """top_k is applied after filtering."""
    await store.upsert(COLLECTION, [
        make_chunk(doc_id="doc-1", embedding=[float(i), 0.0, 0.0, 0.0]) for i in range(1, 6)
    ])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=2,
                                 filters=[Col("doc_id") == "doc-1"])
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_multiple_filters_are_anded(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-1", embedding=[0.9, 0.1, 0.0, 0.0], metadata={"section": "body"})
    c = make_chunk(doc_id="doc-2", embedding=[0.8, 0.2, 0.0, 0.0], metadata={"section": "intro"})
    await store.upsert(COLLECTION, [a, b, c])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5, filters=[
        Col("doc_id") == "doc-1",
        Col("metadata.section") == "intro",
    ])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id


# ---------------------------------------------------------------------------
# Field projection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_fields_omits_embedding(store):
    chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"k": "v"})
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1,
                                 fields=["chunk_id", "doc_id", "text", "metadata"])
    assert results[0].embedding is None
    assert results[0].metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_search_fields_omits_metadata(store):
    chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"k": "v"})
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1,
                                 fields=["chunk_id", "doc_id", "text", "embedding"])
    assert results[0].metadata == {}
    assert results[0].embedding is not None


@pytest.mark.asyncio
async def test_search_fields_none_returns_all(store):
    chunk = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"k": "v"})
    await store.upsert(COLLECTION, [chunk])
    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=1, fields=None)
    assert results[0].embedding is not None
    assert results[0].metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_search_filters_and_fields_combined(store):
    a = make_chunk(doc_id="doc-1", embedding=[1.0, 0.0, 0.0, 0.0], metadata={"section": "intro"})
    b = make_chunk(doc_id="doc-2", embedding=[0.9, 0.1, 0.0, 0.0], metadata={"section": "body"})
    await store.upsert(COLLECTION, [a, b])

    results = await store.search(COLLECTION, [1.0, 0.0, 0.0, 0.0], top_k=5,
                                 filters=[Col("doc_id") == "doc-1"],
                                 fields=["chunk_id", "doc_id", "text", "metadata"])
    assert len(results) == 1
    assert results[0].chunk_id == a.chunk_id
    assert results[0].embedding is None
    assert results[0].metadata == {"section": "intro"}
