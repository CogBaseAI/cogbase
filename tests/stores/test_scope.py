"""Tests for AppScope, _c(), and scoped store proxies."""

import pytest

from cogbase.core.models import Chunk
from cogbase.stores import AppScope, Col, CollectionSchema, FieldSchema, FieldType, VectorCollectionSchema
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.document.local_fs import LocalFSDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.structured.sqlite import SQLiteStructuredStore
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore, FAISSVectorStore


# ---------------------------------------------------------------------------
# AppScope
# ---------------------------------------------------------------------------

def test_app_only_prefix():
    assert AppScope(app_id="myapp").prefix() == "myapp"


def test_namespace_and_app_prefix():
    assert AppScope(namespace_id="eng", app_id="myapp").prefix() == "eng__myapp"


def test_full_hierarchy_prefix():
    assert AppScope(account_id="acme", namespace_id="eng", app_id="myapp").prefix() == "acme__eng__myapp"


def test_all_none_prefix_returns_none():
    assert AppScope().prefix() is None


def test_partial_none_skipped():
    assert AppScope(account_id="acme", app_id="myapp").prefix() == "acme__myapp"


def test_custom_separator():
    assert AppScope(account_id="acme", app_id="myapp").prefix(sep=".") == "acme.myapp"


# ---------------------------------------------------------------------------
# _c() on store bases
# ---------------------------------------------------------------------------

def test_vector_store_c_no_scope():
    store = FAISSMemoryVectorStore()
    assert store._c("chunks") == "chunks"


def test_vector_store_c_with_scope():
    store = FAISSMemoryVectorStore(scope=AppScope(app_id="myapp"))
    assert store._c("chunks") == "myapp__chunks"


def test_structured_store_c_no_scope():
    store = InMemoryStructuredStore()
    assert store._c("contracts") == "contracts"


def test_structured_store_c_with_scope():
    store = InMemoryStructuredStore(scope=AppScope(namespace_id="eng", app_id="myapp"))
    assert store._c("contracts") == "eng__myapp__contracts"


def test_document_store_c_no_scope():
    store = InMemoryDocumentStore()
    assert store._c("docs") == "docs"


def test_document_store_c_with_scope():
    store = InMemoryDocumentStore(scope=AppScope(app_id="myapp"))
    assert store._c("docs") == "myapp__docs"


# ---------------------------------------------------------------------------
# Native scope — InMemoryStructuredStore
# ---------------------------------------------------------------------------

SCHEMA = CollectionSchema(
    name="contracts",
    description="contract records",
    primary_fields=["doc_id"],
    fields={
        "doc_id": FieldSchema(type=FieldType.STRING, nullable=False),
        "value":  FieldSchema(type=FieldType.STRING),
    },
)


async def test_scoped_structured_store_uses_prefixed_frame_key():
    store = InMemoryStructuredStore(scope=AppScope(app_id="myapp"))
    await store.create_collection(SCHEMA)
    assert "myapp__contracts" in store._frames
    assert "contracts" not in store._frames


async def test_scoped_structured_store_save_and_query():
    store = InMemoryStructuredStore(scope=AppScope(app_id="myapp"))
    await store.create_collection(SCHEMA)
    await store.save("contracts", [{"doc_id": "x", "value": "hello"}])
    rows = await store.query("contracts")
    assert rows == [{"doc_id": "x", "value": "hello"}]


async def test_two_scoped_structured_stores_do_not_conflict():
    """Two apps with scoped stores sharing the same underlying instance."""
    raw = InMemoryStructuredStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))

    await a.create_collection(SCHEMA)
    await b.create_collection(SCHEMA)

    await a.save("contracts", [{"doc_id": "x", "value": "from_a"}])
    await b.save("contracts", [{"doc_id": "x", "value": "from_b"}])

    assert (await a.query("contracts")) == [{"doc_id": "x", "value": "from_a"}]
    assert (await b.query("contracts")) == [{"doc_id": "x", "value": "from_b"}]

    # Confirm isolation at the raw frame level
    assert "app_a__contracts" in raw._frames
    assert "app_b__contracts" in raw._frames


async def test_scoped_structured_delete_records_isolated():
    raw = InMemoryStructuredStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))
    await a.create_collection(SCHEMA)
    await b.create_collection(SCHEMA)
    await a.save("contracts", [{"doc_id": "x", "value": "a"}])
    await b.save("contracts", [{"doc_id": "x", "value": "b"}])

    await a.delete_records("contracts", [Col("doc_id") == "x"])

    assert await a.query("contracts") == []
    assert (await b.query("contracts")) == [{"doc_id": "x", "value": "b"}]


async def test_scoped_structured_delete_collection_isolated():
    raw = InMemoryStructuredStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))
    await a.create_collection(SCHEMA)
    await b.create_collection(SCHEMA)
    await a.save("contracts", [{"doc_id": "x", "value": "a"}])
    await b.save("contracts", [{"doc_id": "x", "value": "b"}])

    await a.delete_collection("contracts")

    assert "app_a__contracts" not in raw._frames
    assert "app_b__contracts" in raw._frames


# ---------------------------------------------------------------------------
# Native scope — SQLiteStructuredStore
# ---------------------------------------------------------------------------

async def test_sqlite_scoped_table_name(tmp_path):
    store = SQLiteStructuredStore(tmp_path / "test.db", scope=AppScope(app_id="myapp"))
    await store.create_collection(SCHEMA)
    tables = {row[0] for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "myapp__contracts" in tables
    assert "contracts" not in tables
    store.close()


async def test_sqlite_two_scoped_apps_no_conflict(tmp_path):
    store_a = SQLiteStructuredStore(tmp_path / "a.db", scope=AppScope(app_id="app_a"))
    store_b = SQLiteStructuredStore(tmp_path / "b.db", scope=AppScope(app_id="app_b"))
    await store_a.create_collection(SCHEMA)
    await store_b.create_collection(SCHEMA)
    await store_a.save("contracts", [{"doc_id": "x", "value": "a"}])
    await store_b.save("contracts", [{"doc_id": "x", "value": "b"}])
    assert (await store_a.query("contracts")) == [{"doc_id": "x", "value": "a"}]
    assert (await store_b.query("contracts")) == [{"doc_id": "x", "value": "b"}]
    store_a.close()
    store_b.close()


# ---------------------------------------------------------------------------
# Native scope — FAISSMemoryVectorStore
# ---------------------------------------------------------------------------

VSCHEMA = VectorCollectionSchema(name="chunks", dimensions=2, description="test chunks")


async def test_faiss_scoped_uses_prefixed_key():
    store = FAISSMemoryVectorStore(scope=AppScope(app_id="myapp"))
    await store.create_collection(VSCHEMA)
    assert "myapp__chunks" in store._collections
    assert "chunks" not in store._collections


async def test_faiss_two_scoped_stores_do_not_conflict():
    raw = FAISSMemoryVectorStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))

    await a.create_collection(VSCHEMA)
    await b.create_collection(VSCHEMA)

    chunk_a = Chunk(chunk_id="ca", doc_id="doc-a", text="t", embedding=[1.0, 0.0])
    chunk_b = Chunk(chunk_id="cb", doc_id="doc-b", text="t", embedding=[0.0, 1.0])
    await a.upsert("chunks", [chunk_a])
    await b.upsert("chunks", [chunk_b])

    res_a = await a.search("chunks", "q", [1.0, 0.0], top_k=5)
    res_b = await b.search("chunks", "q", [0.0, 1.0], top_k=5)

    assert len(res_a) == 1 and res_a[0].doc_id == "doc-a"
    assert len(res_b) == 1 and res_b[0].doc_id == "doc-b"

    assert "app_a__chunks" in raw._collections
    assert "app_b__chunks" in raw._collections


async def test_faiss_scoped_delete_isolated():
    raw = FAISSMemoryVectorStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))
    await a.create_collection(VSCHEMA)
    await b.create_collection(VSCHEMA)
    await a.upsert("chunks", [Chunk(chunk_id="ca", doc_id="d1", text="t", embedding=[1.0, 0.0])])
    await b.upsert("chunks", [Chunk(chunk_id="cb", doc_id="d1", text="t", embedding=[0.0, 1.0])])

    await a.delete("chunks", ["ca"])

    assert raw.ntotal("app_a__chunks") == 0
    assert raw.ntotal("app_b__chunks") == 1


async def test_faiss_scoped_delete_doc_isolated():
    raw = FAISSMemoryVectorStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))
    await a.create_collection(VSCHEMA)
    await b.create_collection(VSCHEMA)
    await a.upsert("chunks", [Chunk(chunk_id="ca", doc_id="d1", text="t", embedding=[1.0, 0.0])])
    await b.upsert("chunks", [Chunk(chunk_id="cb", doc_id="d1", text="t", embedding=[0.0, 1.0])])

    await a.delete_doc("chunks", "d1")

    assert raw.ntotal("app_a__chunks") == 0
    assert raw.ntotal("app_b__chunks") == 1


async def test_faiss_scoped_file_store_roundtrip(tmp_path):
    scope = AppScope(app_id="myapp")
    store = FAISSVectorStore(path=tmp_path / "store", scope=scope)
    await store.create_collection(VSCHEMA)
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="hello", embedding=[1.0, 0.0])
    await store.upsert("chunks", [chunk])

    # Reload with the same scope — should find the scoped file
    loaded = FAISSVectorStore(path=tmp_path / "store", scope=scope)
    results = await loaded.search("chunks", "q", [1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id


# ---------------------------------------------------------------------------
# Native scope — Document stores
# ---------------------------------------------------------------------------

async def test_in_memory_doc_scoped_isolation():
    raw = InMemoryDocumentStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))

    await a.save("docs", "file1", "content_a")
    await b.save("docs", "file1", "content_b")

    assert await a.load("docs", "file1") == "content_a"
    assert await b.load("docs", "file1") == "content_b"

    assert ("app_a__docs", "file1") in raw._store
    assert ("app_b__docs", "file1") in raw._store


async def test_in_memory_doc_scoped_delete_isolated():
    raw = InMemoryDocumentStore()
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))

    await a.save("docs", "file1", "a_content")
    await b.save("docs", "file1", "b_content")

    await a.delete("docs", "file1")
    assert not await a.exists("docs", "file1")
    assert await b.exists("docs", "file1")


async def test_local_fs_scoped_path_uses_prefix(tmp_path):
    store = LocalFSDocumentStore(tmp_path, scope=AppScope(app_id="myapp"))
    await store.save("docs", "file1.txt", "hello")
    # The file should live under myapp__docs/, not docs/
    assert (tmp_path / "myapp__docs" / "file1.txt").exists()
    assert not (tmp_path / "docs" / "file1.txt").exists()


async def test_local_fs_two_scoped_apps_isolated(tmp_path):
    raw = LocalFSDocumentStore(tmp_path)
    a = raw.with_scope(AppScope(app_id="app_a"))
    b = raw.with_scope(AppScope(app_id="app_b"))

    await a.save("docs", "file1.txt", "a_content")
    await b.save("docs", "file1.txt", "b_content")

    assert await a.load("docs", "file1.txt") == "a_content"
    assert await b.load("docs", "file1.txt") == "b_content"
    assert (tmp_path / "app_a__docs" / "file1.txt").exists()
    assert (tmp_path / "app_b__docs" / "file1.txt").exists()


# ---------------------------------------------------------------------------
# with_scope returns independent proxies
# ---------------------------------------------------------------------------

async def test_with_scope_does_not_modify_original_store():
    """Calling with_scope() should not alter the underlying store's scope."""
    raw = InMemoryStructuredStore()
    assert raw._scope is None
    _ = raw.with_scope(AppScope(app_id="myapp"))
    assert raw._scope is None  # raw store is unaffected


async def test_register_schema_via_scoped_proxy():
    raw = InMemoryStructuredStore()
    proxy = raw.with_scope(AppScope(app_id="myapp"))
    proxy.register_schema(SCHEMA)
    # Proxy has bare schema; raw store has scoped schema
    assert "contracts" in proxy._schemas
    assert "myapp__contracts" in raw._schemas
