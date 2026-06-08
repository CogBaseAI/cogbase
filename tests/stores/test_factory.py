"""Tests for cogbase.stores.factory."""

from types import SimpleNamespace

import pytest

from cogbase.config.stores import (
    DocumentStoreConfig,
    LogStoreConfig,
    StructuredStoreConfig,
    VectorStoreConfig,
)
from cogbase.stores import (
    AppScope,
    build_document_store,
    build_log_store,
    build_structured_store,
    build_vector_store,
)
from cogbase.stores.document.local_fs import LocalFSDocumentStore
from cogbase.stores.log.local_fs import LocalFSLogStore
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


def test_build_structured_store_memory():
    store = build_structured_store(StructuredStoreConfig(type="memory"))
    assert isinstance(store, InMemoryStructuredStore)


def test_build_vector_store_faiss():
    store = build_vector_store(VectorStoreConfig(type="faiss"))
    assert isinstance(store, FAISSVectorStore)


def test_build_vector_store_faiss_uses_path(tmp_path):
    path = tmp_path / "faiss"
    store = build_vector_store(VectorStoreConfig(type="faiss", path=str(path)))
    assert isinstance(store, FAISSVectorStore)
    assert store.path == path


def test_build_document_store_local(tmp_path):
    cfg = DocumentStoreConfig(type="local", path=str(tmp_path / "docs"))
    store = build_document_store(cfg)
    assert isinstance(store, LocalFSDocumentStore)


def test_build_log_store_local(tmp_path):
    cfg = LogStoreConfig(type="local", path=str(tmp_path / "logs"))
    store = build_log_store(cfg)
    assert isinstance(store, LocalFSLogStore)


def test_build_structured_store_unknown_type_raises():
    cfg = SimpleNamespace(type="unknown", path=None, url=None)
    with pytest.raises(ValueError, match="Unknown structured_store type"):
        build_structured_store(cfg)


def test_build_log_store_unknown_type_raises():
    cfg = SimpleNamespace(type="unknown", path=None, bucket=None, prefix="", region=None)
    with pytest.raises(ValueError, match="Unknown log_store type"):
        build_log_store(cfg)


# ---------------------------------------------------------------------------
# Scope parameter
# ---------------------------------------------------------------------------

def test_build_structured_store_with_scope():
    scope = AppScope(app="myapp")
    store = build_structured_store(StructuredStoreConfig(type="memory"), scope=scope)
    assert isinstance(store, InMemoryStructuredStore)
    assert store._scope == scope
    assert store._c("contracts") == "myapp__contracts"


def test_build_vector_store_faiss_with_scope():
    scope = AppScope(app="myapp")
    store = build_vector_store(VectorStoreConfig(type="faiss"), scope=scope)
    assert isinstance(store, FAISSVectorStore)
    assert store._scope == scope
    assert store._c("chunks") == "myapp__chunks"


def test_build_document_store_local_with_scope(tmp_path):
    scope = AppScope(namespace="eng", app="myapp")
    store = build_document_store(
        DocumentStoreConfig(type="local", path=str(tmp_path / "docs")), scope=scope
    )
    assert isinstance(store, LocalFSDocumentStore)
    assert store._scope == scope
    assert store._c("uploads") == "eng__myapp__uploads"


def test_build_log_store_local_with_scope(tmp_path):
    scope = AppScope(namespace="eng", app="myapp")
    store = build_log_store(
        LogStoreConfig(type="local", path=str(tmp_path / "logs")), scope=scope
    )
    assert isinstance(store, LocalFSLogStore)
    assert store._scope == scope
    assert store._c("episodic") == "eng__myapp__episodic"


def test_build_without_scope_defaults_to_no_prefix():
    store = build_structured_store(StructuredStoreConfig(type="memory"))
    assert store._c("contracts") == "contracts"
