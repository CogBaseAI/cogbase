"""Tests for cogbase.stores.factory."""

from types import SimpleNamespace

import pytest

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.stores.document.local_fs import LocalFSDocumentStore
from cogbase.stores.factory import (
    build_document_store,
    build_structured_store,
    build_vector_store,
)
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSVectorStore


def test_build_structured_store_memory():
    store = build_structured_store(StructuredStoreConfig(type="memory"))
    assert isinstance(store, InMemoryStructuredStore)


def test_build_vector_store_faiss():
    store = build_vector_store(VectorStoreConfig(type="faiss", dim=128))
    assert isinstance(store, FAISSVectorStore)


def test_build_document_store_local(tmp_path):
    cfg = DocumentStoreConfig(type="local", path=str(tmp_path / "docs"))
    store = build_document_store(cfg)
    assert isinstance(store, LocalFSDocumentStore)


def test_build_structured_store_unknown_type_raises():
    cfg = SimpleNamespace(type="unknown", path=None, url=None)
    with pytest.raises(ValueError, match="Unknown structured_store type"):
        build_structured_store(cfg)
