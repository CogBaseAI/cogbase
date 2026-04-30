"""Pipeline test fixtures — parametrized over store configs via the factory API."""

from __future__ import annotations

from typing import Callable

import pytest

from cogbase.config.stores import StructuredStoreConfig, VectorStoreConfig
from cogbase.stores import build_structured_store, build_vector_store
from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.vector.base import VectorStoreBase

_VECTOR_STORE_CONFIGS = [
    pytest.param(VectorStoreConfig(type="faiss"), id="faiss"),
]

_STRUCTURED_STORE_CONFIGS = [
    pytest.param(StructuredStoreConfig(type="sqlite", path=":memory:"), id="sqlite"),
]


@pytest.fixture(params=_VECTOR_STORE_CONFIGS)
def make_vector_store(request) -> Callable[[], VectorStoreBase]:
    """Returns a factory callable that creates a fresh vector store per call."""
    cfg = request.param
    return lambda: build_vector_store(cfg)


@pytest.fixture(params=_STRUCTURED_STORE_CONFIGS)
def make_structured_store(request) -> Callable[[], StructuredStoreBase]:
    """Returns a factory callable that creates a fresh structured store per call."""
    cfg = request.param
    return lambda: build_structured_store(cfg)
