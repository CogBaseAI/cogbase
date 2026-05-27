"""Store builders from typed config-like objects.

This module centralizes backend selection so callers can stay config-driven
without importing concrete store implementations directly.
"""

from __future__ import annotations

from cogbase.config.stores import DocumentStoreConfig, StructuredStoreConfig, VectorStoreConfig
from cogbase.stores.document.base import DocumentStoreBase
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.vector.base import VectorStoreBase


def build_structured_store(
    cfg: StructuredStoreConfig, scope: AppScope | None = None
) -> StructuredStoreBase:
    """Instantiate a structured store from config."""
    if cfg.type == "memory":
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        return InMemoryStructuredStore(scope=scope)
    if cfg.type == "sqlite":
        from cogbase.stores.structured.sqlite import SQLiteStructuredStore
        return SQLiteStructuredStore(cfg.path, scope=scope)
    if cfg.type == "postgres":
        from cogbase.stores.structured.postgres import PostgresStructuredStore
        return PostgresStructuredStore(cfg.url, scope=scope)
    raise ValueError(f"Unknown structured_store type: {cfg.type!r}")


def build_vector_store(
    cfg: VectorStoreConfig, scope: AppScope | None = None
) -> VectorStoreBase:
    """Instantiate a vector store from config."""
    if cfg.type == "faiss":
        from cogbase.stores.vector.faiss_store import FAISSVectorStore
        return FAISSVectorStore(path=cfg.path, scope=scope)
    if cfg.type == "pgvector":
        from cogbase.stores.vector.pgvector_store import PGVectorStore
        return PGVectorStore(dsn=cfg.url, scope=scope)
    raise ValueError(f"Unknown vector_store type: {cfg.type!r}")


def build_document_store(
    cfg: DocumentStoreConfig, scope: AppScope | None = None
) -> DocumentStoreBase:
    """Instantiate a document store from config."""
    if cfg.type == "local":
        from cogbase.stores.document.local_fs import LocalFSDocumentStore
        return LocalFSDocumentStore(cfg.path, scope=scope)
    if cfg.type == "s3":
        from cogbase.stores.document.s3 import S3DocumentStore
        return S3DocumentStore(bucket=cfg.bucket, prefix=cfg.prefix, region=cfg.region, scope=scope)
    raise ValueError(f"Unknown document_store type: {cfg.type!r}")
