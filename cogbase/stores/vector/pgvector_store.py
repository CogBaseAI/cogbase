"""pgvector implementation of VectorStoreBase.

Requires the ``asyncpg`` and ``pgvector`` packages::

    pip install "cogbase[pgvector]"

Also requires the pgvector extension in PostgreSQL::

    CREATE EXTENSION IF NOT EXISTS vector;

Similarity metric: cosine distance (``<=>`` operator).  Vectors are stored as
the pgvector ``vector`` type and queried with ``ORDER BY embedding <=> $query``.

Usage::

    store = PGVectorStore(dsn="postgresql://user:pass@localhost/mydb", dim=1536)
    await store.connect()
    await store.upsert(chunks)
    results = await store.search(query_embedding, top_k=10)
    await store.close()

    # --- or as async context manager ---

    async with PGVectorStore(dsn="postgresql://localhost/mydb", dim=1536) as store:
        ...
"""

from __future__ import annotations

import logging
from typing import Any

from cogbase.core.models import Chunk
from cogbase.stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "asyncpg is required for PGVectorStore. "
        'Install it with: pip install "cogbase[pgvector]"'
    ) from exc

try:
    from pgvector.asyncpg import register_vector
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pgvector is required for PGVectorStore. "
        'Install it with: pip install "cogbase[pgvector]"'
    ) from exc


_TABLE = "cogbase_chunks"


class PGVectorStore(VectorStoreBase):
    """Vector store backed by PostgreSQL + pgvector.

    Similarity metric: cosine distance (nearest-neighbour with ``<=>``).

    The store manages a single table, ``cogbase_chunks``, with the following
    columns:

    ==================  ============================================================
    chunk_id            TEXT PRIMARY KEY
    doc_id              TEXT NOT NULL
    text                TEXT NOT NULL
    embedding           vector(dim) NOT NULL
    metadata            JSONB NOT NULL DEFAULT '{}'
    ==================  ============================================================

    An HNSW index is created on the ``embedding`` column for efficient ANN
    search once ``create_table`` is called.

    Args:
        dim: Embedding dimension.  Must be provided at construction time.
        dsn: asyncpg connection DSN.  Either ``dsn`` or ``pool`` must be given.
        pool: An existing ``asyncpg.Pool``.  Pass either ``dsn`` or ``pool``,
              not both.
        table: Override the default table name (``cogbase_chunks``).
    """

    def __init__(
        self,
        dim: int,
        dsn: str | None = None,
        pool: "asyncpg.Pool | None" = None,
        table: str = _TABLE,
    ) -> None:
        if dsn is None and pool is None:
            raise ValueError("Provide either dsn or pool.")
        if dsn is not None and pool is not None:
            raise ValueError("Provide either dsn or pool, not both.")
        self._dim = dim
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = pool
        self._table = table

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool and register the pgvector codec.

        A no-op if a pool was passed at construction.
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                init=register_vector,
            )
        await self.create_table()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> "PGVectorStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------

    async def create_table(self) -> None:
        """Create the chunks table and HNSW index if they do not exist.

        Safe to call multiple times (idempotent).
        """
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{self._table}" (
                    chunk_id  TEXT PRIMARY KEY,
                    doc_id    TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    embedding vector({self._dim}) NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{{}}'
                )
                """
            )
            # HNSW index for approximate nearest-neighbour search with cosine distance.
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS "{self._table}_embedding_hnsw_idx"
                ON "{self._table}"
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{self._table}_doc_id_idx" '
                f'ON "{self._table}" (doc_id)'
            )

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Add or replace chunks.

        Chunks without an embedding are silently skipped.
        """
        incoming = [c for c in chunks if c.embedding is not None]
        if not incoming:
            return

        pool = self._get_pool()
        rows = [_to_row(c) for c in incoming]
        sql = (
            f'INSERT INTO "{self._table}" (chunk_id, doc_id, text, embedding, metadata) '
            f"VALUES ($1, $2, $3, $4, $5) "
            f"ON CONFLICT (chunk_id) DO UPDATE SET "
            f"  doc_id    = EXCLUDED.doc_id, "
            f"  text      = EXCLUDED.text, "
            f"  embedding = EXCLUDED.embedding, "
            f"  metadata  = EXCLUDED.metadata"
        )
        async with pool.acquire() as conn:
            await conn.executemany(sql, rows)

    async def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """Return up to ``top_k`` chunks ordered by cosine similarity (highest first)."""
        pool = self._get_pool()
        import numpy as np

        vec = np.array(query_embedding, dtype=np.float32)
        sql = (
            f'SELECT chunk_id, doc_id, text, embedding, metadata '
            f'FROM "{self._table}" '
            f'ORDER BY embedding <=> $1 '
            f'LIMIT $2'
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, vec, top_k)

        return [_from_row(row) for row in rows]

    async def delete(self, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id``."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f'DELETE FROM "{self._table}" WHERE doc_id = $1',
                doc_id,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimension this store was configured with."""
        return self._dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pool(self) -> "asyncpg.Pool":
        if self._pool is None:
            raise RuntimeError(
                "Not connected. Call await store.connect() or use as an async context manager."
            )
        return self._pool


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _to_row(chunk: Chunk) -> tuple[Any, ...]:
    import json
    import numpy as np

    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.text,
        np.array(chunk.embedding, dtype=np.float32),
        json.dumps(chunk.metadata),
    )


def _from_row(row: Any) -> Chunk:
    import json

    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        embedding=list(row["embedding"]),
        metadata=metadata,
    )
