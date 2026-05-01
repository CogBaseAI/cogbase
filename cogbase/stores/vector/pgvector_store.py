"""pgvector implementation of VectorStoreBase.

Requires the ``asyncpg`` and ``pgvector`` packages::

    pip install "cogbase[pgvector]"

Also requires the pgvector extension in PostgreSQL::

    CREATE EXTENSION IF NOT EXISTS vector;

Similarity metric: cosine distance (``<=>`` operator).  Vectors are stored as
the pgvector ``vector`` type and queried with ``ORDER BY embedding <=> $query``.

Usage::

    store = PGVectorStore(dsn="postgresql://user:pass@localhost/mydb")
    await store.connect()
    schema = VectorCollectionSchema(name="chunks", dimensions=1536, description="Full-text passage chunks")
    await store.create_collection(schema)
    await store.upsert("chunks", chunks)
    results = await store.search("chunks", query_text, query_embedding, top_k=10)
    await store.close()

    # --- or as async context manager ---

    async with PGVectorStore(dsn="postgresql://localhost/mydb") as store:
        ...
"""

from __future__ import annotations

import logging
from typing import Any

from cogbase.core.models import Chunk
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase
from cogbase.stores.filters import Filter, Op

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


class PGVectorStore(VectorStoreBase):
    """Vector store backed by PostgreSQL + pgvector.

    One table is created per collection via ``create_collection``.  Each table
    has the following columns:

    ==================  ============================================================
    chunk_id            TEXT PRIMARY KEY
    doc_id              TEXT NOT NULL
    text                TEXT NOT NULL
    embedding           vector(dimensions) NOT NULL
    metadata            JSONB NOT NULL DEFAULT '{}'
    ==================  ============================================================

    An HNSW index is created on ``embedding`` and a B-tree index on ``doc_id``
    when ``create_collection`` is called.

    Args:
        dsn:  asyncpg connection DSN.  Either ``dsn`` or ``pool`` must be given.
        pool: An existing ``asyncpg.Pool``.  Pass either ``dsn`` or ``pool``,
              not both.
    """

    def __init__(
        self,
        dsn: str | None = None,
        pool: "asyncpg.Pool | None" = None,
    ) -> None:
        if dsn is None and pool is None:
            raise ValueError("Provide either dsn or pool.")
        if dsn is not None and pool is not None:
            raise ValueError("Provide either dsn or pool, not both.")
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = pool
        self._collection_names: set[str] = set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool and register the pgvector codec.

        A no-op if a pool was passed at construction.
        """
        if self._pool is None:
            conn = await asyncpg.connect(self._dsn)
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await conn.close()
            self._pool = await asyncpg.create_pool(
                self._dsn,
                init=register_vector,
            )

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
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """Create the table and indexes for ``schema.name``.  Idempotent."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{schema.name}" (
                    chunk_id  TEXT PRIMARY KEY,
                    doc_id    TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    embedding vector({schema.dimensions}) NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{{}}'
                )
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS "{schema.name}_embedding_hnsw_idx"
                ON "{schema.name}"
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{schema.name}_doc_id_idx" '
                f'ON "{schema.name}" (doc_id)'
            )
        self._collection_names.add(schema.name)

    async def delete_collection(self, collection: str) -> None:
        """Drop the table for ``collection``.  Idempotent — no-op if absent."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS "{collection}"')
        self._collection_names.discard(collection)

    async def list_collections(self) -> list[str]:
        return sorted(self._collection_names)

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Add or replace chunks in ``collection``.

        Chunks without an embedding are silently skipped.
        """
        incoming = [c for c in chunks if c.embedding is not None]
        if not incoming:
            return

        pool = self._get_pool()
        rows = [_to_row(c) for c in incoming]
        sql = (
            f'INSERT INTO "{collection}" (chunk_id, doc_id, text, embedding, metadata) '
            f"VALUES ($1, $2, $3, $4, $5) "
            f"ON CONFLICT (chunk_id) DO UPDATE SET "
            f"  doc_id    = EXCLUDED.doc_id, "
            f"  text      = EXCLUDED.text, "
            f"  embedding = EXCLUDED.embedding, "
            f"  metadata  = EXCLUDED.metadata"
        )
        async with pool.acquire() as conn:
            await conn.executemany(sql, rows)

    async def search(
        self,
        collection: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[Chunk]:
        """Return up to ``top_k`` chunks from ``collection`` ordered by cosine similarity."""
        import numpy as np

        pool = self._get_pool()
        vec = np.array(query_embedding, dtype=np.float32)

        # $1 = query vector (used in ORDER BY); filter params follow from $2 onward.
        params: list[Any] = [vec]
        where_sql = ""
        if filters:
            where_clause, filter_params = _build_pg_where(filters, param_offset=len(params))
            params.extend(filter_params)
            where_sql = f"WHERE {where_clause}"
        params.append(top_k)

        include_embedding = not fields or "embedding" in fields
        include_metadata = not fields or "metadata" in fields
        select_cols = "chunk_id, doc_id, text"
        if include_embedding:
            select_cols += ", embedding"
        if include_metadata:
            select_cols += ", metadata"

        sql = (
            f"SELECT {select_cols} "
            f'FROM "{collection}" '
            f"{where_sql} "
            f"ORDER BY embedding <=> $1 "
            f"LIMIT ${len(params)}"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_from_row(row, include_embedding, include_metadata) for row in rows]

    async def delete(self, collection: str, doc_id: str) -> None:
        """Remove all chunks for ``doc_id`` from ``collection``."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f'DELETE FROM "{collection}" WHERE doc_id = $1',
                doc_id,
            )

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


def _from_row(row: Any, include_embedding: bool = True, include_metadata: bool = True) -> Chunk:
    import json

    embedding = list(row["embedding"]) if include_embedding else None
    if include_metadata:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
    else:
        metadata = {}

    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        embedding=embedding,
        metadata=metadata,
    )


# Top-level Chunk columns that can be filtered directly in SQL.
_DIRECT_COLS = {"chunk_id", "doc_id", "text"}


def _build_pg_where(filters: list[Filter], param_offset: int) -> tuple[str, list[Any]]:
    """Translate a Filter list to a parameterised PostgreSQL WHERE clause fragment.

    Supports top-level Chunk columns (``chunk_id``, ``doc_id``, ``text``) and
    dot-notation for JSONB metadata sub-keys (``metadata.key``).

    Args:
        filters:      Filter expressions to translate.
        param_offset: Number of params already bound before these (e.g. 1 if $1
                      is the query vector).  Filter params start at $param_offset+1.

    Returns:
        ``(where_clause, params)`` — clause is a non-empty string ready to be
        placed after ``WHERE``; params are the corresponding bound values.
    """
    clauses: list[str] = []
    params: list[Any] = []
    n = param_offset + 1  # 1-based asyncpg param index

    for f in filters:
        is_meta = "." in f.field and f.field.startswith("metadata.")
        if is_meta:
            key = f.field.split(".", 1)[1]
            text_expr = f"metadata->>'{key}'"
            num_expr = f"(metadata->>'{key}')::numeric"
        elif f.field in _DIRECT_COLS:
            text_expr = f'"{f.field}"'
            num_expr = f'"{f.field}"'
        else:
            continue  # unknown field — skip

        match f.op:
            case Op.EQ:
                clauses.append(f"{text_expr} = ${n}")
                params.append(str(f.value) if is_meta else f.value)
                n += 1
            case Op.NE:
                clauses.append(f"{text_expr} != ${n}")
                params.append(str(f.value) if is_meta else f.value)
                n += 1
            case Op.LT:
                clauses.append(f"{num_expr} < ${n}")
                params.append(f.value)
                n += 1
            case Op.GT:
                clauses.append(f"{num_expr} > ${n}")
                params.append(f.value)
                n += 1
            case Op.LTE:
                clauses.append(f"{num_expr} <= ${n}")
                params.append(f.value)
                n += 1
            case Op.GTE:
                clauses.append(f"{num_expr} >= ${n}")
                params.append(f.value)
                n += 1
            case Op.IN:
                values = [str(v) for v in f.value] if is_meta else list(f.value)
                clauses.append(f"{text_expr} = ANY(${n})")
                params.append(values)
                n += 1
            case Op.NOT_IN:
                values = [str(v) for v in f.value] if is_meta else list(f.value)
                clauses.append(f"NOT ({text_expr} = ANY(${n}))")
                params.append(values)
                n += 1
            case Op.LIKE:
                clauses.append(f"{text_expr} LIKE ${n}")
                params.append(f.value)
                n += 1
            case Op.IS_NULL:
                clauses.append(f"{text_expr} IS NULL")
            case Op.IS_NOT_NULL:
                clauses.append(f"{text_expr} IS NOT NULL")

    return (" AND ".join(clauses), params)
