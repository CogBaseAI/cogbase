"""PostgreSQL implementation of StructuredStoreBase.

Requires the ``asyncpg`` package (``pip install cogbase[postgres]``).

JSON fields use the native JSONB column type, so all filter operators are
pushed to the database — there is no Python post-filtering.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from cogbase.stores.base import StructuredStoreBase
from cogbase.stores.filters import Filter, Op
from cogbase.stores.schema import CollectionSchema, FieldType

try:
    import asyncpg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "asyncpg is required for PostgresStructuredStore. "
        "Install it with: pip install cogbase[postgres]"
    ) from exc


_PG_TYPE: dict[FieldType, str] = {
    FieldType.STRING:  "TEXT",
    FieldType.INTEGER: "BIGINT",
    FieldType.FLOAT:   "DOUBLE PRECISION",
    FieldType.BOOLEAN: "BOOLEAN",
    FieldType.JSON:    "JSONB",
}


class PostgresStructuredStore(StructuredStoreBase):
    """PostgreSQL-backed structured store.

    All filter operators — including filters on JSONB columns — are pushed to
    the database as parameterised SQL; no Python post-filtering is required.

    Args:
        dsn: asyncpg connection DSN, e.g.
             ``"postgresql://user:pass@localhost/dbname"``.
        pool: An existing ``asyncpg.Pool``.  Pass either ``dsn`` or ``pool``,
              not both.  If ``dsn`` is given, call ``await store.connect()``
              before first use (or use as an async context manager).

    Usage::

        store = PostgresStructuredStore(dsn="postgresql://localhost/mydb")
        await store.connect()

        # --- or ---

        async with PostgresStructuredStore(dsn="postgresql://localhost/mydb") as store:
            ...
    """

    def __init__(
        self,
        dsn: str | None = None,
        pool: "asyncpg.Pool | None" = None,  # type: ignore[name-defined]
    ) -> None:
        if dsn is None and pool is None:
            raise ValueError("Provide either dsn or pool.")
        if dsn is not None and pool is not None:
            raise ValueError("Provide either dsn or pool, not both.")
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = pool  # type: ignore[name-defined]
        self._schemas: dict[str, CollectionSchema] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool.  A no-op if a pool was passed at construction."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> "PostgresStructuredStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # StructuredStoreBase
    # ------------------------------------------------------------------

    async def create_collection(self, schema: CollectionSchema) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            # Create table if it does not exist.
            col_defs = _col_defs(schema)
            await conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{schema.name}" ({", ".join(col_defs)})'
            )

            # Add any columns that are in the schema but missing from the table.
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = $1",
                schema.name,
            )
            existing_cols: set[str] = {row["column_name"] for row in rows}
            for field_name, field in schema.fields.items():
                if field_name not in existing_cols:
                    pg_type = _PG_TYPE[field.type]
                    await conn.execute(
                        f'ALTER TABLE "{schema.name}" ADD COLUMN "{field_name}" {pg_type}'
                    )

            # Ensure indexes exist (idempotent via IF NOT EXISTS).
            for field_name, field in schema.fields.items():
                if field.index and field_name != schema.id_field:
                    idx_name = f"idx_{schema.name}_{field_name}"
                    await conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
                        f'ON "{schema.name}" ("{field_name}")'
                    )

        self._schemas[schema.name] = schema

    async def save(self, collection: str, records: list[BaseModel]) -> None:
        schema = self._get_schema(collection)
        pool = self._get_pool()
        cols = list(schema.fields.keys())
        col_list = ", ".join(f'"{c}"' for c in cols)

        # Build the ON CONFLICT … DO UPDATE SET … clause (upsert).
        update_cols = [c for c in cols if c != schema.id_field]
        if update_cols:
            update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
            conflict_clause = (
                f'ON CONFLICT ("{schema.id_field}") DO UPDATE SET {update_set}'
            )
        else:
            # Only the PK column exists — nothing to update on conflict.
            conflict_clause = f'ON CONFLICT ("{schema.id_field}") DO NOTHING'

        placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
        sql = (
            f'INSERT INTO "{collection}" ({col_list}) VALUES ({placeholders}) '
            f"{conflict_clause}"
        )

        rows = [_to_pg_row(r, schema) for r in records]
        async with pool.acquire() as conn:
            await conn.executemany(sql, rows)

    async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]:
        schema = self._get_schema(collection)
        pool = self._get_pool()
        where, params = _to_pg_where(filters or [], schema)
        sql = f'SELECT * FROM "{collection}"'
        if where:
            sql += f" WHERE {where}"

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_from_pg_row(dict(row), schema) for row in rows]

    async def update_collection(self, schema: CollectionSchema) -> None:
        old_schema = self._get_schema(schema.name)
        if schema.id_field != old_schema.id_field:
            raise ValueError(
                f"Cannot change id_field from '{old_schema.id_field}' to "
                f"'{schema.id_field}' — update_collection does not support "
                "primary-key migration"
            )

        pool = self._get_pool()
        old_fields = set(old_schema.fields)
        new_fields = set(schema.fields)
        added = new_fields - old_fields
        removed = old_fields - new_fields

        async with pool.acquire() as conn:
            async with conn.transaction():
                for field_name in added:
                    pg_type = _PG_TYPE[schema.fields[field_name].type]
                    await conn.execute(
                        f'ALTER TABLE "{schema.name}" ADD COLUMN "{field_name}" {pg_type}'
                    )
                for field_name in removed:
                    await conn.execute(
                        f'ALTER TABLE "{schema.name}" DROP COLUMN "{field_name}"'
                    )

                # Ensure indexes exist for all indexed fields.
                for field_name, field in schema.fields.items():
                    if field.index and field_name != schema.id_field:
                        idx_name = f"idx_{schema.name}_{field_name}"
                        await conn.execute(
                            f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
                            f'ON "{schema.name}" ("{field_name}")'
                        )

        self._schemas[schema.name] = schema

    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        schema = self._get_schema(collection)
        pool = self._get_pool()
        where, params = _to_pg_where(filters or [], schema)
        sql = f'DELETE FROM "{collection}"'
        if where:
            sql += f" WHERE {where}"
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    # ------------------------------------------------------------------

    def _get_schema(self, collection: str) -> CollectionSchema:
        if collection not in self._schemas:
            raise KeyError(f"Collection '{collection}' not found. Call create_collection first.")
        return self._schemas[collection]

    def _get_pool(self) -> "asyncpg.Pool":  # type: ignore[name-defined]
        if self._pool is None:
            raise RuntimeError(
                "Not connected. Call await store.connect() or use as an async context manager."
            )
        return self._pool


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _to_pg_row(record: BaseModel, schema: CollectionSchema) -> tuple:
    raw = record.model_dump(mode="json")
    row = []
    for field_name, field in schema.fields.items():
        val = raw.get(field_name)
        if val is None:
            row.append(None)
        elif field.type == FieldType.JSON:
            # asyncpg accepts a JSON string for JSONB columns.
            row.append(json.dumps(val))
        else:
            row.append(val)
    return tuple(row)


def _from_pg_row(row: dict, schema: CollectionSchema) -> dict:
    result: dict = {}
    for field_name, field in schema.fields.items():
        val = row.get(field_name)
        if val is None:
            result[field_name] = None
        elif field.type == FieldType.JSON:
            # asyncpg returns JSONB as a Python dict/list already.
            result[field_name] = val if not isinstance(val, str) else json.loads(val)
        else:
            result[field_name] = val
    return result


def _col_defs(schema: CollectionSchema) -> list[str]:
    defs: list[str] = []
    for field_name, field in schema.fields.items():
        pg_type = _PG_TYPE[field.type]
        not_null = "" if field.nullable else " NOT NULL"
        if field_name == schema.id_field:
            defs.append(f'"{field_name}" {pg_type} PRIMARY KEY')
        else:
            defs.append(f'"{field_name}" {pg_type}{not_null}')
    return defs


# ---------------------------------------------------------------------------
# SQL translation — PostgreSQL dialect ($1, $2, … placeholders; JSONB operators)
# ---------------------------------------------------------------------------

def _to_pg_where(
    filters: list[Filter],
    schema: CollectionSchema,
) -> tuple[str, list[Any]]:
    """Translate filters to a parameterised PostgreSQL WHERE clause.

    Dot-notation fields (``"metadata.key"``) are translated to JSONB
    text-extraction: ``"metadata"->>'key'``.  All operators are pushed to the DB.
    """
    clauses: list[str] = []
    params: list[Any] = []

    for f in filters:
        idx = len(params)

        # Build the SQL column expression.
        # Dot-notation (e.g. "metadata.status") → JSONB text extraction: "metadata"->>'status'
        # The key is embedded directly (not parameterised) because asyncpg does not support
        # parameterising JSON key names, and the key comes from the schema/LLM — not raw
        # user input that could contain SQL.
        if "." in f.field:
            col_name, key = f.field.split(".", 1)
            col_expr = f'"{col_name}"->>\'{key}\''
            is_json_subkey = True
        else:
            col_expr = f'"{f.field}"'
            is_json_subkey = False

        match f.op:
            case Op.EQ:
                clauses.append(f"{col_expr} = ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.NE:
                clauses.append(f"{col_expr} != ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.LT:
                clauses.append(f"{col_expr} < ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.GT:
                clauses.append(f"{col_expr} > ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.LTE:
                clauses.append(f"{col_expr} <= ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.GTE:
                clauses.append(f"{col_expr} >= ${idx + 1}")
                params.append(str(f.value) if is_json_subkey else f.value)
            case Op.IN:
                placeholders = ", ".join(f"${idx + i + 1}" for i in range(len(f.value)))
                clauses.append(f"{col_expr} IN ({placeholders})")
                params.extend(str(v) for v in f.value) if is_json_subkey else params.extend(f.value)
            case Op.NOT_IN:
                placeholders = ", ".join(f"${idx + i + 1}" for i in range(len(f.value)))
                clauses.append(f"{col_expr} NOT IN ({placeholders})")
                params.extend(str(v) for v in f.value) if is_json_subkey else params.extend(f.value)
            case Op.LIKE:
                clauses.append(f"{col_expr} ILIKE ${idx + 1}")
                params.append(f.value)
            case Op.IS_NULL:
                # ->> returns NULL when the key is absent or the value is JSON null.
                clauses.append(f"{col_expr} IS NULL")
            case Op.IS_NOT_NULL:
                clauses.append(f"{col_expr} IS NOT NULL")

    return (" AND ".join(clauses), params)
