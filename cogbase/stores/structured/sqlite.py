"""SQLite implementation of StructuredStoreBase."""

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.filters import Filter, matches, to_sql_where
from cogbase.stores.schema import CollectionSchema, FieldType

_SQL_TYPE: dict[FieldType, str] = {
    FieldType.STRING:  "TEXT",
    FieldType.INTEGER: "INTEGER",
    FieldType.FLOAT:   "REAL",
    FieldType.BOOLEAN: "INTEGER",  # SQLite has no native bool
    FieldType.JSON:    "TEXT",
}


class SQLiteStructuredStore(StructuredStoreBase):
    """SQLite-backed structured store.

    Primitive-column filters (STRING, INTEGER, FLOAT, BOOLEAN) are pushed to
    SQL WHERE clauses.  JSON-column filters are post-filtered in Python, as
    SQLite does not natively support the full filter DSL over JSON columns.

    Args:
        path: Path to the SQLite file, or ``":memory:"`` for an in-process db.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._schemas: dict[str, CollectionSchema] = {}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteStructuredStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # StructuredStoreBase
    # ------------------------------------------------------------------

    async def create_collection(self, schema: CollectionSchema) -> None:
        # 1. Create the table if it does not exist yet.
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{schema.name}" ({", ".join(_col_defs(schema))})'
        )

        # 2. Add any columns present in the desired schema but missing from the
        #    table (handles fields added after the initial create).  Columns
        #    removed from the schema are left in the table but silently ignored
        #    by the row helpers — no data is lost, and DROP COLUMN is not needed.
        existing_cols = {
            row[1]
            for row in self._conn.execute(f'PRAGMA table_info("{schema.name}")')
        }
        for field_name, field in schema.fields.items():
            if field_name not in existing_cols:
                sql_type = _SQL_TYPE[field.type]
                self._conn.execute(
                    f'ALTER TABLE "{schema.name}" ADD COLUMN "{field_name}" {sql_type}'
                )

        # 3. Ensure indexes exist (idempotent).
        for field_name, field in schema.fields.items():
            if field_name in schema.primary_fields:
                continue
            if field.unique:
                self._conn.execute(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS "uq_{schema.name}_{field_name}" '
                    f'ON "{schema.name}" ("{field_name}")'
                )
            elif field.index:
                self._conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "idx_{schema.name}_{field_name}" '
                    f'ON "{schema.name}" ("{field_name}")'
                )

        self._conn.commit()
        self._schemas[schema.name] = schema

    async def save(self, collection: str, records: list[BaseModel]) -> None:
        schema = self._get_schema(collection)
        cols = list(schema.fields.keys())
        col_list = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["?"] * len(cols))
        update_cols = [c for c in cols if c not in schema.primary_fields]
        conflict_target = ", ".join(f'"{f}"' for f in schema.primary_fields)
        if update_cols:
            update_set = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
            conflict_clause = f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}"
        else:
            conflict_clause = f"ON CONFLICT ({conflict_target}) DO NOTHING"
        sql = f'INSERT INTO "{collection}" ({col_list}) VALUES ({placeholders}) {conflict_clause}'
        self._conn.executemany(sql, [_to_sql_row(r, schema) for r in records])
        self._conn.commit()

    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        schema = self._get_schema(collection)
        fs = filters or []
        json_fields = _json_fields(schema)

        where, params = to_sql_where(fs, json_fields)

        # Determine which columns to SELECT.  When a projection is requested we
        # still need to fetch any JSON columns that will be post-filtered in
        # Python; those extra columns are stripped from the final results.
        if fields:
            projection = {c for c in fields if c in schema.fields}
            # JSON columns needed for Python post-filtering but not in the projection.
            py_filter_fields = {f.field.split(".")[0] for f in fs if f.field.split(".")[0] in json_fields}
            extra_fetch = py_filter_fields - projection
            fetch_cols = sorted(projection | extra_fetch)  # sorted for determinism
        else:
            fetch_cols = list(schema.fields.keys())
            extra_fetch = set()

        col_list = ", ".join(f'"{c}"' for c in fetch_cols)
        sql = f'SELECT {col_list} FROM "{collection}"'
        if where:
            sql += f" WHERE {where}"

        rows = [_from_sql_row(dict(r), schema) for r in self._conn.execute(sql, params).fetchall()]

        # Post-filter JSON-column conditions in Python.
        py_filters = [f for f in fs if f.field.split(".")[0] in json_fields]
        if py_filters:
            rows = [r for r in rows if matches(r, py_filters)]

        # Strip columns that were fetched only for post-filtering.
        if extra_fetch:
            rows = [{k: v for k, v in row.items() if k not in extra_fetch} for row in rows]

        return rows

    async def update_collection(self, schema: CollectionSchema) -> None:
        old_schema = self._get_schema(schema.name)
        if schema.primary_fields != old_schema.primary_fields:
            raise ValueError(
                f"Cannot change primary_fields from {old_schema.primary_fields!r} "
                f"to {schema.primary_fields!r} — "
                "update_collection does not support primary-key migration"
            )

        old_fields = set(old_schema.fields)
        new_fields = set(schema.fields)
        added = new_fields - old_fields
        removed = old_fields - new_fields

        if removed:
            # Table rebuild: create a temp table with the new schema, copy the
            # surviving columns, swap the tables.  This is the only portable way
            # to drop columns in SQLite (DROP COLUMN requires ≥ 3.35).
            tmp = f"_{schema.name}_upd_tmp"
            col_defs = _col_defs(schema)
            carry = sorted(old_fields & new_fields)  # preserve column order determinism
            carry_sql = ", ".join(f'"{c}"' for c in carry)
            with self._conn:
                self._conn.execute(f'CREATE TABLE "{tmp}" ({", ".join(col_defs)})')
                self._conn.execute(
                    f'INSERT INTO "{tmp}" ({carry_sql}) '
                    f'SELECT {carry_sql} FROM "{schema.name}"'
                )
                self._conn.execute(f'DROP TABLE "{schema.name}"')
                self._conn.execute(f'ALTER TABLE "{tmp}" RENAME TO "{schema.name}"')
        elif added:
            # Additions only — cheaper path: just ALTER TABLE ADD COLUMN.
            for field_name in added:
                field = schema.fields[field_name]
                sql_type = _SQL_TYPE[field.type]
                self._conn.execute(
                    f'ALTER TABLE "{schema.name}" ADD COLUMN "{field_name}" {sql_type}'
                )

        # Ensure indexes exist for all indexed fields (idempotent).
        for field_name, field in schema.fields.items():
            if field_name in schema.primary_fields:
                continue
            if field.unique:
                self._conn.execute(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS "uq_{schema.name}_{field_name}" '
                    f'ON "{schema.name}" ("{field_name}")'
                )
            elif field.index:
                self._conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "idx_{schema.name}_{field_name}" '
                    f'ON "{schema.name}" ("{field_name}")'
                )

        self._conn.commit()
        self._schemas[schema.name] = schema

    async def delete_collection(self, collection: str) -> None:
        self._get_schema(collection)  # raises KeyError if unknown
        self._conn.execute(f'DROP TABLE "{collection}"')
        self._conn.commit()
        del self._schemas[collection]

    async def list_collections(self) -> list[str]:
        return list(self._schemas.keys())

    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        schema = self._get_schema(collection)
        fs = filters or []

        if not fs:
            self._conn.execute(f'DELETE FROM "{collection}"')
            self._conn.commit()
            return

        # Reuse query() so JSON-column filters work correctly
        matching = await self.query(collection, fs)
        if not matching:
            return

        key_clauses: list[str] = []
        params: list[object] = []
        for row in matching:
            key_clauses.append(
                "(" + " AND ".join(f'"{field}" = ?' for field in schema.primary_fields) + ")"
            )
            params.extend(row[field] for field in schema.primary_fields)
        self._conn.execute(
            f'DELETE FROM "{collection}" WHERE ' + " OR ".join(key_clauses),
            params,
        )
        self._conn.commit()

    # ------------------------------------------------------------------

    def _get_schema(self, collection: str) -> CollectionSchema:
        if collection not in self._schemas:
            raise KeyError(f"Collection '{collection}' not found. Call create_collection first.")
        return self._schemas[collection]


# ------------------------------------------------------------------
# Row helpers
# ------------------------------------------------------------------

def _to_sql_row(record: BaseModel, schema: CollectionSchema) -> tuple:
    raw = record.model_dump(mode="json")
    row = []
    for field_name, field in schema.fields.items():
        val = raw.get(field_name)
        if val is None:
            row.append(None)
        elif field.type == FieldType.JSON:
            row.append(json.dumps(val))
        elif field.type == FieldType.BOOLEAN:
            row.append(int(val))
        else:
            row.append(val)
    return tuple(row)


def _from_sql_row(row: dict, schema: CollectionSchema) -> dict:
    """Deserialise a fetched SQL row dict into plain Python types.

    Only the fields present in *row* are included in the result — callers that
    SELECT a subset of columns will get back only those columns.
    """
    result: dict = {}
    for field_name, val in row.items():
        field = schema.fields.get(field_name)
        if field is None:
            continue  # column not in schema (e.g. old column not yet migrated away)
        if val is None:
            result[field_name] = None
        elif field.type == FieldType.JSON:
            result[field_name] = json.loads(val)
        elif field.type == FieldType.BOOLEAN:
            result[field_name] = bool(val)
        else:
            result[field_name] = val
    return result


def _json_fields(schema: CollectionSchema) -> set[str]:
    return {name for name, f in schema.fields.items() if f.type == FieldType.JSON}


def _col_defs(schema: CollectionSchema) -> list[str]:
    """Return a list of SQL column definition strings for *schema*."""
    defs: list[str] = []
    for field_name, field in schema.fields.items():
        sql_type = _SQL_TYPE[field.type]
        is_primary = field_name in schema.primary_fields
        not_null = " NOT NULL" if (is_primary or not field.nullable) else ""
        defs.append(f'"{field_name}" {sql_type}{not_null}')
    pk_cols = ", ".join(f'"{field}"' for field in schema.primary_fields)
    defs.append(f"PRIMARY KEY ({pk_cols})")
    return defs
