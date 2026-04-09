"""SQLite implementation of StructuredStoreBase."""

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel

from cogbase.stores.base import StructuredStoreBase
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
        if schema.name in self._schemas:
            return

        col_defs: list[str] = []
        for field_name, field in schema.fields.items():
            sql_type = _SQL_TYPE[field.type]
            not_null = "" if field.nullable else " NOT NULL"
            if field_name == schema.id_field:
                col_defs.append(f'"{field_name}" {sql_type} PRIMARY KEY')
            else:
                col_defs.append(f'"{field_name}" {sql_type}{not_null}')

        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{schema.name}" ({", ".join(col_defs)})'
        )
        for field_name, field in schema.fields.items():
            if field.index and field_name != schema.id_field:
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
        sql = f'INSERT OR REPLACE INTO "{collection}" ({col_list}) VALUES ({placeholders})'
        self._conn.executemany(sql, [_to_sql_row(r, schema) for r in records])
        self._conn.commit()

    async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]:
        schema = self._get_schema(collection)
        fs = filters or []
        json_fields = _json_fields(schema)

        where, params = to_sql_where(fs, json_fields)
        sql = f'SELECT * FROM "{collection}"'
        if where:
            sql += f" WHERE {where}"

        rows = [_from_sql_row(dict(r), schema) for r in self._conn.execute(sql, params).fetchall()]

        # Post-filter JSON-column conditions in Python
        py_filters = [f for f in fs if f.field in json_fields]
        if py_filters:
            rows = [r for r in rows if matches(r, py_filters)]

        return rows

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

        id_field = schema.id_field
        ids = [r[id_field] for r in matching]
        placeholders = ", ".join(["?"] * len(ids))
        self._conn.execute(
            f'DELETE FROM "{collection}" WHERE "{id_field}" IN ({placeholders})', ids
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
    result: dict = {}
    for field_name, field in schema.fields.items():
        val = row.get(field_name)
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
