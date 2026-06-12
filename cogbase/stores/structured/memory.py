"""In-memory implementation of StructuredStoreBase — backed by pandas DataFrames."""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.filters import Filter, Op, _like
from cogbase.stores.schema import CollectionSchema, FieldType
from cogbase.stores.scope import AppScope

_PANDAS_DTYPE: dict[FieldType, str] = {
    FieldType.STRING:  "object",
    FieldType.INTEGER: "Int64",    # nullable integer
    FieldType.FLOAT:   "float64",
    FieldType.BOOLEAN: "boolean",  # nullable boolean
    FieldType.JSON:    "object",
}


class InMemoryStructuredStore(StructuredStoreBase):
    """In-memory store backed by a pandas DataFrame per collection.

    All filtering is translated to pandas boolean masks.  JSON-column filters
    fall back to per-row Python evaluation via the existing ``_like`` helper
    since pandas has no native understanding of the filter DSL over object columns.
    """

    def __init__(self, scope: AppScope | None = None) -> None:
        super().__init__(scope)
        self._frames: dict[str, pd.DataFrame] = {}

    async def create_collection(self, schema: CollectionSchema) -> None:
        key = self._c(schema.name)
        if schema.name not in self._schemas:
            self._schemas[schema.name] = schema
            self._frames[key] = _empty_frame(schema)
            return

        # Collection already exists — add columns for any fields that are new
        # in the desired schema (handles fields added between restarts or within
        # the same session).  Removed fields are silently ignored: the DataFrame
        # retains the old column but the row helpers only read/write schema fields.
        old_fields = self._schemas[schema.name].fields
        df = self._frames[key]
        for field_name, field in schema.fields.items():
            if field_name not in old_fields:
                df[field_name] = pd.Series(dtype=_PANDAS_DTYPE[field.type])
        self._frames[key] = df
        self._schemas[schema.name] = schema

    async def update_collection(self, schema: CollectionSchema) -> None:
        old_schema = self._get_schema(schema.name)
        if schema.primary_fields != old_schema.primary_fields:
            raise ValueError(
                f"Cannot change primary_fields from {old_schema.primary_fields!r} "
                f"to {schema.primary_fields!r} — "
                "update_collection does not support primary-key migration"
            )

        key = self._c(schema.name)
        old_fields = set(old_schema.fields)
        new_fields = set(schema.fields)
        df = self._frames[key]

        # Add columns for new fields.
        for field_name in new_fields - old_fields:
            df[field_name] = pd.Series(dtype=_PANDAS_DTYPE[schema.fields[field_name].type])

        # Drop columns for removed fields.
        removed = old_fields - new_fields
        if removed:
            df = df.drop(columns=[c for c in removed if c in df.columns])

        self._frames[key] = df
        self._schemas[schema.name] = schema

    async def _save(self, collection: str, records: list[dict]) -> None:
        schema = self._get_schema(collection)
        rows = [_serialize(r, schema) for r in records]
        new_df = _to_frame(rows, schema)
        key = self._c(collection)
        df = self._frames[key]
        # Compute surviving rows: drop those whose PK appears in the incoming batch.
        if not df.empty and not new_df.empty:
            existing_keys = pd.Series(list(df[schema.primary_fields].itertuples(index=False, name=None)))
            incoming_keys = set(new_df[schema.primary_fields].itertuples(index=False, name=None))
            surviving = df[~existing_keys.isin(incoming_keys).to_numpy()]
        else:
            surviving = df
        # Enforce unique constraints against surviving rows.
        for field_name, field in schema.fields.items():
            if not field.unique or field_name in schema.primary_fields:
                continue
            if surviving.empty:
                continue
            incoming_vals = new_df[field_name].dropna()
            if incoming_vals.empty:
                continue
            conflicts = surviving[field_name].isin(incoming_vals)
            if conflicts.any():
                conflict_val = surviving.loc[conflicts, field_name].iloc[0]
                raise ValueError(
                    f"Unique constraint violation on '{collection}.{field_name}': "
                    f"value {conflict_val!r} already exists"
                )
        self._frames[key] = pd.concat([surviving, new_df], ignore_index=True)

    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        schema = self._get_schema(collection)
        df = self._frames[self._c(collection)]
        if filters:
            mask = _build_mask(df, filters)
            df = df[mask]
        # Determine which columns to return.  Start from schema fields (drops
        # any columns removed from the schema), then narrow to the requested
        # projection when one is provided.
        schema_cols = [c for c in schema.fields if c in df.columns]
        if fields:
            schema_cols = [c for c in schema_cols if c in fields]
        return _to_records(df[schema_cols])

    async def delete_collection(self, collection: str) -> None:
        self._schemas.pop(collection, None)
        self._frames.pop(self._c(collection), None)

    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        schema = self._get_schema(collection)
        key = self._c(collection)
        if not filters:
            self._frames[key] = _empty_frame(schema)
            return
        df = self._frames[key]
        mask = _build_mask(df, filters)
        self._frames[key] = df[~mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def persist(self, path: str | Path) -> None:
        """Persist all collections to *path*.

        *path* is a directory that will be created if it does not exist.  Two
        kinds of files are written inside it:

        * ``_schemas.json``       — all ``CollectionSchema`` definitions as JSON
        * ``{collection}.pkl``    — one pickle file per collection (DataFrame)
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._persist_sync, Path(path))

    def _persist_sync(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        schemas_data = {name: schema.model_dump() for name, schema in self._schemas.items()}
        (path / "_schemas.json").write_text(json.dumps(schemas_data), encoding="utf-8")
        for name, df in self._frames.items():
            with open(path / f"{name}.pkl", "wb") as fh:
                pickle.dump(df, fh)

    async def load(self, path: str | Path) -> None:
        """Load a previously persisted store from *path* (a directory written by ``persist``).

        Replaces the current in-memory state entirely.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync, Path(path))

    def _load_sync(self, path: Path) -> None:
        schemas_data: dict = json.loads((path / "_schemas.json").read_text(encoding="utf-8"))
        self._schemas = {name: CollectionSchema(**data) for name, data in schemas_data.items()}
        self._frames = {}
        for name, schema in self._schemas.items():
            key = self._c(name)
            pkl_path = path / f"{key}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as fh:
                    self._frames[key] = pickle.load(fh)
            else:
                self._frames[key] = _empty_frame(schema)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_frame(schema: CollectionSchema) -> pd.DataFrame:
    return pd.DataFrame({
        name: pd.Series(dtype=_PANDAS_DTYPE[field.type])
        for name, field in schema.fields.items()
    })


def _to_frame(rows: list[dict], schema: CollectionSchema) -> pd.DataFrame:
    if not rows:
        return _empty_frame(schema)
    df = pd.DataFrame(rows)
    for name, field in schema.fields.items():
        if name in df.columns:
            df[name] = df[name].astype(_PANDAS_DTYPE[field.type])
    return df


def _extract_col(df: pd.DataFrame, field: str) -> pd.Series:
    """Return a Series for *field*, supporting dotted paths into JSON object columns.

    ``"payload.count"`` → ``df["payload"].apply(lambda v: v.get("count"))``

    Nested access stops early and returns ``None`` if any intermediate value is
    not a dict (e.g. the path leads into a list or a scalar).
    """
    if "." not in field:
        return df[field]
    parts = field.split(".")
    base, path = parts[0], parts[1:]
    return df[base].apply(lambda v: _nested_get(v, path))


def _nested_get(v: Any, path: list[str]) -> Any:
    for key in path:
        if not isinstance(v, dict):
            return None
        v = v.get(key)
    return v


def _build_mask(df: pd.DataFrame, filters: list[Filter]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for f in filters:
        col = _extract_col(df, f.field)
        match f.op:
            case Op.EQ:
                mask &= (col == f.value).fillna(False)
            case Op.NE:
                mask &= (col != f.value).fillna(False)
            case Op.LT:
                mask &= col.notna() & (col < f.value)
            case Op.GT:
                mask &= col.notna() & (col > f.value)
            case Op.LTE:
                mask &= col.notna() & (col <= f.value)
            case Op.GTE:
                mask &= col.notna() & (col >= f.value)
            case Op.IN:
                mask &= col.isin(f.value)
            case Op.NOT_IN:
                mask &= ~col.isin(f.value)
            case Op.OVERLAPS:
                wanted = set(f.value)
                mask &= col.apply(
                    lambda v: isinstance(v, (list, tuple, set)) and bool(wanted & set(v))
                )
            case Op.LIKE:
                mask &= col.apply(lambda v: _like(v, f.value))
            case Op.IS_NULL:
                mask &= col.isna()
            case Op.IS_NOT_NULL:
                mask &= col.notna()
    return mask


def _to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: _clean(v) for k, v in row.items()}
        for row in df.to_dict("records")
    ]


def _clean(v: Any) -> Any:
    """Convert pandas/numpy scalars and NA sentinels to plain Python types."""
    if v is pd.NA:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _serialize(record: dict, schema: CollectionSchema) -> dict:
    raw = record
    row: dict = {}
    for name, field in schema.fields.items():
        val = raw.get(name)
        if val is None:
            row[name] = None
            continue
        if field.type == FieldType.BOOLEAN:
            row[name] = bool(val)
        elif field.type == FieldType.INTEGER:
            row[name] = int(val)
        elif field.type == FieldType.FLOAT:
            row[name] = float(val)
        else:
            row[name] = val  # STRING and JSON stored as-is
    return row
