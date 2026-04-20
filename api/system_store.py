"""System store — persists application metadata using SQLiteStructuredStore."""

from __future__ import annotations

from pydantic import BaseModel

from cogbase.stores.filters import Col
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured.sqlite import SQLiteStructuredStore

APP_RECORDS_SCHEMA = CollectionSchema(
    name="app_records",
    primary_fields=["app_id"],
    fields={
        "app_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "name":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "config_yaml": FieldSchema(type=FieldType.STRING, nullable=False),
        "status":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "error":       FieldSchema(type=FieldType.STRING, nullable=True),
        "created_at":  FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":  FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


class AppRecord(BaseModel):
    app_id: str
    name: str
    config_yaml: str
    status: str       # "initializing" | "active" | "error"
    error: str | None = None
    created_at: str   # ISO-8601 UTC
    updated_at: str   # ISO-8601 UTC


class SystemStore:
    """Thin persistence layer over a dedicated SQLite database.

    Args:
        db_path: Path to the SQLite file.  Defaults to ``./cogbase_system.db``.
    """

    def __init__(self, db_path: str = "./cogbase_system.db") -> None:
        self._store = SQLiteStructuredStore(db_path)

    async def setup(self) -> None:
        """Create the app_records table if it does not exist. Idempotent."""
        await self._store.create_collection(APP_RECORDS_SCHEMA)

    async def save_app(self, record: AppRecord) -> None:
        await self._store.save("app_records", [record])

    async def get_app(self, app_id: str) -> AppRecord | None:
        rows = await self._store.query_as(
            "app_records",
            filters=[Col("app_id") == app_id],
            model=AppRecord,
        )
        return rows[0] if rows else None

    async def get_app_by_name(self, name: str) -> AppRecord | None:
        rows = await self._store.query_as(
            "app_records",
            filters=[Col("name") == name],
            model=AppRecord,
        )
        return rows[0] if rows else None

    async def list_apps(self) -> list[AppRecord]:
        return await self._store.query_as("app_records", filters=None, model=AppRecord)

    async def delete_app(self, app_id: str) -> None:
        await self._store.delete_records(
            "app_records",
            filters=[Col("app_id") == app_id],
        )

    def close(self) -> None:
        self._store.close()
