"""System store — persists application metadata in a configurable structured store."""

from __future__ import annotations

from pydantic import BaseModel

from cogbase.stores import Col, CollectionSchema, FieldSchema, FieldType, StructuredStoreBase

APP_RECORDS_SCHEMA = CollectionSchema(
    name="app_records",
    description="CogBase application registry: configuration, status, and error state per named application.",
    primary_fields=["name"],
    fields={
        "name":        FieldSchema(type=FieldType.STRING, nullable=False),
        "config_yaml": FieldSchema(type=FieldType.STRING, nullable=False),
        "status":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "error":       FieldSchema(type=FieldType.STRING, nullable=True),
        "created_at":  FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":  FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


class AppRecord(BaseModel):
    name: str
    config_yaml: str
    status: str       # "initializing" | "active" | "error"
    error: str | None = None
    created_at: str   # ISO-8601 UTC
    updated_at: str   # ISO-8601 UTC


class SystemStore:
    """Thin persistence layer for application metadata.

    Accepts any ``StructuredStoreBase`` backend — SQLite, Postgres, or
    in-memory — configured via ``system_db`` in ``cogbase_system.yaml``.

    Args:
        store: A ready-to-use structured store instance.
    """

    def __init__(self, store: StructuredStoreBase) -> None:
        self._store = store

    async def setup(self) -> None:
        """Create the app_records collection if it does not exist. Idempotent."""
        await self._store.create_collection(APP_RECORDS_SCHEMA)

    async def save_app(self, record: AppRecord) -> None:
        await self._store.save("app_records", [record])

    async def get_app(self, name: str) -> AppRecord | None:
        rows = await self._store.query_as(
            "app_records",
            filters=[Col("name") == name],
            model=AppRecord,
        )
        return rows[0] if rows else None

    async def list_apps(self) -> list[AppRecord]:
        return await self._store.query_as("app_records", filters=None, model=AppRecord)

    async def delete_app(self, name: str) -> None:
        await self._store.delete_records(
            "app_records",
            filters=[Col("name") == name],
        )
