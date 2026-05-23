"""System store — persists application metadata in a configurable structured store."""

from __future__ import annotations

from datetime import datetime, timezone

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


SYSTEM_CONFIG_OVERRIDES_SCHEMA = CollectionSchema(
    name="system_config_overrides",
    description="Runtime overrides for system-level LLM and embedding configuration, set via PATCH /system/config.",
    primary_fields=["key"],
    fields={
        "key":        FieldSchema(type=FieldType.STRING, nullable=False),
        "value_json": FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at": FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


class SystemConfigOverride(BaseModel):
    key: str         # "llm" | "embedding"
    value_json: str  # JSON-serialized LLMConfig or EmbeddingConfig
    updated_at: str  # ISO-8601 UTC


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
        """Create managed collections if they do not exist. Idempotent."""
        await self._store.create_collection(APP_RECORDS_SCHEMA)
        await self._store.create_collection(SYSTEM_CONFIG_OVERRIDES_SCHEMA)

    async def save_app(self, record: AppRecord) -> None:
        await self._store.save("app_records", [record.model_dump()])

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

    async def save_system_config_override(self, key: str, value_json: str) -> None:
        record = SystemConfigOverride(
            key=key,
            value_json=value_json,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._store.save("system_config_overrides", [record.model_dump()])

    async def load_system_config_overrides(self) -> dict[str, str]:
        rows = await self._store.query_as(
            "system_config_overrides",
            filters=None,
            model=SystemConfigOverride,
        )
        return {r.key: r.value_json for r in rows}
