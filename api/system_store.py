"""System store — persists application metadata in a configurable structured store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel

from cogbase.core.models import DocWorkflowStatus, TaskStatus
from cogbase.stores import Col, CollectionSchema, FieldSchema, FieldType, StructuredStoreBase

DOC_REGISTRY_SCHEMA = CollectionSchema(
    name="doc_registry",
    description="Document registry: one record per successfully ingested document per application.",
    primary_fields=["app_id", "doc_id"],
    fields={
        "app_id":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "status":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "ingested_at": FieldSchema(type=FieldType.STRING, nullable=False),
        "metadata":    FieldSchema(type=FieldType.STRING, nullable=True),  # JSON blob
    },
)


APP_RECORDS_SCHEMA = CollectionSchema(
    name="app_records",
    description="CogBase application registry: configuration, status, and error state per application.",
    primary_fields=["app_id"],
    fields={
        "app_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "name":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),  # client-facing handle (unique)
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


TASKS_SCHEMA = CollectionSchema(
    name="tasks",
    description="Background task records tracking ingest and workflow execution per application.",
    primary_fields=["task_id"],
    fields={
        "task_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "app_id":       FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "task_type":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "task_name":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":       FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "params_json":  FieldSchema(type=FieldType.STRING, nullable=True),
        "status":       FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "created_at":   FieldSchema(type=FieldType.STRING, nullable=False),
        "started_at":   FieldSchema(type=FieldType.STRING, nullable=True),
        "completed_at": FieldSchema(type=FieldType.STRING, nullable=True),
        "error":        FieldSchema(type=FieldType.STRING, nullable=True),
    },
)


SKILL_RECORDS_SCHEMA = CollectionSchema(
    name="skill_records",
    description="System-wide skill registry: metadata and bundle location per uploaded skill.",
    primary_fields=["skill_id"],
    fields={
        "skill_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "name":          FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "description":   FieldSchema(type=FieldType.STRING, nullable=True),
        "metadata_json": FieldSchema(type=FieldType.STRING, nullable=True),  # JSON blob
        "bundle_key":    FieldSchema(type=FieldType.STRING, nullable=False),  # document-store key
        "created_at":    FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":    FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


DOC_WORKFLOW_REGISTRY_SCHEMA = CollectionSchema(
    name="doc_workflow_registry",
    description="Workflow processing status per document per workflow. One record per (app, doc, workflow).",
    primary_fields=["app_id", "doc_id", "workflow_name"],
    fields={
        "app_id":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "workflow_name": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "status":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "updated_at":    FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


class DocRecord(BaseModel):
    app_id: str
    doc_id: str
    status: str        # "active" | "failed" | "deleted"
    ingested_at: str   # ISO-8601 UTC
    metadata: str | None = None  # JSON blob


class TaskRecord(BaseModel):
    task_id: str
    app_id: str
    task_type: str      # "ingest" | "workflow"
    task_name: str      # "ingest" for ingest tasks; workflow name for workflow tasks
    doc_id: str | None = None
    params_json: str | None = None  # JSON-serialized params
    status: TaskStatus
    created_at: str     # ISO-8601 UTC — when the task was enqueued
    started_at: str | None = None   # ISO-8601 UTC — when execution began
    completed_at: str | None = None
    error: str | None = None


class DocWorkflowRecord(BaseModel):
    app_id: str
    doc_id: str
    workflow_name: str
    status: DocWorkflowStatus
    updated_at: str  # ISO-8601 UTC


class SystemConfigOverride(BaseModel):
    key: str         # "llm" | "embedding"
    value_json: str  # JSON-serialized LLMConfig or EmbeddingConfig
    updated_at: str  # ISO-8601 UTC


class AppRecord(BaseModel):
    app_id: str       # stable internal id (primary key)
    name: str         # client-facing handle (unique, mutable)
    config_yaml: str
    status: str       # "initializing" | "active" | "error"
    error: str | None = None
    created_at: str   # ISO-8601 UTC
    updated_at: str   # ISO-8601 UTC


class SkillRecord(BaseModel):
    skill_id: str
    name: str
    description: str | None = None
    metadata_json: str | None = None  # JSON blob
    bundle_key: str                   # key of the ZIP bundle in the document store
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
        await self._store.create_collection(DOC_REGISTRY_SCHEMA)
        await self._store.create_collection(APP_RECORDS_SCHEMA)
        await self._store.create_collection(SYSTEM_CONFIG_OVERRIDES_SCHEMA)
        await self._store.create_collection(TASKS_SCHEMA)
        await self._store.create_collection(DOC_WORKFLOW_REGISTRY_SCHEMA)
        await self._store.create_collection(SKILL_RECORDS_SCHEMA)

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

    async def delete_app(self, app_id: str) -> None:
        await self._store.delete_records("app_records", filters=[Col("app_id") == app_id])
        await self._store.delete_records("doc_registry", filters=[Col("app_id") == app_id])
        await self._store.delete_records("doc_workflow_registry", filters=[Col("app_id") == app_id])
        await self._store.delete_records("tasks", filters=[Col("app_id") == app_id])

    # ------------------------------------------------------------------
    # Skill registry
    # ------------------------------------------------------------------

    async def save_skill(self, record: SkillRecord) -> None:
        await self._store.save("skill_records", [record.model_dump()])

    async def get_skill(self, skill_id: str) -> SkillRecord | None:
        rows = await self._store.query_as(
            "skill_records",
            filters=[Col("skill_id") == skill_id],
            model=SkillRecord,
        )
        return rows[0] if rows else None

    async def list_skills(self) -> list[SkillRecord]:
        return await self._store.query_as("skill_records", filters=None, model=SkillRecord)

    async def delete_skill(self, skill_id: str) -> None:
        await self._store.delete_records("skill_records", filters=[Col("skill_id") == skill_id])

    # ------------------------------------------------------------------
    # Doc registry
    # ------------------------------------------------------------------

    async def save_doc(self, record: DocRecord) -> None:
        await self._store.save("doc_registry", [record.model_dump()])

    async def get_doc(self, app_id: str, doc_id: str) -> DocRecord | None:
        rows = await self._store.query_as(
            "doc_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
            model=DocRecord,
        )
        return rows[0] if rows else None

    async def list_docs(
        self,
        app_id: str,
        *,
        status: str | None = None,
    ) -> list[DocRecord]:
        filters = [Col("app_id") == app_id]
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("doc_registry", filters=filters, model=DocRecord)

    async def delete_doc(self, app_id: str, doc_id: str) -> None:
        await self._store.delete_records(
            "doc_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
        )
        await self._store.delete_records(
            "tasks",
            filters=[
                Col("app_id") == app_id,
                Col("doc_id") == doc_id,
                Col("task_type") == "workflow",
            ],
        )
        await self._store.delete_records(
            "doc_workflow_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
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

    # ------------------------------------------------------------------
    # Doc workflow registry
    # ------------------------------------------------------------------

    async def upsert_doc_workflow_status(
        self,
        app_id: str,
        doc_id: str,
        workflow_name: str,
        status: DocWorkflowStatus,
    ) -> None:
        """Create or overwrite the workflow processing status for a document."""
        record = DocWorkflowRecord(
            app_id=app_id,
            doc_id=doc_id,
            workflow_name=workflow_name,
            status=status,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._store.save("doc_workflow_registry", [record.model_dump()])

    async def get_doc_workflow(
        self,
        app_id: str,
        doc_id: str,
        workflow_name: str,
    ) -> DocWorkflowRecord | None:
        rows = await self._store.query_as(
            "doc_workflow_registry",
            filters=[
                Col("app_id") == app_id,
                Col("doc_id") == doc_id,
                Col("workflow_name") == workflow_name,
            ],
            model=DocWorkflowRecord,
        )
        return rows[0] if rows else None

    async def list_doc_workflows(
        self,
        app_id: str,
        *,
        workflow_name: str | None = None,
        doc_id: str | None = None,
        status: DocWorkflowStatus | None = None,
    ) -> list[DocWorkflowRecord]:
        filters = [Col("app_id") == app_id]
        if workflow_name is not None:
            filters.append(Col("workflow_name") == workflow_name)
        if doc_id is not None:
            filters.append(Col("doc_id") == doc_id)
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("doc_workflow_registry", filters=filters, model=DocWorkflowRecord)

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    async def create_task(self, record: TaskRecord) -> None:
        await self._store.save("tasks", [record.model_dump()])

    async def update_task(self, task_id: str, **fields) -> None:
        task = await self.get_task(task_id)
        if task is None:
            return
        await self._store.save("tasks", [task.model_copy(update=fields).model_dump()])

    async def get_task(self, task_id: str) -> TaskRecord | None:
        rows = await self._store.query_as(
            "tasks", filters=[Col("task_id") == task_id], model=TaskRecord
        )
        return rows[0] if rows else None

    async def list_tasks(
        self,
        app_id: str,
        *,
        task_type: str | None = None,
        task_name: str | None = None,
        doc_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        filters = [Col("app_id") == app_id]
        if task_type is not None:
            filters.append(Col("task_type") == task_type)
        if task_name is not None:
            filters.append(Col("task_name") == task_name)
        if doc_id is not None:
            filters.append(Col("doc_id") == doc_id)
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("tasks", filters=filters, model=TaskRecord)

    async def create_workflow_task(
        self,
        app_id: str,
        workflow_name: str,
        doc_id: str | None,
        params_json: str | None,
    ) -> str:
        """Create a workflow task record and return its task_id."""
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.create_task(TaskRecord(
            task_id=task_id,
            app_id=app_id,
            task_type="workflow",
            task_name=workflow_name,
            doc_id=doc_id,
            params_json=params_json,
            status=TaskStatus.PENDING,
            created_at=now,
        ))
        return task_id

    async def complete_workflow_task(
        self,
        task_id: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Mark a workflow task as done or failed."""
        await self.update_task(
            task_id,
            status=TaskStatus.DONE if success else TaskStatus.FAILED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )
