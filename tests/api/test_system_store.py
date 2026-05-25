"""Unit tests for api/system_store.py — SystemStore and AppRecord."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cogbase.stores.structured.memory import InMemoryStructuredStore
from api.system_store import AppRecord, DocRecord, DocWorkflowRecord, SystemStore, TaskRecord


def _make_record(name: str = "my-app", status: str = "active") -> AppRecord:
    return AppRecord(
        name=name,
        config_yaml="name: my-app\nllm:\n  model: gpt-4o-mini\n",
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest_asyncio.fixture
async def store() -> SystemStore:
    backend = InMemoryStructuredStore()
    ss = SystemStore(store=backend)
    await ss.setup()
    return ss


class TestSystemStoreSetup:
    @pytest.mark.asyncio
    async def test_setup_idempotent(self):
        backend = InMemoryStructuredStore()
        ss = SystemStore(store=backend)
        await ss.setup()
        await ss.setup()  # second call must not raise
        assert await ss.list_apps() == []


class TestSystemStoreSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_get_app(self, store):
        record = _make_record()
        await store.save_app(record)
        fetched = await store.get_app("my-app")
        assert fetched is not None
        assert fetched.name == "my-app"
        assert fetched.status == "active"

    @pytest.mark.asyncio
    async def test_get_app_returns_none_for_unknown(self, store):
        result = await store.get_app("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_upserts_record(self, store):
        record = _make_record()
        await store.save_app(record)
        updated = record.model_copy(update={"status": "error", "error": "something failed"})
        await store.save_app(updated)
        fetched = await store.get_app("my-app")
        assert fetched.status == "error"
        assert fetched.error == "something failed"


class TestSystemStoreListApps:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        assert await store.list_apps() == []

    @pytest.mark.asyncio
    async def test_list_returns_all(self, store):
        await store.save_app(_make_record(name="a"))
        await store.save_app(_make_record(name="b"))
        apps = await store.list_apps()
        assert len(apps) == 2
        names = {r.name for r in apps}
        assert names == {"a", "b"}


class TestSystemStoreDeleteApp:
    @pytest.mark.asyncio
    async def test_delete_removes_record(self, store):
        await store.save_app(_make_record(name="my-app"))
        await store.delete_app("my-app")
        assert await store.get_app("my-app") is None

    @pytest.mark.asyncio
    async def test_delete_only_removes_target(self, store):
        await store.save_app(_make_record(name="a"))
        await store.save_app(_make_record(name="b"))
        await store.delete_app("a")
        assert await store.get_app("a") is None
        assert await store.get_app("b") is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, store):
        # Must not raise
        await store.delete_app("ghost")


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "t-001",
    app_name: str = "my-app",
    task_type: str = "ingest",
    task_name: str = "ingest",
    doc_id: str | None = "doc-1",
    status: str = "pending",
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        app_name=app_name,
        task_type=task_type,
        task_name=task_name,
        doc_id=doc_id,
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
    )


class TestCreateAndGetTask:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store):
        task = _make_task()
        await store.create_task(task)
        fetched = await store.get_task("t-001")
        assert fetched is not None
        assert fetched.task_id == "t-001"
        assert fetched.app_name == "my-app"
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, store):
        assert await store.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_create_preserves_optional_fields(self, store):
        task = _make_task().model_copy(update={
            "params_json": '{"issue": "breach"}',
            "completed_at": "2026-01-01T01:00:00+00:00",
            "error": "oops",
        })
        await store.create_task(task)
        fetched = await store.get_task("t-001")
        assert fetched.params_json == '{"issue": "breach"}'
        assert fetched.completed_at == "2026-01-01T01:00:00+00:00"
        assert fetched.error == "oops"


class TestUpdateTask:
    @pytest.mark.asyncio
    async def test_update_status(self, store):
        await store.create_task(_make_task())
        await store.update_task("t-001", status="running")
        fetched = await store.get_task("t-001")
        assert fetched.status == "running"

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, store):
        await store.create_task(_make_task())
        await store.update_task("t-001", status="done", completed_at="2026-01-01T02:00:00+00:00")
        fetched = await store.get_task("t-001")
        assert fetched.status == "done"
        assert fetched.completed_at == "2026-01-01T02:00:00+00:00"

    @pytest.mark.asyncio
    async def test_update_preserves_unchanged_fields(self, store):
        await store.create_task(_make_task(doc_id="doc-42"))
        await store.update_task("t-001", status="running")
        fetched = await store.get_task("t-001")
        assert fetched.doc_id == "doc-42"
        assert fetched.app_name == "my-app"

    @pytest.mark.asyncio
    async def test_update_nonexistent_is_noop(self, store):
        # Must not raise
        await store.update_task("ghost", status="done")


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        assert await store.list_tasks("my-app") == []

    @pytest.mark.asyncio
    async def test_list_returns_all_for_app(self, store):
        await store.create_task(_make_task(task_id="t-1", doc_id="d1"))
        await store.create_task(_make_task(task_id="t-2", doc_id="d2"))
        tasks = await store.list_tasks("my-app")
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_isolated_by_app(self, store):
        await store.create_task(_make_task(task_id="t-1", app_name="app-a"))
        await store.create_task(_make_task(task_id="t-2", app_name="app-b"))
        assert len(await store.list_tasks("app-a")) == 1
        assert len(await store.list_tasks("app-b")) == 1
        assert len(await store.list_tasks("app-c")) == 0

    @pytest.mark.asyncio
    async def test_filter_by_task_type(self, store):
        await store.create_task(_make_task(task_id="t-1", task_type="ingest"))
        await store.create_task(_make_task(task_id="t-2", task_type="workflow", task_name="analyze"))
        ingest = await store.list_tasks("my-app", task_type="ingest")
        workflow = await store.list_tasks("my-app", task_type="workflow")
        assert len(ingest) == 1 and ingest[0].task_type == "ingest"
        assert len(workflow) == 1 and workflow[0].task_type == "workflow"

    @pytest.mark.asyncio
    async def test_filter_by_task_name(self, store):
        await store.create_task(_make_task(task_id="t-1", task_type="workflow", task_name="analyze"))
        await store.create_task(_make_task(task_id="t-2", task_type="workflow", task_name="summarize"))
        results = await store.list_tasks("my-app", task_name="analyze")
        assert len(results) == 1 and results[0].task_name == "analyze"

    @pytest.mark.asyncio
    async def test_filter_by_doc_id(self, store):
        await store.create_task(_make_task(task_id="t-1", doc_id="doc-a"))
        await store.create_task(_make_task(task_id="t-2", doc_id="doc-b"))
        results = await store.list_tasks("my-app", doc_id="doc-a")
        assert len(results) == 1 and results[0].doc_id == "doc-a"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, store):
        await store.create_task(_make_task(task_id="t-1", status="pending"))
        await store.create_task(_make_task(task_id="t-2", status="done"))
        done = await store.list_tasks("my-app", status="done")
        assert len(done) == 1 and done[0].status == "done"

    @pytest.mark.asyncio
    async def test_filter_combined(self, store):
        await store.create_task(_make_task(task_id="t-1", task_type="workflow", task_name="analyze", doc_id="doc-1", status="done"))
        await store.create_task(_make_task(task_id="t-2", task_type="workflow", task_name="analyze", doc_id="doc-2", status="done"))
        await store.create_task(_make_task(task_id="t-3", task_type="workflow", task_name="analyze", doc_id="doc-1", status="failed"))
        results = await store.list_tasks("my-app", task_type="workflow", task_name="analyze", doc_id="doc-1", status="done")
        assert len(results) == 1 and results[0].task_id == "t-1"


class TestCreateWorkflowTask:
    @pytest.mark.asyncio
    async def test_returns_unique_task_id(self, store):
        id1 = await store.create_workflow_task("app", "wf-a", "doc-1", None)
        id2 = await store.create_workflow_task("app", "wf-a", "doc-1", None)
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_persists_record(self, store):
        task_id = await store.create_workflow_task("my-app", "analyze", "doc-42", '{"issue": "x"}')
        task = await store.get_task(task_id)
        assert task is not None
        assert task.app_name == "my-app"
        assert task.task_type == "workflow"
        assert task.task_name == "analyze"
        assert task.doc_id == "doc-42"
        assert task.params_json == '{"issue": "x"}'
        assert task.status == "pending"
        assert task.completed_at is None

    @pytest.mark.asyncio
    async def test_doc_id_may_be_none(self, store):
        task_id = await store.create_workflow_task("my-app", "wf", None, None)
        task = await store.get_task(task_id)
        assert task.doc_id is None


class TestCompleteWorkflowTask:
    @pytest.mark.asyncio
    async def test_marks_done_on_success(self, store):
        task_id = await store.create_workflow_task("my-app", "wf", "doc-1", None)
        await store.complete_workflow_task(task_id, success=True)
        task = await store.get_task(task_id)
        assert task.status == "done"
        assert task.completed_at is not None
        assert task.error is None

    @pytest.mark.asyncio
    async def test_marks_failed_on_failure(self, store):
        task_id = await store.create_workflow_task("my-app", "wf", "doc-1", None)
        await store.complete_workflow_task(task_id, success=False, error="LLM timeout")
        task = await store.get_task(task_id)
        assert task.status == "failed"
        assert task.completed_at is not None
        assert task.error == "LLM timeout"

    @pytest.mark.asyncio
    async def test_completed_at_is_set(self, store):
        task_id = await store.create_workflow_task("my-app", "wf", "doc-1", None)
        await store.complete_workflow_task(task_id, success=True)
        task = await store.get_task(task_id)
        # ISO-8601 UTC string
        assert "T" in task.completed_at and task.completed_at.endswith("+00:00")


# ---------------------------------------------------------------------------
# DocWorkflowRecord
# ---------------------------------------------------------------------------


class TestUpsertAndGetDocWorkflow:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "pending")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert record is not None
        assert record.app_name == "my-app"
        assert record.doc_id == "doc-1"
        assert record.workflow_name == "analyze"
        assert record.status == "pending"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, store):
        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_status(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "pending")
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert record.status == "done"

    @pytest.mark.asyncio
    async def test_upsert_sets_updated_at(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "running")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert "T" in record.updated_at and record.updated_at.endswith("+00:00")

    @pytest.mark.asyncio
    async def test_distinct_workflows_for_same_doc(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-1", "summarize", "pending")
        r_analyze = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        r_summarize = await store.get_doc_workflow("my-app", "doc-1", "summarize")
        assert r_analyze.status == "done"
        assert r_summarize.status == "pending"

    @pytest.mark.asyncio
    async def test_same_workflow_distinct_docs(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-2", "analyze", "failed")
        r1 = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        r2 = await store.get_doc_workflow("my-app", "doc-2", "analyze")
        assert r1.status == "done"
        assert r2.status == "failed"


class TestListDocWorkflows:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        assert await store.list_doc_workflows("my-app") == []

    @pytest.mark.asyncio
    async def test_list_returns_all_for_app(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-2", "analyze", "pending")
        results = await store.list_doc_workflows("my-app")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_isolated_by_app(self, store):
        await store.upsert_doc_workflow_status("app-a", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("app-b", "doc-1", "analyze", "done")
        assert len(await store.list_doc_workflows("app-a")) == 1
        assert len(await store.list_doc_workflows("app-b")) == 1
        assert len(await store.list_doc_workflows("app-c")) == 0

    @pytest.mark.asyncio
    async def test_filter_by_workflow_name(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-1", "summarize", "pending")
        results = await store.list_doc_workflows("my-app", workflow_name="analyze")
        assert len(results) == 1 and results[0].workflow_name == "analyze"

    @pytest.mark.asyncio
    async def test_filter_by_doc_id(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-2", "analyze", "done")
        results = await store.list_doc_workflows("my-app", doc_id="doc-1")
        assert len(results) == 1 and results[0].doc_id == "doc-1"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-2", "analyze", "pending")
        done = await store.list_doc_workflows("my-app", status="done")
        assert len(done) == 1 and done[0].status == "done"

    @pytest.mark.asyncio
    async def test_filter_combined(self, store):
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-1", "summarize", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-2", "analyze", "done")
        results = await store.list_doc_workflows("my-app", workflow_name="analyze", doc_id="doc-1", status="done")
        assert len(results) == 1
        assert results[0].doc_id == "doc-1"
        assert results[0].workflow_name == "analyze"


class TestDeleteDocCleansDocWorkflowRegistry:
    @pytest.mark.asyncio
    async def test_delete_doc_removes_workflow_records(self, store):
        await store.save_doc(DocRecord(
            app_name="my-app", doc_id="doc-1", status="active",
            ingested_at="2026-01-01T00:00:00+00:00",
        ))
        await store.upsert_doc_workflow_status("my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("my-app", "doc-1", "summarize", "done")

        await store.delete_doc("my-app", "doc-1")

        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None
        assert await store.get_doc_workflow("my-app", "doc-1", "summarize") is None

    @pytest.mark.asyncio
    async def test_delete_doc_only_removes_target_doc(self, store):
        for doc_id in ("doc-1", "doc-2"):
            await store.save_doc(DocRecord(
                app_name="my-app", doc_id=doc_id, status="active",
                ingested_at="2026-01-01T00:00:00+00:00",
            ))
            await store.upsert_doc_workflow_status("my-app", doc_id, "analyze", "done")

        await store.delete_doc("my-app", "doc-1")

        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None
        assert await store.get_doc_workflow("my-app", "doc-2", "analyze") is not None
