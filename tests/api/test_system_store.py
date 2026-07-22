"""Unit tests for api/system_store.py — SystemStore and AppRecord."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cogbase.stores.structured.memory import InMemoryStructuredStore
from api.system_store import AppRecord, DocRecord, DocWorkflowRecord, NamespaceRecord, SkillRecord, SystemStore, TaskRecord, new_app_id


def _make_record(name: str = "my-app", status: str = "active") -> AppRecord:
    return AppRecord(
        account_id="default", namespace_id="default",
        app_id=name,
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


class TestNewAppId:
    def test_starts_with_app_prefix(self):
        assert new_app_id().startswith("app_")

    def test_valid_identifier_prefix(self):
        # Scoped collection names are "<app_id>__<collection>" and must start
        # with a letter or underscore, never a digit.
        app_id = new_app_id()
        assert app_id[0].isalpha() or app_id[0] == "_"
        assert app_id.isidentifier()

    def test_unique_across_calls(self):
        ids = {new_app_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_hex_suffix_is_32_chars(self):
        app_id = new_app_id()
        suffix = app_id.removeprefix("app_")
        assert len(suffix) == 32
        # uuid4().hex is lowercase hexadecimal
        int(suffix, 16)
        assert suffix == suffix.lower()


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
        fetched = await store.get_app("default", "default", "my-app")
        assert fetched is not None
        assert fetched.name == "my-app"
        assert fetched.status == "active"

    @pytest.mark.asyncio
    async def test_get_app_returns_none_for_unknown(self, store):
        result = await store.get_app("default", "default", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_upserts_record(self, store):
        record = _make_record()
        await store.save_app(record)
        updated = record.model_copy(update={"status": "error", "error": "something failed"})
        await store.save_app(updated)
        fetched = await store.get_app("default", "default", "my-app")
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
        assert await store.get_app("default", "default", "my-app") is None

    @pytest.mark.asyncio
    async def test_delete_only_removes_target(self, store):
        await store.save_app(_make_record(name="a"))
        await store.save_app(_make_record(name="b"))
        await store.delete_app("a")
        assert await store.get_app("default", "default", "a") is None
        assert await store.get_app("default", "default", "b") is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, store):
        # Must not raise
        await store.delete_app("ghost")

    @pytest.mark.asyncio
    async def test_delete_cascades_doc_records(self, store):
        await store.save_app(_make_record(name="my-app"))
        await store.save_doc(DocRecord(
        account_id="default", namespace_id="default",
            app_id="my-app", doc_id="doc-1", status="active",
            ingested_at="2026-01-01T00:00:00+00:00",
        ))
        await store.save_doc(DocRecord(
        account_id="default", namespace_id="default",
            app_id="my-app", doc_id="doc-2", status="active",
            ingested_at="2026-01-01T00:00:00+00:00",
        ))
        await store.delete_app("my-app")
        assert await store.list_docs("my-app") == []

    @pytest.mark.asyncio
    async def test_delete_cascades_task_records(self, store):
        await store.save_app(_make_record(name="my-app"))
        await store.create_task(_make_task(task_id="t-1", app_id="my-app"))
        await store.create_task(_make_task(task_id="t-2", app_id="my-app", doc_id="doc-2"))
        await store.delete_app("my-app")
        assert await store.list_tasks("my-app") == []

    @pytest.mark.asyncio
    async def test_delete_cascades_doc_workflow_records(self, store):
        await store.save_app(_make_record(name="my-app"))
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "summarize", "pending")
        await store.delete_app("my-app")
        assert await store.list_doc_workflows("my-app") == []

    @pytest.mark.asyncio
    async def test_delete_does_not_affect_other_apps(self, store):
        for name in ("app-a", "app-b"):
            await store.save_app(_make_record(name=name))
            await store.save_doc(DocRecord(
        account_id="default", namespace_id="default",
                app_id=name, doc_id="doc-1", status="active",
                ingested_at="2026-01-01T00:00:00+00:00",
            ))
            await store.create_task(_make_task(task_id=f"t-{name}", app_id=name))
            await store.upsert_doc_workflow_status("default", "default", name, "doc-1", "analyze", "done")

        await store.delete_app("app-a")

        assert await store.get_app("default", "default", "app-b") is not None
        assert len(await store.list_docs("app-b")) == 1
        assert len(await store.list_tasks("app-b")) == 1
        assert len(await store.list_doc_workflows("app-b")) == 1


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "t-001",
    app_id: str = "my-app",
    task_type: str = "ingest",
    task_name: str = "ingest",
    doc_id: str | None = "doc-1",
    status: str = "pending",
) -> TaskRecord:
    return TaskRecord(
        account_id="default", namespace_id="default",
        task_id=task_id,
        app_id=app_id,
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
        assert fetched.app_id == "my-app"
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
        assert fetched.app_id == "my-app"

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
        await store.create_task(_make_task(task_id="t-1", app_id="app-a"))
        await store.create_task(_make_task(task_id="t-2", app_id="app-b"))
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


class TestCreateWorkflowTasks:
    @pytest.mark.asyncio
    async def test_creates_all_in_order_with_unique_ids(self, store):
        params_list = [{"issue": "a"}, {"issue": "b"}, None]
        records = await store.create_workflow_tasks("default", "default", "my-app", "analyze", "doc-1", params_list)
        assert len(records) == 3
        assert len({r.task_id for r in records}) == 3
        assert [r.params_json for r in records] == ['{"issue": "a"}', '{"issue": "b"}', None]
        for r in records:
            assert r.app_id == "my-app"
            assert r.task_type == "workflow"
            assert r.task_name == "analyze"
            assert r.doc_id == "doc-1"
            assert r.status == "pending"
            assert r.completed_at is None

    @pytest.mark.asyncio
    async def test_persists_every_record(self, store):
        records = await store.create_workflow_tasks("default", "default", "my-app", "wf", "doc-1", [{"x": 1}, {"x": 2}])
        for r in records:
            fetched = await store.get_task(r.task_id)
            assert fetched is not None
            assert fetched.params_json == r.params_json

    @pytest.mark.asyncio
    async def test_doc_id_may_be_none(self, store):
        records = await store.create_workflow_tasks("default", "default", "my-app", "wf", None, [None])
        assert records[0].doc_id is None

    @pytest.mark.asyncio
    async def test_empty_list_creates_nothing(self, store):
        records = await store.create_workflow_tasks("default", "default", "my-app", "wf", "doc-1", [])
        assert records == []


class TestCompleteWorkflowTask:
    @staticmethod
    async def _make_task_id(store):
        records = await store.create_workflow_tasks("default", "default", "my-app", "wf", "doc-1", [None])
        return records[0].task_id

    @pytest.mark.asyncio
    async def test_marks_done_on_success(self, store):
        task_id = await self._make_task_id(store)
        await store.complete_workflow_task(task_id, success=True)
        task = await store.get_task(task_id)
        assert task.status == "done"
        assert task.completed_at is not None
        assert task.error is None

    @pytest.mark.asyncio
    async def test_marks_failed_on_failure(self, store):
        task_id = await self._make_task_id(store)
        await store.complete_workflow_task(task_id, success=False, error="LLM timeout")
        task = await store.get_task(task_id)
        assert task.status == "failed"
        assert task.completed_at is not None
        assert task.error == "LLM timeout"

    @pytest.mark.asyncio
    async def test_completed_at_is_set(self, store):
        task_id = await self._make_task_id(store)
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
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "pending")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert record is not None
        assert record.app_id == "my-app"
        assert record.doc_id == "doc-1"
        assert record.workflow_name == "analyze"
        assert record.status == "pending"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, store):
        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_status(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "pending")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert record.status == "done"

    @pytest.mark.asyncio
    async def test_upsert_sets_updated_at(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "running")
        record = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        assert "T" in record.updated_at and record.updated_at.endswith("+00:00")

    @pytest.mark.asyncio
    async def test_distinct_workflows_for_same_doc(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "summarize", "pending")
        r_analyze = await store.get_doc_workflow("my-app", "doc-1", "analyze")
        r_summarize = await store.get_doc_workflow("my-app", "doc-1", "summarize")
        assert r_analyze.status == "done"
        assert r_summarize.status == "pending"

    @pytest.mark.asyncio
    async def test_same_workflow_distinct_docs(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "analyze", "failed")
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
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "analyze", "pending")
        results = await store.list_doc_workflows("my-app")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_isolated_by_app(self, store):
        await store.upsert_doc_workflow_status("default", "default", "app-a", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "app-b", "doc-1", "analyze", "done")
        assert len(await store.list_doc_workflows("app-a")) == 1
        assert len(await store.list_doc_workflows("app-b")) == 1
        assert len(await store.list_doc_workflows("app-c")) == 0

    @pytest.mark.asyncio
    async def test_filter_by_workflow_name(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "summarize", "pending")
        results = await store.list_doc_workflows("my-app", workflow_name="analyze")
        assert len(results) == 1 and results[0].workflow_name == "analyze"

    @pytest.mark.asyncio
    async def test_filter_by_doc_id(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "analyze", "done")
        results = await store.list_doc_workflows("my-app", doc_id="doc-1")
        assert len(results) == 1 and results[0].doc_id == "doc-1"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "analyze", "pending")
        done = await store.list_doc_workflows("my-app", status="done")
        assert len(done) == 1 and done[0].status == "done"

    @pytest.mark.asyncio
    async def test_filter_combined(self, store):
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "summarize", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-2", "analyze", "done")
        results = await store.list_doc_workflows("my-app", workflow_name="analyze", doc_id="doc-1", status="done")
        assert len(results) == 1
        assert results[0].doc_id == "doc-1"
        assert results[0].workflow_name == "analyze"


class TestDeleteDocCleansDocWorkflowRegistry:
    @pytest.mark.asyncio
    async def test_delete_doc_removes_workflow_records(self, store):
        await store.save_doc(DocRecord(
        account_id="default", namespace_id="default",
            app_id="my-app", doc_id="doc-1", status="active",
            ingested_at="2026-01-01T00:00:00+00:00",
        ))
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "analyze", "done")
        await store.upsert_doc_workflow_status("default", "default", "my-app", "doc-1", "summarize", "done")

        await store.delete_doc("my-app", "doc-1")

        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None
        assert await store.get_doc_workflow("my-app", "doc-1", "summarize") is None

    @pytest.mark.asyncio
    async def test_delete_doc_only_removes_target_doc(self, store):
        for doc_id in ("doc-1", "doc-2"):
            await store.save_doc(DocRecord(
        account_id="default", namespace_id="default",
                app_id="my-app", doc_id=doc_id, status="active",
                ingested_at="2026-01-01T00:00:00+00:00",
            ))
            await store.upsert_doc_workflow_status("default", "default", "my-app", doc_id, "analyze", "done")

        await store.delete_doc("my-app", "doc-1")

        assert await store.get_doc_workflow("my-app", "doc-1", "analyze") is None
        assert await store.get_doc_workflow("my-app", "doc-2", "analyze") is not None


def _make_namespace_record(
    namespace_id: str = "team-a",
    account_id: str = "default",
    description: str | None = None,
) -> NamespaceRecord:
    return NamespaceRecord(
        account_id=account_id,
        namespace_id=namespace_id,
        name=namespace_id,
        description=description,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class TestSystemStoreNamespaces:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        await store.save_namespace(_make_namespace_record(description="Team A"))
        got = await store.get_namespace("default", "team-a")
        assert got is not None
        assert got.namespace_id == "team-a"
        assert got.name == "team-a"
        assert got.description == "Team A"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, store):
        assert await store.get_namespace("default", "nope") is None

    @pytest.mark.asyncio
    async def test_get_is_scoped_by_account(self, store):
        await store.save_namespace(_make_namespace_record(account_id="acct-1"))
        assert await store.get_namespace("acct-1", "team-a") is not None
        assert await store.get_namespace("acct-2", "team-a") is None

    @pytest.mark.asyncio
    async def test_save_overwrites(self, store):
        await store.save_namespace(_make_namespace_record(description="v1"))
        await store.save_namespace(_make_namespace_record(description="v2"))
        got = await store.get_namespace("default", "team-a")
        assert got.description == "v2"
        assert len(await store.list_namespaces("default")) == 1

    @pytest.mark.asyncio
    async def test_list_scoped_by_account(self, store):
        await store.save_namespace(_make_namespace_record("team-a", account_id="acct-1"))
        await store.save_namespace(_make_namespace_record("team-b", account_id="acct-1"))
        await store.save_namespace(_make_namespace_record("team-c", account_id="acct-2"))
        rows = await store.list_namespaces("acct-1")
        assert {r.namespace_id for r in rows} == {"team-a", "team-b"}

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.save_namespace(_make_namespace_record())
        await store.delete_namespace("default", "team-a")
        assert await store.get_namespace("default", "team-a") is None

    @pytest.mark.asyncio
    async def test_ensure_creates_when_absent(self, store):
        await store.ensure_namespace("default", "team-x")
        got = await store.get_namespace("default", "team-x")
        assert got is not None
        # auto-registered namespace: name coincides with the id
        assert got.name == "team-x"

    @pytest.mark.asyncio
    async def test_ensure_is_idempotent_and_preserves_metadata(self, store):
        await store.save_namespace(_make_namespace_record("team-a", description="Team A"))
        await store.ensure_namespace("default", "team-a")
        got = await store.get_namespace("default", "team-a")
        # ensure must not clobber an existing record's metadata
        assert got.description == "Team A"
        assert len(await store.list_namespaces("default")) == 1


def _make_skill_record(skill_id: str = "uuid-1", name: str = "greeter") -> SkillRecord:
    return SkillRecord(
        account_id="default", namespace_id="default",
        skill_id=skill_id,
        name=name,
        description="Says hi.",
        metadata_json=None,
        bundle_key=f"{skill_id}.zip",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class TestSystemStoreSkills:
    @pytest.mark.asyncio
    async def test_save_and_get_skill(self, store):
        await store.save_skill(_make_skill_record())
        got = await store.get_skill("uuid-1")
        assert got is not None
        assert got.name == "greeter"
        assert got.bundle_key == "uuid-1.zip"

    @pytest.mark.asyncio
    async def test_get_missing_skill_returns_none(self, store):
        assert await store.get_skill("nope") is None

    @pytest.mark.asyncio
    async def test_list_skills(self, store):
        await store.save_skill(_make_skill_record("uuid-1", "alpha"))
        await store.save_skill(_make_skill_record("uuid-2", "beta"))
        rows = await store.list_skills()
        assert {r.skill_id for r in rows} == {"uuid-1", "uuid-2"}

    @pytest.mark.asyncio
    async def test_save_skill_overwrites(self, store):
        await store.save_skill(_make_skill_record("uuid-1", "v1"))
        await store.save_skill(_make_skill_record("uuid-1", "v2"))
        got = await store.get_skill("uuid-1")
        assert got.name == "v2"
        assert len(await store.list_skills()) == 1

    @pytest.mark.asyncio
    async def test_delete_skill(self, store):
        await store.save_skill(_make_skill_record())
        await store.delete_skill("uuid-1")
        assert await store.get_skill("uuid-1") is None


# ---------------------------------------------------------------------------
# Session index (conversation history list)
# ---------------------------------------------------------------------------


class TestSystemStoreSessions:
    @pytest.mark.asyncio
    async def test_touch_creates_row_and_get_returns_it(self, store):
        await store.touch_session("default", "default", "app-a", "sess-1", "hi there")
        got = await store.get_session("app-a", "sess-1")
        assert got is not None
        assert got.app_id == "app-a"
        assert got.title == "hi there"
        assert got.message_count == 1
        assert got.status == "open"

    @pytest.mark.asyncio
    async def test_touch_increments_count_and_keeps_title(self, store):
        await store.touch_session("default", "default", "app-a", "sess-1", "first message")
        await store.touch_session("default", "default", "app-a", "sess-1", "second message")
        got = await store.get_session("app-a", "sess-1")
        assert got.message_count == 2
        # Title stays the first user message; later turns don't overwrite it.
        assert got.title == "first message"

    @pytest.mark.asyncio
    async def test_get_is_scoped_by_app_id(self, store):
        # session_id is client-suppliable, so it must be scoped by the resolved
        # app_id — another app addressing the same id must not read the row.
        await store.touch_session("default", "default", "app-a", "sess-1", "hi")
        assert await store.get_session("app-a", "sess-1") is not None
        assert await store.get_session("app-b", "sess-1") is None

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, store):
        assert await store.get_session("app-a", "nope") is None

    @pytest.mark.asyncio
    async def test_close_is_scoped_by_app_id(self, store):
        await store.touch_session("default", "default", "app-a", "sess-1", "hi")
        # A foreign app addressing the same session_id must not close app-a's row.
        await store.close_session_record("app-b", "sess-1")
        assert (await store.get_session("app-a", "sess-1")).status == "open"
        # The owning app closes it.
        await store.close_session_record("app-a", "sess-1")
        assert (await store.get_session("app-a", "sess-1")).status == "closed"

    @pytest.mark.asyncio
    async def test_close_missing_row_is_noop(self, store):
        # Must not raise (a session opened but never took a turn has no index row).
        await store.close_session_record("app-a", "sess-1")

    @pytest.mark.asyncio
    async def test_delete_is_scoped_by_app_id(self, store):
        await store.touch_session("default", "default", "app-a", "sess-1", "hi")
        # A foreign app addressing the same session_id must not delete app-a's row.
        await store.delete_session_record("app-b", "sess-1")
        assert await store.get_session("app-a", "sess-1") is not None
        # The owning app deletes it.
        await store.delete_session_record("app-a", "sess-1")
        assert await store.get_session("app-a", "sess-1") is None

    @pytest.mark.asyncio
    async def test_same_session_id_across_apps_do_not_collide(self, store):
        # (app_id, session_id) is the identity, so a reused session_id in a second
        # app is a distinct row — neither touch clobbers the other's.
        await store.touch_session("default", "default", "app-a", "sess-1", "a-first")
        await store.touch_session("default", "default", "app-b", "sess-1", "b-first")
        a = await store.get_session("app-a", "sess-1")
        b = await store.get_session("app-b", "sess-1")
        assert a.title == "a-first" and a.message_count == 1
        assert b.title == "b-first" and b.message_count == 1
        # Deleting one leaves the other intact.
        await store.delete_session_record("app-a", "sess-1")
        assert await store.get_session("app-a", "sess-1") is None
        assert await store.get_session("app-b", "sess-1") is not None

    @pytest.mark.asyncio
    async def test_list_session_records_isolated_by_app(self, store):
        await store.touch_session("default", "default", "app-a", "sess-1", "a1")
        await store.touch_session("default", "default", "app-a", "sess-2", "a2")
        await store.touch_session("default", "default", "app-b", "sess-3", "b1")
        assert {r.session_id for r in await store.list_session_records("app-a")} == {"sess-1", "sess-2"}
        assert {r.session_id for r in await store.list_session_records("app-b")} == {"sess-3"}
        assert await store.list_session_records("app-c") == []
