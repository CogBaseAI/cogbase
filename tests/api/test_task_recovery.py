"""Tests for durable background-task recovery (api/task_runner.py).

These exercise the startup sweep directly (no HTTP): seed orphaned task records
into an in-memory system store, then assert ``recover_orphaned_tasks`` requeues
and completes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from api.app_cache import AppCache
from api.system_store import AppRecord, SystemStore, TaskRecord, TaskStatus, new_app_id
from api.task_runner import recover_orphaned_tasks
from cogbase.core.models import DocWorkflowStatus
from cogbase.stores.structured.memory import InMemoryStructuredStore

APP_NAME = "recovery-app"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _FakeIngestResult:
    doc_id: str
    success: bool = True
    records_extracted: int = 1
    chunks_written: int = 2
    error: Exception | None = None

    @property
    def ingested_nothing(self) -> bool:
        return self.success and self.chunks_written == 0 and self.records_extracted == 0


@pytest_asyncio.fixture
async def store() -> SystemStore:
    s = SystemStore(store=InMemoryStructuredStore())
    await s.setup()
    return s


async def _seed_app(store: SystemStore, *, status: str = "active") -> str:
    app_id = new_app_id()
    await store.save_app(AppRecord(
        app_id=app_id, account_id="default", namespace_id="default",
        name=APP_NAME, config_yaml="name: recovery-app",
        status=status, created_at=_now(), updated_at=_now(),
    ))
    return app_id


def _mock_ingest_app(app_id: str) -> MagicMock:
    """App whose document store returns bytes and whose ingest always succeeds."""
    app = MagicMock()
    app.app_id = app_id
    app.account_id = "default"
    app.namespace_id = "default"
    app.name = APP_NAME
    app.document_store = MagicMock()
    app.document_store.load_bytes = AsyncMock(return_value=b"raw bytes")
    app.ingest_documents = AsyncMock(
        side_effect=lambda docs: [_FakeIngestResult(doc_id=docs[0].doc_id)]
    )
    return app


def _ingest_task(app_id: str, task_id: str, doc_id: str, status: TaskStatus) -> TaskRecord:
    return TaskRecord(
        account_id="default", namespace_id="default",
        task_id=task_id, app_id=app_id, task_type="ingest", task_name="ingest",
        doc_id=doc_id, batch_id="b1",
        params_json=json.dumps({"doc_path": f"originals/{doc_id}.txt",
                                "doc_metadata": {"source_filename": f"{doc_id}.txt"}}),
        status=status, created_at=_now(),
    )


# ---------------------------------------------------------------------------
# Ingest recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovers_pending_and_running_ingest_tasks(store):
    app_id = await _seed_app(store)
    app = _mock_ingest_app(app_id)
    await store.create_task(_ingest_task(app_id, "t-pending", "doc-a", TaskStatus.PENDING))
    await store.create_task(_ingest_task(app_id, "t-running", "doc-b", TaskStatus.RUNNING))

    async def resolve(name):
        return app

    with patch("api.task_runner.parse_to_markdown", return_value="parsed text"):
        n = await recover_orphaned_tasks(store, resolve, AppCache())

    assert n == 2
    for tid, doc_id in (("t-pending", "doc-a"), ("t-running", "doc-b")):
        task = await store.get_task(tid)
        assert task.status == TaskStatus.DONE
        doc = await store.get_doc(app_id, doc_id)
        assert doc is not None and doc.status == "active"


@pytest.mark.asyncio
async def test_running_task_is_reset_to_pending_before_execution(store):
    """A RUNNING task (interrupted mid-flight) is requeued, not left stranded."""
    app_id = await _seed_app(store)
    app = _mock_ingest_app(app_id)
    await store.create_task(_ingest_task(app_id, "t-running", "doc-b", TaskStatus.RUNNING))

    async def resolve(name):
        return app

    with patch("api.task_runner.parse_to_markdown", return_value="text"):
        await recover_orphaned_tasks(store, resolve, AppCache())

    # It was picked up despite starting in RUNNING, and ran to completion.
    assert (await store.get_task("t-running")).status == TaskStatus.DONE
    app.ingest_documents.assert_awaited()


@pytest.mark.asyncio
async def test_recovery_is_idempotent_on_rerun(store):
    app_id = await _seed_app(store)
    app = _mock_ingest_app(app_id)
    await store.create_task(_ingest_task(app_id, "t1", "doc-a", TaskStatus.PENDING))

    async def resolve(name):
        return app

    with patch("api.task_runner.parse_to_markdown", return_value="text"):
        first = await recover_orphaned_tasks(store, resolve, AppCache())
        # Second sweep finds nothing left to do — the task is already DONE.
        second = await recover_orphaned_tasks(store, resolve, AppCache())

    assert first == 1
    assert second == 0
    assert app.ingest_documents.await_count == 1


# ---------------------------------------------------------------------------
# Distill recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovers_distill_task(store):
    app_id = await _seed_app(store)
    app = MagicMock()
    app.app_id = app_id
    app.account_id = "default"
    app.namespace_id = "default"
    app.name = APP_NAME
    app.distiller = MagicMock()
    app.distiller.distill_session = AsyncMock(return_value=["mem-1"])
    await store.create_task(TaskRecord(
        account_id="default", namespace_id="default",
        task_id="d1", app_id=app_id, task_type="distill", task_name="distill",
        doc_id="sess-1", params_json=json.dumps({"session_id": "sess-1"}),
        status=TaskStatus.PENDING, created_at=_now(),
    ))

    async def resolve(name):
        return app

    await recover_orphaned_tasks(store, resolve, AppCache())

    assert (await store.get_task("d1")).status == TaskStatus.DONE
    app.distiller.distill_session.assert_awaited_once_with(session_id="sess-1")


@pytest.mark.asyncio
async def test_distill_task_fails_cleanly_without_distiller(store):
    app_id = await _seed_app(store)
    app = MagicMock()
    app.app_id = app_id
    app.account_id = "default"
    app.namespace_id = "default"
    app.name = APP_NAME
    app.distiller = None
    await store.create_task(TaskRecord(
        account_id="default", namespace_id="default",
        task_id="d2", app_id=app_id, task_type="distill", task_name="distill",
        doc_id="sess-2", params_json=json.dumps({"session_id": "sess-2"}),
        status=TaskStatus.PENDING, created_at=_now(),
    ))

    async def resolve(name):
        return app

    await recover_orphaned_tasks(store, resolve, AppCache())

    task = await store.get_task("d2")
    assert task.status == TaskStatus.FAILED
    assert "distiller" in (task.error or "")


# ---------------------------------------------------------------------------
# Workflow recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovers_workflow_task_and_rolls_up_doc_status(store):
    app_id = await _seed_app(store)
    app = MagicMock()
    app.app_id = app_id
    app.account_id = "default"
    app.namespace_id = "default"
    app.name = APP_NAME

    async def _run(params):
        for rec in [{"ok": True}]:
            yield rec

    runner = MagicMock()
    runner.run = _run
    app.get_workflow = MagicMock(return_value=runner)

    await store.create_task(TaskRecord(
        account_id="default", namespace_id="default",
        task_id="w1", app_id=app_id, task_type="workflow", task_name="summarize",
        doc_id="doc-x", params_json=json.dumps({"doc_id": "doc-x"}),
        status=TaskStatus.PENDING, created_at=_now(),
    ))

    async def resolve(name):
        return app

    await recover_orphaned_tasks(store, resolve, AppCache())

    assert (await store.get_task("w1")).status == TaskStatus.DONE
    wf = await store.get_doc_workflow(app_id, "doc-x", "summarize")
    assert wf is not None and wf.status == DocWorkflowStatus.DONE


# ---------------------------------------------------------------------------
# Dispatch edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactive_app_is_skipped(store):
    app_id = await _seed_app(store, status="error")
    await store.create_task(_ingest_task(app_id, "t1", "doc-a", TaskStatus.PENDING))

    resolve = AsyncMock()
    n = await recover_orphaned_tasks(store, resolve, AppCache())

    assert n == 0
    resolve.assert_not_called()
    assert (await store.get_task("t1")).status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_unresolvable_app_is_skipped(store):
    app_id = await _seed_app(store)
    await store.create_task(_ingest_task(app_id, "t1", "doc-a", TaskStatus.PENDING))

    async def resolve(name):
        return None

    n = await recover_orphaned_tasks(store, resolve, AppCache())
    assert n == 0


@pytest.mark.asyncio
async def test_unknown_task_type_is_skipped_not_crashed(store):
    app_id = await _seed_app(store)
    app = MagicMock(app_id=app_id, name=APP_NAME)
    await store.create_task(TaskRecord(
        account_id="default", namespace_id="default",
        task_id="u1", app_id=app_id, task_type="mystery", task_name="mystery",
        status=TaskStatus.PENDING, created_at=_now(),
    ))

    async def resolve(name):
        return app

    # Counts as requeued (dispatched) but the executor no-ops with a warning.
    n = await recover_orphaned_tasks(store, resolve, AppCache())
    assert n == 1
    assert (await store.get_task("u1")).status == TaskStatus.PENDING
