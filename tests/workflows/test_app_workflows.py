"""Unit tests for workflow integration in CogBaseApp."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.config.config import (
    WorkflowConfig,
    WorkflowTriggerConfig,
    WorkflowParamsFromCollectionConfig,
    WhenCondition,
)
from cogbase.core.app import CogBaseApp
from cogbase.core.models import Document
from cogbase.core.query_runner import MemoryTiers, QueryRunner, RetrievalResources
from cogbase.pipeline.ingestion_pipeline import IngestionPipeline
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.workflows.runner import WorkflowRunner


def _mock_task_store():
    m = MagicMock()
    m.create_workflow_task = AsyncMock(return_value=None)
    m.complete_workflow_task = AsyncMock()
    m.upsert_doc_workflow_status = AsyncMock()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_app(workflow_runners: dict | None = None) -> CogBaseApp:
    """Build the smallest possible CogBaseApp for workflow tests."""
    store = InMemoryStructuredStore()
    pipeline = IngestionPipeline(name="test")
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": "ok", "tool_calls": None})
    doc_store = InMemoryDocumentStore()
    runner = QueryRunner(app_id="test-app", llm=llm, resources=RetrievalResources(document_store=doc_store, structured_store=store))
    return CogBaseApp(
        "test-app",
        [pipeline],
        runner,
        app_id="test-app",
        document_store=doc_store,
        structured_store=store,
        workflow_runners=workflow_runners or {},
        llm=llm,
        task_store=_mock_task_store(),
    )


_DEFAULT_PARAMS_FROM_COLLECTION = WorkflowParamsFromCollectionConfig(
    collection="docs",
    filters={"doc_id": "{{ doc.doc_id }}"},
    params={"doc_id": "{{ record.doc_id }}"},
)


def _make_wf_runner(
    name: str = "my-wf",
    trigger_type: str = "manual",
    when_metadata: dict | None = None,
    params_from_collection: WorkflowParamsFromCollectionConfig | None = None,
) -> WorkflowRunner:
    trigger = WorkflowTriggerConfig(
        type=trigger_type,
        when=WhenCondition(metadata=when_metadata or {}) if when_metadata else None,
    )
    wf = WorkflowConfig(
        name=name,
        trigger=trigger,
        params_from_collection=params_from_collection or _DEFAULT_PARAMS_FROM_COLLECTION,
        steps=[],
    )
    return WorkflowRunner(wf)


def _after_ingest_source() -> WorkflowParamsFromCollectionConfig:
    return WorkflowParamsFromCollectionConfig(
        collection="facts",
        filters={"doc_id": "{{ doc.doc_id }}"},
        params={"issue": "{{ record.issue }}"},
    )


# ---------------------------------------------------------------------------
# Workflow accessors
# ---------------------------------------------------------------------------

class TestWorkflowAccessors:
    def test_workflows_empty_by_default(self):
        app = _minimal_app()
        assert app.workflows == []

    def test_workflows_lists_names(self):
        runners = {
            "wf-a": _make_wf_runner("wf-a"),
            "wf-b": _make_wf_runner("wf-b"),
        }
        app = _minimal_app(workflow_runners=runners)
        assert set(app.workflows) == {"wf-a", "wf-b"}

    def test_get_workflow_returns_runner(self):
        runner = _make_wf_runner("compliance")
        app = _minimal_app(workflow_runners={"compliance": runner})
        assert app.get_workflow("compliance") is runner

    def test_get_workflow_unknown_raises_key_error(self):
        app = _minimal_app()
        with pytest.raises(KeyError, match="no-such-wf"):
            app.get_workflow("no-such-wf")

    def test_workflow_runners_none_defaults_to_empty(self):
        app = _minimal_app(workflow_runners=None)
        assert app._workflows == {}


# ---------------------------------------------------------------------------
# after_ingest trigger
# ---------------------------------------------------------------------------

class TestAfterIngestTrigger:
    _PATCH = "cogbase.core.app.asyncio.create_task"

    @staticmethod
    def _discard_task(coro):
        """Close the coroutine immediately so it doesn't leak."""
        coro.close()
        return MagicMock()

    async def test_after_ingest_fires_for_matching_doc(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        await app.ingest_documents([Document(doc_id="d-001", text="some text")])
        app._structured_store.query.assert_called()

    async def test_after_ingest_not_fired_for_manual_trigger(self):
        runner = _make_wf_runner("check", trigger_type="manual")
        app = _minimal_app(workflow_runners={"check": runner})

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([Document(doc_id="d-001", text="some text")])
            mock_ct.assert_not_called()

    async def test_after_ingest_metadata_filter_matches(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            when_metadata={"doc_type": "contract"},
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        await app.ingest_documents([
            Document(doc_id="d-001", text="contract text", metadata={"doc_type": "contract"}),
        ])
        app._structured_store.query.assert_called()

    async def test_after_ingest_metadata_filter_no_match(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            when_metadata={"doc_type": "contract"},
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([
                Document(doc_id="d-001", text="rules text", metadata={"doc_type": "rules"}),
            ])
            mock_ct.assert_not_called()

    async def test_after_ingest_not_fired_for_failed_docs(self):
        """A document that fails pipeline ingestion must not trigger the workflow."""
        pipeline = MagicMock(spec=IngestionPipeline)
        pipeline.name = "test-pipeline"
        from cogbase.pipeline.ingestion_pipeline import IngestResult
        pipeline.ingest_documents = AsyncMock(return_value=[
            IngestResult(doc_id="d-fail", success=False, error=RuntimeError("boom")),
        ])

        store = InMemoryStructuredStore()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value={"content": "ok", "tool_calls": None})
        doc_store = InMemoryDocumentStore()
        qrunner = QueryRunner(app_id="test-app", llm=llm, resources=RetrievalResources(document_store=doc_store, structured_store=store))

        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            params_from_collection=_after_ingest_source(),
        )
        app = CogBaseApp("test-app", [pipeline], qrunner, app_id="test-app", document_store=doc_store, structured_store=store, workflow_runners={"check": runner}, llm=llm, task_store=_mock_task_store())

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([Document(doc_id="d-fail", text="text")])
            mock_ct.assert_not_called()

    def test_workflow_requires_params_from_collection(self):
        with pytest.raises(ValueError, match="params_from_collection"):
            WorkflowConfig(name="check", trigger=WorkflowTriggerConfig(), steps=[])

    async def test_after_ingest_can_build_params_from_structured_records(self):
        source = WorkflowParamsFromCollectionConfig(
            collection="facts",
            filters={"doc_id": "{{ doc.doc_id }}"},
            params={"issue": "{{ record.issue }}"},
        )
        runner = _make_wf_runner(
            "detect-contradictions",
            trigger_type="after_ingest",
            params_from_collection=source,
        )
        app = _minimal_app(workflow_runners={"detect-contradictions": runner})
        app._structured_store.query = AsyncMock(return_value=[
            {"doc_id": "d-001", "issue": "late_delivery"},
            {"doc_id": "d-001", "issue": "late_delivery"},
            {"doc_id": "d-001", "issue": "payment"},
        ])

        params = await app.resolve_workflow_params(runner, "d-001")
        assert params == [{"issue": "late_delivery"}, {"issue": "payment"}]

    async def test_multiple_after_ingest_workflows_all_fire(self):
        runners = {
            "wf-a": _make_wf_runner("wf-a", trigger_type="after_ingest", params_from_collection=_after_ingest_source()),
            "wf-b": _make_wf_runner("wf-b", trigger_type="after_ingest", params_from_collection=_after_ingest_source()),
        }
        app = _minimal_app(workflow_runners=runners)
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        await app.ingest_documents([Document(doc_id="d-001", text="text")])
        assert app._structured_store.query.call_count == 2

    async def test_upsert_doc_workflow_status_pending_called_on_ingest(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        with patch(self._PATCH, side_effect=self._discard_task):
            await app.ingest_documents([Document(doc_id="d-001", text="some text")])

        app._task_store.upsert_doc_workflow_status.assert_awaited_once_with(
            "test-app", "d-001", "check", "pending"
        )

    async def test_create_workflow_task_called_for_after_ingest(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._task_store.create_workflow_task = AsyncMock(return_value="task-001")
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "late_delivery"}])

        with patch(self._PATCH, side_effect=self._discard_task):
            await app.ingest_documents([Document(doc_id="d-001", text="some text")])

        import json
        app._task_store.create_workflow_task.assert_awaited_once_with(
            "test-app", "check", "d-001", json.dumps({"issue": "late_delivery"})
        )

    async def test_pending_not_marked_when_params_empty(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[])

        await app.ingest_documents([Document(doc_id="d-001", text="some text")])

        app._task_store.upsert_doc_workflow_status.assert_not_awaited()

    async def test_manual_trigger_marks_ready_but_no_task_created(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="manual",
            params_from_collection=_after_ingest_source(),
        )
        app = _minimal_app(workflow_runners={"check": runner})
        app._structured_store.query = AsyncMock(return_value=[{"doc_id": "d-001", "issue": "x"}])

        await app.ingest_documents([Document(doc_id="d-001", text="some text")])

        app._task_store.upsert_doc_workflow_status.assert_awaited_once_with(
            "test-app", "d-001", "check", "ready"
        )
        app._task_store.create_workflow_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# _run_workflow_tasks_bg
# ---------------------------------------------------------------------------

class TestRunWorkflowBg:
    async def test_bg_drains_runner_and_logs_done(self):
        wf_runner = _make_wf_runner("wf")  # real WorkflowRunner with empty workflow

        records_emitted: list[dict] = []
        original_run = wf_runner.run

        async def _patched_run(params):
            yield {"finding_id": "f1"}
            yield {"finding_id": "f2"}
            async for r in original_run(params):
                records_emitted.append(r)

        wf_runner.run = _patched_run

        app = _minimal_app()
        # Should complete without raising
        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, None)])

    async def test_bg_logs_exception_and_does_not_raise(self):
        wf_runner = _make_wf_runner("wf")

        async def _failing_run(params):
            raise RuntimeError("workflow failed")
            yield  # make it an async generator

        wf_runner.run = _failing_run

        app = _minimal_app()
        # Must not propagate the exception
        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, None)])

    async def test_bg_marks_done_after_all_tasks_succeed(self):
        wf_runner = _make_wf_runner("wf")
        app = _minimal_app()

        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, None)])

        app._task_store.upsert_doc_workflow_status.assert_awaited_once_with(
            "test-app", "d-001", "wf", "done"
        )

    async def test_bg_marks_failed_after_task_exception(self):
        wf_runner = _make_wf_runner("wf")

        async def _failing_run(params):
            raise RuntimeError("boom")
            yield

        wf_runner.run = _failing_run
        app = _minimal_app()

        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, None)])

        app._task_store.upsert_doc_workflow_status.assert_awaited_once_with(
            "test-app", "d-001", "wf", "failed"
        )

    async def test_bg_completes_task_success_when_task_id_present(self):
        wf_runner = _make_wf_runner("wf")
        app = _minimal_app()

        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, "task-42")])

        app._task_store.complete_workflow_task.assert_awaited_once_with("task-42", success=True)

    async def test_bg_completes_task_failure_when_task_id_present(self):
        wf_runner = _make_wf_runner("wf")

        async def _failing_run(params):
            raise RuntimeError("workflow error")
            yield

        wf_runner.run = _failing_run
        app = _minimal_app()

        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, "task-99")])

        app._task_store.complete_workflow_task.assert_awaited_once_with(
            "task-99", success=False, error="workflow error"
        )

    async def test_bg_no_complete_task_when_task_id_none(self):
        wf_runner = _make_wf_runner("wf")
        app = _minimal_app()

        await app._run_workflow_tasks_bg(wf_runner, "d-001", [({"doc_id": "d-001"}, None)])

        app._task_store.complete_workflow_task.assert_not_awaited()
