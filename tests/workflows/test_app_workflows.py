"""Unit tests for workflow integration in CogBaseApp."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogbase.config.config import WorkflowConfig, WorkflowStepConfig, WorkflowTriggerConfig, WhenCondition
from cogbase.core.app import CogBaseApp
from cogbase.core.models import Document
from cogbase.core.query_runner import QueryRunner
from cogbase.pipeline.ingestion_pipeline import IngestionPipeline
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.workflows.runner import WorkflowRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_app(workflow_runners: dict | None = None) -> CogBaseApp:
    """Build the smallest possible CogBaseApp for workflow tests."""
    store = InMemoryStructuredStore()
    pipeline = IngestionPipeline(name="test")
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": "ok", "tool_calls": None})
    runner = QueryRunner(llm=llm, structured_store=store)
    return CogBaseApp("test-app", pipeline, runner, workflow_runners=workflow_runners)


def _make_wf_runner(
    name: str = "my-wf",
    trigger_type: str = "manual",
    when_metadata: dict | None = None,
) -> WorkflowRunner:
    trigger = WorkflowTriggerConfig(
        type=trigger_type,
        when=WhenCondition(metadata=when_metadata or {}) if when_metadata else None,
    )
    wf = WorkflowConfig(name=name, trigger=trigger, steps=[])
    return WorkflowRunner(wf)


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
        runner = _make_wf_runner("check", trigger_type="after_ingest")
        app = _minimal_app(workflow_runners={"check": runner})

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([Document(doc_id="d-001", text="some text")])
            assert mock_ct.called

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
        )
        app = _minimal_app(workflow_runners={"check": runner})

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([
                Document(doc_id="d-001", text="contract text", metadata={"doc_type": "contract"}),
            ])
            assert mock_ct.called

    async def test_after_ingest_metadata_filter_no_match(self):
        runner = _make_wf_runner(
            "check",
            trigger_type="after_ingest",
            when_metadata={"doc_type": "contract"},
        )
        app = _minimal_app(workflow_runners={"check": runner})

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([
                Document(doc_id="d-001", text="rules text", metadata={"doc_type": "rules"}),
            ])
            mock_ct.assert_not_called()

    async def test_after_ingest_not_fired_for_failed_docs(self):
        """A document that fails pipeline ingestion must not trigger the workflow."""
        pipeline = MagicMock(spec=IngestionPipeline)
        from cogbase.pipeline.ingestion_pipeline import IngestResult
        pipeline.ingest_documents = AsyncMock(return_value=[
            IngestResult(doc_id="d-fail", success=False, error=RuntimeError("boom")),
        ])

        store = InMemoryStructuredStore()
        llm = MagicMock()
        llm.complete = AsyncMock(return_value={"content": "ok", "tool_calls": None})
        qrunner = QueryRunner(llm=llm, structured_store=store)

        runner = _make_wf_runner("check", trigger_type="after_ingest")
        app = CogBaseApp("test-app", pipeline, qrunner, workflow_runners={"check": runner})

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([Document(doc_id="d-fail", text="text")])
            mock_ct.assert_not_called()

    async def test_after_ingest_passes_doc_id_as_param(self):
        runner = _make_wf_runner("check", trigger_type="after_ingest")
        app = _minimal_app(workflow_runners={"check": runner})

        passed_params: list[dict] = []
        futures: list = []

        async def _bg(wf_runner, params):
            passed_params.append(params)

        app._run_workflow_bg = _bg

        def _fake_create_task(coro):
            fut = asyncio.ensure_future(coro)
            futures.append(fut)
            return fut

        with patch(self._PATCH, side_effect=_fake_create_task):
            await app.ingest_documents([Document(doc_id="d-001", text="text")])

        if futures:
            await asyncio.gather(*futures, return_exceptions=True)

        assert passed_params == [{"doc_id": "d-001"}]

    async def test_multiple_after_ingest_workflows_all_fire(self):
        runners = {
            "wf-a": _make_wf_runner("wf-a", trigger_type="after_ingest"),
            "wf-b": _make_wf_runner("wf-b", trigger_type="after_ingest"),
        }
        app = _minimal_app(workflow_runners=runners)

        with patch(self._PATCH, side_effect=self._discard_task) as mock_ct:
            await app.ingest_documents([Document(doc_id="d-001", text="text")])
            assert mock_ct.call_count == 2


# ---------------------------------------------------------------------------
# _run_workflow_bg
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
        await app._run_workflow_bg(wf_runner, {"doc_id": "d-001"})

    async def test_bg_logs_exception_and_does_not_raise(self):
        wf_runner = _make_wf_runner("wf")

        async def _failing_run(params):
            raise RuntimeError("workflow failed")
            yield  # make it an async generator

        wf_runner.run = _failing_run

        app = _minimal_app()
        # Must not propagate the exception
        await app._run_workflow_bg(wf_runner, {"doc_id": "d-001"})
