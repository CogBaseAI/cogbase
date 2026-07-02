"""WorkflowRunner — executes a WorkflowConfig against live resources."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, TYPE_CHECKING

from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.stores import Col, StructuredStoreBase, VectorStoreBase
from cogbase.workflows.context import render_value
from cogbase.workflows.tools import run_tool

from cogbase.config.config import ForeachStepConfig, StructuredSaveStepConfig, _iter_save_steps

if TYPE_CHECKING:
    from cogbase.config.config import WorkflowConfig, WorkflowStepConfig

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes a ``WorkflowConfig`` sequentially, yielding saved records.

    Resources (stores, embedder, LLM) are injected at construction time so the
    runner is self-contained.  Call :meth:`run` with the workflow's input
    parameters to start execution.

    Yields
    ------
    dict
        One dict per record written by a ``structured-save`` step, in the order
        they are saved.  Callers can stream these as SSE events or collect them
        into a list.
    """

    def __init__(
        self,
        workflow: "WorkflowConfig",
        *,
        app_id: str | None = None,
        structured_store: StructuredStoreBase | None = None,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        llm: LLMBase | None = None,
    ) -> None:
        self.workflow = workflow
        # Stable app id, included in every log line so a workflow run can be
        # traced back to the application that owns it.
        self.app_id = app_id
        self._structured_store = structured_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm

    async def purge_document(self, doc_id: str) -> None:
        """Delete this doc's prior output from every structured-save collection.

        For each ``structured-save`` step, delete rows where any of the step's
        ``purge_by`` fields equals ``doc_id`` (OR across fields via one delete
        each). Deletes only rows referencing this doc, so a re-run replaces the
        doc's own findings and never removes cross-document findings it isn't
        authoritative for (e.g. a contradiction between two *other* docs).
        Deleting a ``doc_id`` absent from a collection is a no-op — safe on first
        ingest. Call once per (doc, workflow) before the param-set fan-out, so
        multiple param-sets can't purge each other's regenerated rows.
        """
        if self._structured_store is None:
            return

        # collection -> linkage fields (union across steps writing the collection)
        targets: dict[str, set[str]] = {}
        for step in _iter_save_steps(self.workflow.steps):
            targets.setdefault(step.collection, set()).update(step.purge_by)

        for collection, fields in targets.items():
            for field in sorted(fields):
                await self._structured_store.delete_records(
                    collection, [Col(field) == doc_id]
                )

    async def run(self, params: dict[str, Any]) -> AsyncGenerator[dict, None]:
        """Execute the workflow with *params* and yield each saved record."""
        logger.info(
            "workflow.run.start app=%s workflow=%s steps=%d params=%s",
            self.app_id, self.workflow.name, len(self.workflow.steps), params,
        )
        ctx: dict[str, Any] = {"input": params, "steps": {}}
        saved = 0
        async for record in self._run_steps(self.workflow.steps, ctx):
            saved += 1
            yield record
        logger.info(
            "workflow.run.done app=%s workflow=%s saved_records=%d",
            self.app_id, self.workflow.name, saved,
        )

    async def _run_steps(
        self,
        steps: list["WorkflowStepConfig"],
        ctx: dict[str, Any],
    ) -> AsyncGenerator[dict, None]:
        for step in steps:
            if isinstance(step, ForeachStepConfig):
                items = render_value(step.foreach, ctx)
                if not isinstance(items, list):
                    raise ValueError(
                        f"Workflow step '{step.id}' foreach resolved to "
                        f"{type(items).__name__!r}, expected list"
                    )
                logger.info(
                    "workflow.foreach.start app=%s workflow=%s step=%s items=%d",
                    self.app_id, self.workflow.name, step.id, len(items),
                )
                for idx, item in enumerate(items):
                    logger.info(
                        "workflow.foreach.iter app=%s workflow=%s step=%s iter=%d/%d",
                        self.app_id, self.workflow.name, step.id, idx + 1, len(items),
                    )
                    # Fresh steps namespace per iteration so outputs don't cross-contaminate.
                    iter_ctx = {**ctx, "item": item, "steps": dict(ctx["steps"])}
                    async for record in self._run_steps(step.steps, iter_ctx):
                        yield record
                logger.info(
                    "workflow.foreach.done app=%s workflow=%s step=%s items=%d",
                    self.app_id, self.workflow.name, step.id, len(items),
                )
            else:
                logger.info(
                    "workflow.step.start app=%s workflow=%s step=%s tool=%s",
                    self.app_id, self.workflow.name, step.id, step.tool,
                )
                try:
                    output = await run_tool(
                        step, ctx,
                        self._structured_store,
                        self._vector_store,
                        self._embedder,
                        self._llm,
                    )
                except Exception:
                    logger.exception(
                        "workflow.step.failed app=%s workflow=%s step=%s tool=%s",
                        self.app_id, self.workflow.name, step.id, step.tool,
                    )
                    raise
                ctx["steps"][step.id] = output
                logger.info(
                    "workflow.step.done app=%s workflow=%s step=%s tool=%s result_keys=%s",
                    self.app_id, self.workflow.name, step.id, step.tool,
                    list(output.keys()),
                )
                if isinstance(step, StructuredSaveStepConfig):
                    for record in output.get("records", []):
                        if hasattr(record, "model_dump"):
                            yield record.model_dump()
                        elif isinstance(record, dict):
                            yield record
