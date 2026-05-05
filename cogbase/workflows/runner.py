"""WorkflowRunner — executes a WorkflowConfig against live resources."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, TYPE_CHECKING

from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.stores import StructuredStoreBase, VectorStoreBase
from cogbase.workflows.context import render_value
from cogbase.workflows.tools import run_tool

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
        structured_store: StructuredStoreBase | None = None,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        llm: LLMBase | None = None,
    ) -> None:
        self.workflow = workflow
        self._structured_store = structured_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm

    async def run(self, params: dict[str, Any]) -> AsyncGenerator[dict, None]:
        """Execute the workflow with *params* and yield each saved record."""
        ctx: dict[str, Any] = {"input": params, "steps": {}}
        async for record in self._run_steps(self.workflow.steps, ctx):
            yield record

    async def _run_steps(
        self,
        steps: list["WorkflowStepConfig"],
        ctx: dict[str, Any],
    ) -> AsyncGenerator[dict, None]:
        for step in steps:
            if step.foreach is not None:
                items = render_value(step.foreach, ctx)
                if not isinstance(items, list):
                    raise ValueError(
                        f"Workflow step '{step.id}' foreach resolved to "
                        f"{type(items).__name__!r}, expected list"
                    )
                for item in items:
                    # Fresh steps namespace per iteration so outputs don't cross-contaminate.
                    iter_ctx = {**ctx, "item": item, "steps": dict(ctx["steps"])}
                    async for record in self._run_steps(step.steps or [], iter_ctx):
                        yield record
            elif step.tool is not None:
                logger.info(
                    "workflow.step.start workflow=%s step=%s tool=%s",
                    self.workflow.name, step.id, step.tool,
                )
                output = await run_tool(
                    step, ctx,
                    self._structured_store,
                    self._vector_store,
                    self._embedder,
                    self._llm,
                )
                ctx["steps"][step.id] = output
                logger.info(
                    "workflow.step.done workflow=%s step=%s tool=%s",
                    self.workflow.name, step.id, step.tool,
                )
                if step.tool == "structured-save":
                    for record in output.get("records", []):
                        if hasattr(record, "model_dump"):
                            yield record.model_dump()
                        elif isinstance(record, dict):
                            yield record
            else:
                logger.warning(
                    "workflow.step.skipped workflow=%s step=%s (no tool, no foreach)",
                    self.workflow.name, step.id,
                )
