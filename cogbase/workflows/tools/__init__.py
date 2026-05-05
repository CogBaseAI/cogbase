"""Dispatch for built-in workflow step tools."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.stores import StructuredStoreBase, VectorStoreBase

if TYPE_CHECKING:
    from cogbase.config.config import WorkflowStepConfig


async def run_tool(
    step: "WorkflowStepConfig",
    ctx: dict,
    structured_store: StructuredStoreBase | None,
    vector_store: VectorStoreBase | None,
    embedder: EmbeddingBase | None,
    llm: LLMBase | None,
) -> dict[str, Any]:
    """Dispatch a single leaf step to its built-in tool implementation."""
    tool = step.tool
    if tool == "structured-query":
        from cogbase.workflows.tools.structured_query import run
        return await run(step, ctx, structured_store)
    if tool == "vector-search":
        from cogbase.workflows.tools.vector_search import run
        return await run(step, ctx, vector_store, embedder)
    if tool == "llm-structured":
        from cogbase.workflows.tools.llm_structured import run
        return await run(step, ctx, llm)
    if tool == "structured-save":
        from cogbase.workflows.tools.structured_save import run
        return await run(step, ctx, structured_store)
    raise ValueError(f"Unknown workflow tool: {tool!r}")
