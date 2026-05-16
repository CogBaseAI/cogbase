"""vector-search tool — embed a query and search a vector collection."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from cogbase.embeddings.base import EmbeddingBase
from cogbase.stores import VectorStoreBase
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import VectorSearchStepConfig


async def run(
    step: "VectorSearchStepConfig",
    ctx: dict,
    vector_store: VectorStoreBase | None,
    embedder: EmbeddingBase | None,
) -> dict[str, Any]:
    if vector_store is None:
        raise RuntimeError("vector-search requires a vector store")
    if embedder is None:
        raise RuntimeError("vector-search requires an embedder")

    query_text = str(render_value(step.query, ctx))
    (embedding,) = await embedder.embed([query_text])
    chunks = await vector_store.search(step.collection, query_text, embedding, top_k=step.top_k)
    return {"chunks": chunks}
