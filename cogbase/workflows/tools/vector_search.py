"""vector-search tool — embed a query and search a vector collection."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from cogbase.embeddings.base import EmbeddingBase
from cogbase.stores import VectorStoreBase
from cogbase.stores.filters import Col
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.config.config import VectorSearchStepConfig

logger = logging.getLogger(__name__)


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
    filters = [
        Col(field) == render_value(val_template, ctx)
        for field, val_template in step.filters.items()
    ]
    chunks = await vector_store.search(
        step.collection, query_text, embedding, top_k=step.top_k,
        filters=filters or None,
    )
    logger.info(
        "workflow.tool.vector_search collection=%s top_k=%d query=%s filters=%d chunks=%d",
        step.collection, step.top_k, query_text[:120], len(filters), len(chunks),
    )
    return {"chunks": chunks}
