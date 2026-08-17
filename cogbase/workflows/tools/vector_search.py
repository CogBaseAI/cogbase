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
    # Drop the vectors before the chunks enter the step context. Stores return
    # them populated, and the usual next step is `llm-structured` with
    # `{{ steps.<id>.chunks }}` as an input — which JSON-serializes each chunk in
    # full, so a top_k of 6 over 1536-dimension embeddings spends ~45k tokens per
    # call on numbers no model can read. Nothing downstream can use a raw vector:
    # this step is the only thing that embeds, and it does its own.
    #
    # Stripped here rather than via the `fields` projection `search()` accepts, so
    # the result is identical on every backend — the projection's per-field
    # semantics differ between FAISS and pgvector, and this needs to be a fact
    # about the tool, not about the store behind it.
    chunks = [c.model_copy(update={"embedding": None}) for c in chunks]
    logger.info(
        "workflow.tool.vector_search collection=%s top_k=%d query=%s filters=%d chunks=%d",
        step.collection, step.top_k, query_text[:120], len(filters), len(chunks),
    )
    # The counterpart of structured-query's min_records, and it exists for the same
    # failure: a search over a collection that was never populated returns [], the
    # step that consumes the chunks has nothing to judge, and the run *succeeds*
    # having examined nothing. Left at its default of 0 an empty result is treated
    # as a legitimate "no match", which it is for any search whose collection is
    # produced by the run rather than seeded before it.
    if len(chunks) < step.min_chunks:
        raise RuntimeError(
            f"vector-search step {step.id!r}: collection {step.collection!r} returned "
            f"{len(chunks)} chunks, below min_chunks={step.min_chunks}. The collection "
            "is a precondition of this run — it is empty, or the filters exclude "
            "everything in it — so the run fails rather than reporting that nothing "
            "was found."
        )
    return {"chunks": chunks}
