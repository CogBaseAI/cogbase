"""Pattern C/D retriever — queries both stores and merges results.

Also doubles as a dispatch entry point: call ``HybridRetriever.retrieve`` for
any pattern and it will delegate to the right underlying retriever.

Pattern mapping:
    A — StructuredRetriever only.
    B — VectorRetriever only.
    C — Both retrievers; results merged.
    D — Both retrievers; results merged (same as C — the caller differentiates
        output format, not retrieval strategy).

Example::

    from cogbase.engine.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        top_k=10,
    )
    result = await retriever.retrieve(route)
    # result.structured_records — from structured store (patterns A, C, D)
    # result.chunks             — from vector store     (patterns B, C, D)
"""

from __future__ import annotations

import asyncio

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.retrieval.structured import StructuredRetriever
from cogbase.engine.retrieval.vector import VectorRetriever
from cogbase.engine.router import QueryPattern, RouteResult
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase


class HybridRetriever(RetrieverBase):
    """Dispatches to StructuredRetriever, VectorRetriever, or both.

    Use this as the single retriever in the engine — it inspects
    ``route.pattern`` and delegates automatically.

    For patterns C and D both stores are queried concurrently; the results are
    merged into a single ``RetrievalResult``.

    Args:
        structured_store: Any ``StructuredStoreBase`` implementation.
        vector_store:     Any ``VectorStoreBase`` implementation.
        embedder:         Any ``EmbedderBase`` implementation. Required for
                          patterns B, C, and D; unused for pattern A.
        top_k:            Number of vector-search results to return. Defaults to 10.
    """

    def __init__(
        self,
        structured_store: StructuredStoreBase,
        vector_store: VectorStoreBase,
        embedder: EmbedderBase,
        top_k: int = 10,
    ) -> None:
        self._structured = StructuredRetriever(structured_store)
        self._vector = VectorRetriever(vector_store, embedder, top_k)

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        match route.pattern:
            case QueryPattern.A:
                return await self._structured.retrieve(route)

            case QueryPattern.B:
                return await self._vector.retrieve(route)

            case QueryPattern.C | QueryPattern.D:
                # Both stores queried concurrently; merge into one result.
                structured_task = asyncio.create_task(self._structured_safe(route))
                vector_task = asyncio.create_task(self._vector.retrieve(route))
                structured_result, vector_result = await asyncio.gather(
                    structured_task, vector_task
                )
                return RetrievalResult(
                    structured_records=structured_result.structured_records,
                    chunks=vector_result.chunks,
                    route=route,
                )

    async def _structured_safe(self, route: RouteResult) -> RetrievalResult:
        """Query structured store, returning an empty result when collection is unknown."""
        if not route.collection:
            return RetrievalResult(route=route)
        return await self._structured.retrieve(route)
