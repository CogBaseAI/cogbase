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
import logging

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.retrieval.structured import StructuredRetriever
from cogbase.engine.retrieval.vector import VectorRetriever
from cogbase.engine.router import QueryPattern, RouteResult
from cogbase.embeddings import EmbeddingBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase

logger = logging.getLogger(__name__)


class HybridRetriever(RetrieverBase):
    """Dispatches to StructuredRetriever, VectorRetriever, or both.

    Use this as the single retriever in the engine — it inspects
    ``route.pattern`` and delegates automatically.

    For patterns C and D both stores are queried concurrently; the results are
    merged into a single ``RetrievalResult``.

    Args:
        structured_store: Any ``StructuredStoreBase`` implementation.
        vector_store:     Any ``VectorStoreBase`` implementation.  ``None``
                          disables vector retrieval — patterns B, C, and D
                          return empty chunks rather than raising.
        embedder:         Any ``EmbeddingBase`` implementation. Required when
                          *vector_store* is provided; ignored otherwise.
        top_k:            Number of vector-search results to return. Defaults to 10.
    """

    def __init__(
        self,
        structured_store: StructuredStoreBase | None = None,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        top_k: int = 10,
    ) -> None:
        self._structured = StructuredRetriever(structured_store) if structured_store is not None else None
        self._vector = (
            VectorRetriever(vector_store, embedder, top_k)
            if vector_store is not None and embedder is not None
            else None
        )

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        logger.info("hybrid_retriever.retrieve.start pattern=%s", route.pattern.value)
        match route.pattern:
            case QueryPattern.A:
                if self._structured is None:
                    logger.info("hybrid_retriever.retrieve.structured_disabled pattern=%s", route.pattern.value)
                    return RetrievalResult(route=route)
                result = await self._structured.retrieve(route)
                logger.debug(
                    "hybrid_retriever.retrieve.done pattern=%s structured_records=%d chunks=%d",
                    route.pattern.value,
                    len(result.structured_records),
                    len(result.chunks),
                )
                return result

            case QueryPattern.B:
                if self._vector is None:
                    logger.info("hybrid_retriever.retrieve.vector_disabled pattern=%s", route.pattern.value)
                    return RetrievalResult(route=route)
                result = await self._vector.retrieve(route)
                logger.debug(
                    "hybrid_retriever.retrieve.done pattern=%s structured_records=%d chunks=%d",
                    route.pattern.value,
                    len(result.structured_records),
                    len(result.chunks),
                )
                return result

            case QueryPattern.C | QueryPattern.D:
                # Both stores queried concurrently where possible; merge results.
                if self._vector is not None:
                    structured_task = asyncio.create_task(self._structured_safe(route))
                    vector_task = asyncio.create_task(self._vector.retrieve(route))
                    structured_result, vector_result = await asyncio.gather(
                        structured_task, vector_task
                    )
                    chunks = vector_result.chunks
                elif self._structured is not None:
                    structured_result = await self._structured_safe(route)
                    chunks = []
                else:
                    return RetrievalResult(route=route)
                result = RetrievalResult(
                    structured_records=structured_result.structured_records,
                    chunks=chunks,
                    route=route,
                )
                logger.debug(
                    "hybrid_retriever.retrieve.done pattern=%s structured_records=%d chunks=%d",
                    route.pattern.value,
                    len(result.structured_records),
                    len(result.chunks),
                )
                return result

    async def _structured_safe(self, route: RouteResult) -> RetrievalResult:
        """Query structured store, returning an empty result when no targets are known."""
        if self._structured is None or not route.structured_targets:
            return RetrievalResult(route=route)
        return await self._structured.retrieve(route)
