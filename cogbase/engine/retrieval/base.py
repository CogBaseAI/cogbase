"""Abstract contract for retrieval executors.

A retriever takes a ``RouteResult`` (the router's decision) and returns a
``RetrievalResult`` — the evidence the reasoning engine will use to answer
the query.

Four concrete retrievers correspond to the four query patterns:

    StructuredRetriever  — Pattern A: queries the structured store only.
    VectorRetriever      — Pattern B: embeds the query and searches the vector store.
    HybridRetriever      — Pattern C/D: queries both stores and merges results.

The engine selects the right retriever based on ``RouteResult.pattern``; callers
should not need to instantiate retrievers directly — use ``HybridRetriever`` as
the single entry point if you want automatic dispatch.
"""

from __future__ import annotations

import abc

from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.engine.router import RouteResult


class RetrievalResult(BaseModel):
    """Evidence gathered from one or more stores for a single query.

    Attributes:
        structured_records: Raw dicts returned from the structured store.
            Empty list when the structured store was not queried.
        chunks: Ranked ``Chunk`` objects returned from the vector store.
            Empty list when the vector store was not queried.
        route: The routing decision that produced this result, preserved so
            downstream components (generation, skill orchestration) can
            inspect pattern, filters, and collection without re-routing.
    """

    structured_records: list[dict] = []
    chunks: list[Chunk] = []
    route: RouteResult


class RetrieverBase(abc.ABC):
    """Abstract retriever — executes a ``RouteResult`` against one or more stores."""

    @abc.abstractmethod
    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        """Fetch evidence matching the routing decision.

        Args:
            route: Output of a ``QueryRouter.route`` call.

        Returns:
            ``RetrievalResult`` populated with whatever evidence this retriever
            is responsible for.  Fields irrelevant to this retriever's pattern
            are left as empty lists.

        Raises:
            Any store or embedding error propagates to the caller — there is no
            silent fallback.
        """
