"""Pattern A retriever — queries the structured store only.

Used when the router classifies a query as Pattern A (structured lookup).
No embedding is computed and the vector store is never touched.

Example::

    from cogbase.engine.retrieval.structured import StructuredRetriever

    retriever = StructuredRetriever(structured_store)
    result = await retriever.retrieve(route)
    # result.structured_records — list of matching dicts
    # result.chunks             — always []
"""

from __future__ import annotations

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.router import RouteResult
from cogbase.stores.base import StructuredStoreBase


class StructuredRetriever(RetrieverBase):
    """Retrieves records from the structured store using filters from the route.

    When ``route.collection`` is ``None`` the retriever cannot determine which
    collection to query and raises ``ValueError`` — the router is expected to
    populate ``collection`` for Pattern A queries.

    When ``route.filters`` is ``None`` or empty, all records in the collection
    are returned (no filtering applied).

    Args:
        store: Any ``StructuredStoreBase`` implementation.
    """

    def __init__(self, store: StructuredStoreBase) -> None:
        self._store = store

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        if not route.collection:
            raise ValueError(
                "StructuredRetriever requires route.collection to be set. "
                "The query router should populate this for Pattern A queries."
            )

        records = await self._store.query(route.collection, route.filters or [])
        return RetrievalResult(structured_records=records, route=route)
