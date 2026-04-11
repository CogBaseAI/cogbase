"""Pattern A retriever — queries the structured store only.

Used when the router classifies a query as Pattern A (structured lookup).
No embedding is computed and the vector store is never touched.

Example::

    from cogbase.engine.retrieval.structured import StructuredRetriever

    retriever = StructuredRetriever(structured_store)
    result = await retriever.retrieve(route)
    # result.structured_records — list of matching dicts (merged across all targets)
    # result.chunks             — always []
"""

from __future__ import annotations

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.router import RouteResult
from cogbase.stores.base import StructuredStoreBase


class StructuredRetriever(RetrieverBase):
    """Retrieves records from the structured store using targets from the route.

    Each ``CollectionTarget`` in ``route.structured_targets`` is queried
    independently; results are merged in order into a single list.

    Raises ``ValueError`` when ``route.structured_targets`` is empty — the
    router is expected to populate at least one target for Pattern A queries.

    When a target's ``filters`` list is empty, all records in that collection
    are returned (no filtering applied).

    Args:
        store: Any ``StructuredStoreBase`` implementation.
    """

    def __init__(self, store: StructuredStoreBase) -> None:
        self._store = store

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        if not route.structured_targets:
            raise ValueError(
                "StructuredRetriever requires at least one entry in "
                "route.structured_targets. The query router should populate "
                "this for Pattern A queries; provide a collection schema to "
                "LLMRouter so it can determine the correct collection(s)."
            )

        all_records: list[dict] = []
        for target in route.structured_targets:
            records = await self._store.query(target.collection, target.filters)
            all_records.extend(records)

        return RetrievalResult(structured_records=all_records, route=route)
