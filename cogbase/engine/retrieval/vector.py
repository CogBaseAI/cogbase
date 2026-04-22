"""Pattern B retriever — embeds the query and searches the vector store.

Used when the router classifies a query as Pattern B (semantic search).
The structured store is never touched.

Example::

    from cogbase.engine.retrieval.vector import VectorRetriever

    retriever = VectorRetriever(vector_store, embedder, top_k=10)
    result = await retriever.retrieve(route)
    # result.chunks             — ranked Chunk objects
    # result.structured_records — always []
"""

from __future__ import annotations

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.router import RouteResult
from cogbase.embeddings import EmbeddingBase
from cogbase.stores.base import VectorStoreBase


class VectorRetriever(RetrieverBase):
    """Embeds ``route.semantic_query`` and returns the nearest chunks.

    Args:
        collection_name: Vector store collection to search.
        store:           Any ``VectorStoreBase`` implementation.
        embedder:        Any ``EmbeddingBase`` implementation.  The same embedder used
                         at ingest time must be used here so the vector spaces match.
        top_k:           Number of nearest neighbours to return. Defaults to 10.
    """

    def __init__(
        self,
        collection_name: str,
        store: VectorStoreBase,
        embedder: EmbeddingBase,
        top_k: int = 10,
    ) -> None:
        self._collection_name = collection_name
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        (query_embedding,) = await self._embedder.embed([route.semantic_query])
        if query_embedding is None:
            raise RuntimeError("Embedder returned no embedding vector for the query.")
        chunks = await self._store.search(self._collection_name, query_embedding, self._top_k)
        return RetrievalResult(chunks=chunks, route=route)
