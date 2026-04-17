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

from cogbase.core.models import Chunk
from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.router import RouteResult
from cogbase.embeddings import EmbeddingBase
from cogbase.stores.base import VectorStoreBase


class VectorRetriever(RetrieverBase):
    """Embeds ``route.semantic_query`` and returns the nearest chunks.

    Args:
        store:    Any ``VectorStoreBase`` implementation.
        embedder: Any ``EmbeddingBase`` implementation.  The same embedder used
                  at ingest time must be used here so the vector spaces match.
        top_k:    Number of nearest neighbours to return. Defaults to 10.
    """

    def __init__(
        self,
        store: VectorStoreBase,
        embedder: EmbeddingBase,
        top_k: int = 10,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    async def retrieve(self, route: RouteResult) -> RetrievalResult:
        # Embed the cleaned query as a single synthetic chunk, then extract its vector.
        query_chunk = Chunk(doc_id="__query__", text=route.semantic_query)
        (embedded,) = await self._embedder.embed([query_chunk])

        if embedded.embedding is None:
            raise RuntimeError("Embedder returned a chunk with no embedding vector.")

        chunks = await self._store.search(embedded.embedding, self._top_k)
        return RetrievalResult(chunks=chunks, route=route)
