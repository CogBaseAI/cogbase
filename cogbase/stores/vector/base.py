"""Abstract adapter contracts for vector stores."""

import abc

from pydantic import BaseModel, field_validator

from cogbase.core.models import Chunk
from cogbase.stores.filters import Filter


class VectorCollectionSchema(BaseModel):
    """Schema for a vector store collection (namespace/index).

    Args:
        name:        Collection name - must be a valid identifier
                     (``[a-zA-Z_][a-zA-Z0-9_]*``).
        dimensions:  Embedding vector dimensionality. All chunks upserted into
                     this collection must carry embeddings of exactly this length.
        description: Short description shown to the LLM to help it choose the
                     right collection (e.g. "Full-text passage chunks for detailed
                     document questions").
        metadata:    Optional free-form str->str metadata stored at the
                     collection level (e.g. embedding model name, distance
                     metric).
    """

    name: str
    dimensions: int
    description: str
    metadata: dict[str, str] = {}

    @field_validator("description")
    @classmethod
    def _non_empty_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("VectorCollectionSchema.description must be set")
        return v

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", v):
            raise ValueError(
                f"Collection name '{v}' is invalid - use letters, digits, and underscores only"
            )
        return v

    @field_validator("dimensions")
    @classmethod
    def _positive_dimensions(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"dimensions must be positive, got {v}")
        return v


class VectorStoreBase(abc.ABC):
    """Contract for any vector store backend.

    Collections must be declared with ``create_collection`` before use.
    Each collection is an isolated namespace - chunks in different collections
    never mix during search or delete.

    Example::

        schema = VectorCollectionSchema(name="legal_chunks", dimensions=1536, description="Full-text passage chunks")
        await store.create_collection(schema)
        await store.upsert("legal_chunks", chunks)
        results = await store.search("legal_chunks", "notice period", query_embedding, top_k=5)
        await store.delete("legal_chunks", doc_id="doc-42")
    """

    @abc.abstractmethod
    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """Declare a vector collection. Idempotent - safe to call on every startup."""

    @abc.abstractmethod
    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Insert or update chunks in ``collection``.

        Each chunk must carry an ``embedding`` whose length matches the
        collection's declared ``dimensions``.
        """

    @abc.abstractmethod
    async def search(
        self,
        collection: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[Chunk]:
        """Return the ``top_k`` nearest chunks from ``collection``.

        Args:
            collection:      Target collection name.
            query:           Original query text.  Backends that support keyword or
                             hybrid search (e.g. Elasticsearch, Weaviate, pgvector with
                             full-text) may combine this with ``query_embedding`` for
                             better recall.  Pure ANN backends may ignore it.
            query_embedding: Query vector; must match the collection's dimensions.
            top_k:           Maximum number of results to return.
            filters:         AND-combined metadata filter expressions applied before
                             (or alongside) the ANN search.  Supports top-level Chunk
                             fields (``doc_id``, ``chunk_id``) and dot-notation for
                             metadata sub-keys (``metadata.source``, ``metadata.page``).
                             ``None`` / ``[]`` means no filter.  Example::

                                 from cogbase.stores.filters import Col

                                 await store.search(
                                     "legal_chunks", "notice period", embedding, top_k=5,
                                     filters=[
                                         Col("doc_id").in_(["doc-1", "doc-2"]),
                                         Col("metadata.section") == "definitions",
                                     ],
                                 )

            fields:          Chunk field names to populate in each returned object.
                             ``None`` / ``[]`` returns all fields (default).  Backends
                             that support projection (e.g. Pinecone ``include_metadata``
                             / ``include_values``) may use this to reduce payload size.
                             Unknown names are silently ignored.
        """

    @abc.abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Drop ``collection`` and all its chunks permanently."""

    @abc.abstractmethod
    async def delete(self, collection: str, doc_id: str) -> None:
        """Delete all chunks for ``doc_id`` from ``collection``."""

    @abc.abstractmethod
    async def list_collections(self) -> list[str]:
        """Return the names of all registered collections."""
