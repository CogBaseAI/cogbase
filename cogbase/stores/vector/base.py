"""Abstract adapter contracts for vector stores."""

from __future__ import annotations

import abc

from pydantic import BaseModel, field_validator

from cogbase.core.models import Chunk
from cogbase.stores.filters import Filter
from cogbase.stores.schema import validate_resource_name
from cogbase.stores.scope import AppScope


class VectorCollectionSchema(BaseModel):
    """Schema for a vector store collection (namespace/index).

    Args:
        name:        Collection name — must start with a letter or underscore,
                     followed by letters, digits, underscores, or hyphens
                     (``[a-zA-Z_][a-zA-Z0-9_-]*``).
        dimensions:  Embedding vector dimensionality. All chunks upserted into
                     this collection must carry embeddings of exactly this length.
        description: Short description shown to the LLM to help it choose the
                     right collection (e.g. "Full-text passage chunks for detailed
                     document questions").
        metadata_fields: Document metadata keys projected into each chunk's
                         metadata at ingest time.
    """

    name: str
    dimensions: int
    description: str
    metadata_fields: list[str] = []

    @field_validator("description")
    @classmethod
    def _non_empty_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("VectorCollectionSchema.description must be set")
        return v

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        return validate_resource_name(v)

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
        await store.delete("memories", chunk_ids=["mem-1", "mem-2"])
        await store.delete_doc("legal_chunks", doc_id="doc-42")
    """

    def __init__(self, scope: AppScope | None = None) -> None:
        self._scope = scope

    def _c(self, collection: str) -> str:
        """Return the backend-internal name for *collection* (bare name → scoped name)."""
        prefix = self._scope.prefix() if self._scope else None
        return f"{prefix}__{collection}" if prefix else collection

    def with_scope(self, scope: AppScope) -> "VectorStoreBase":
        """Return a scoped proxy that prefixes all collection names with *scope*."""
        from cogbase.stores.scoped import ScopedVectorStore
        return ScopedVectorStore(self, scope)

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

    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Chunk]:
        """Return every chunk matching *filters*, without a similarity search.

        The counterpart of :meth:`StructuredStoreBase.query`, and the difference
        from :meth:`search` is the point: ``search`` ranks by distance and truncates
        to ``top_k``, so a caller that needs *all* the matches has to guess a ``top_k``
        large enough and gets silent, distance-ordered truncation when the guess is
        wrong. This returns the whole matching set, ordered by ``chunk_id`` so the
        result is stable across backends and calls.

        Use it to enumerate — verifying that seeded reference data landed, listing a
        document's chunks, counting a collection. Do not use it to retrieve: with no
        query vector there is nothing to rank by, and ``limit`` here truncates in id
        order, not by relevance.

        Args:
            collection: Target collection name.
            filters:    AND-combined filter expressions, same grammar as ``search``
                        (top-level ``Chunk`` fields, dot-notation for metadata
                        sub-keys). ``None`` / ``[]`` returns the whole collection.
            fields:     Chunk field names to populate. ``None`` / ``[]`` returns all,
                        matching ``search`` — which means embeddings too. Pass an
                        explicit list on a large scan; a full read of a chunk
                        collection is one vector per row of pure payload.
            limit:      Maximum chunks to return, applied after ordering. ``None``
                        (the default) returns every match, which is the semantic that
                        makes this an enumeration API.

        Raises:
            NotImplementedError: if the backend cannot filter without a query vector.

        **Not abstract, deliberately.** Most backends have this natively — pgvector is
        SQL, Qdrant has ``scroll``, Milvus ``query``, Chroma ``get(where=)``, Weaviate
        a filtered ``Get`` — but some are vector-first and genuinely do not: Pinecone's
        ``query`` requires a vector, and offers only ``fetch`` by id and ``list`` by id
        prefix. The workaround there is a zero-vector query with a guessed ``top_k``,
        which is the silent truncation this method exists to avoid, so a backend that
        cannot do it should say so and let the caller fail rather than emulate it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot query by filter without a query vector. "
            "Backends that are vector-first (e.g. Pinecone) support retrieval by "
            "similarity and fetch by id, but not enumeration by metadata; a caller "
            "that needs the whole matching set has to keep the records in a "
            "structured store as well."
        )

    @abc.abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Drop ``collection`` and all its chunks permanently."""

    @abc.abstractmethod
    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        """Delete the chunks identified by ``chunk_ids`` from ``collection``.

        Record-level deletion (the counterpart of ``upsert``): removes exactly
        the chunks whose ``chunk_id`` is listed, regardless of which document
        they belong to. This is the right call for collections whose records are
        addressed individually rather than by document (e.g. long-term memory
        records). ``chunk_id`` values that are not present are silently ignored;
        an empty list is a no-op. To remove every chunk of a source document,
        use :meth:`delete_doc`.
        """

    @abc.abstractmethod
    async def delete_doc(self, collection: str, doc_id: str) -> None:
        """Delete all chunks for ``doc_id`` from ``collection``.

        Document-level deletion: removes every chunk produced from a single
        source document. For record-level deletion (e.g. individual memory
        records) use :meth:`delete` instead.
        """

