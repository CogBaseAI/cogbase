"""Abstract adapter contracts for structured and vector stores."""

import abc
from typing import TypeVar

from pydantic import BaseModel, field_validator

from cogbase.core.models import Chunk
from cogbase.stores.filters import Filter
from cogbase.stores.schema import CollectionSchema

M = TypeVar("M", bound=BaseModel)


class StructuredStoreBase(abc.ABC):
    """Generic contract for any structured store backend.

    Collections must be declared with ``create_collection`` before use.
    The schema controls column types, indexing, and which fields make up the
    primary key (used for upsert semantics in ``save``).

    Filters are ``Filter`` objects built with ``Col``::

        from cogbase.stores.filters import Col

        await store.query("facts", [
            Col("type") == "notice_period",
            Col("confidence") >= 0.8,
            Col("doc_id").in_(["doc-1", "doc-2"]),
        ])

    All filters are ANDed together.  ``None`` or an empty list means "no filter"
    (return / delete all).  How each filter is evaluated is adapter-defined —
    backends with native JSON support (e.g. PostgreSQL) may push JSON-column
    filters to the engine; others may evaluate them in Python after the fetch.
    """

    @abc.abstractmethod
    async def create_collection(self, schema: CollectionSchema) -> None:
        """Declare a collection. Idempotent — safe to call on every startup."""

    @abc.abstractmethod
    async def save(self, collection: str, records: list[BaseModel]) -> None:
        """Upsert records into ``collection``.

        Fields not declared in the schema are dropped; ``primary_fields`` drive
        the upsert key.
        """

    @abc.abstractmethod
    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Return all records matching every filter as plain dicts.

        Args:
            collection: Target collection name.
            filters:    AND-combined filter expressions.  ``None`` / ``[]`` means no filter.
            fields:     Field names to include in each returned dict.  ``None`` / ``[]``
                        returns all fields (default behaviour, backward-compatible).
                        Unknown field names are silently ignored.

        Use ``query_as`` to deserialise into a Pydantic model.
        """

    @abc.abstractmethod
    async def update_collection(self, schema: CollectionSchema) -> None:
        """Migrate an existing collection to a new schema.

        - Fields present in *schema* but absent from the current schema are added
          (new rows receive ``None``; existing rows receive ``None`` for the new column).
        - Fields absent from *schema* but present in the current schema are removed
          and their data is permanently discarded.
        - The ``primary_fields`` must remain the same; changing them raises
          ``ValueError``.
        - Calling this on a collection that does not exist raises ``KeyError``.

        Use ``create_collection`` for first-time setup; ``update_collection`` for
        subsequent schema changes.
        """

    @abc.abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Drop ``collection`` and all its records permanently."""

    @abc.abstractmethod
    async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        """Delete all records matching every filter.

        ``None`` or ``[]`` deletes the entire collection's contents.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    async def query_as(
        self,
        collection: str,
        filters: list[Filter] | None,
        model: type[M],
        fields: list[str] | None = None,
    ) -> list[M]:
        """Typed wrapper around ``query`` — deserialises results into ``model``."""
        return [
            model.model_validate(row)
            for row in await self.query(collection, filters, fields)
        ]


class VectorCollectionSchema(BaseModel):
    """Schema for a vector store collection (namespace/index).

    Args:
        name:       Collection name — must be a valid identifier
                    (``[a-zA-Z_][a-zA-Z0-9_]*``).
        dimensions: Embedding vector dimensionality. All chunks upserted into
                    this collection must carry embeddings of exactly this length.
        metadata:   Optional free-form str→str metadata stored at the
                    collection level (e.g. embedding model name, distance
                    metric).
    """

    name: str
    dimensions: int
    metadata: dict[str, str] = {}

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", v):
            raise ValueError(
                f"Collection name '{v}' is invalid — use letters, digits, and underscores only"
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
    Each collection is an isolated namespace — chunks in different collections
    never mix during search or delete.

    Example::

        schema = VectorCollectionSchema(name="legal_chunks", dimensions=1536)
        await store.create_collection(schema)
        await store.upsert("legal_chunks", chunks)
        results = await store.search("legal_chunks", query_embedding, top_k=5)
        await store.delete("legal_chunks", doc_id="doc-42")
    """

    @abc.abstractmethod
    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        """Declare a vector collection. Idempotent — safe to call on every startup."""

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
        query_embedding: list[float],
        top_k: int,
    ) -> list[Chunk]:
        """Return the ``top_k`` nearest chunks from ``collection``."""

    @abc.abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Drop ``collection`` and all its chunks permanently."""

    @abc.abstractmethod
    async def delete(self, collection: str, doc_id: str) -> None:
        """Delete all chunks for ``doc_id`` from ``collection``."""
