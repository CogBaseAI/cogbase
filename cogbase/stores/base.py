"""Abstract adapter contracts for structured and vector stores."""

import abc
from typing import TypeVar

from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.stores.filters import Filter
from cogbase.stores.schema import CollectionSchema

M = TypeVar("M", bound=BaseModel)


class StructuredStoreBase(abc.ABC):
    """Generic contract for any structured store backend.

    Collections must be declared with ``create_collection`` before use.
    The schema controls column types, indexing, and which field is the primary key
    (used for upsert semantics in ``save``).

    Filters are ``Filter`` objects built with ``Col``::

        from cogbase.stores.filters import Col

        store.query("facts", [
            Col("type") == "notice_period",
            Col("confidence") >= 0.8,
            Col("doc_id").in_(["doc-1", "doc-2"]),
        ])

    All filters are ANDed together.  Primitive-column filters are pushed to the
    database engine; JSON-column filters are applied in Python after the fetch.
    ``None`` or an empty list means "no filter" (return / delete all).
    """

    @abc.abstractmethod
    def create_collection(self, schema: CollectionSchema) -> None:
        """Declare a collection. Idempotent — safe to call on every startup."""

    @abc.abstractmethod
    def save(self, collection: str, records: list[BaseModel]) -> None:
        """Upsert records into ``collection``.

        Fields not declared in the schema are dropped; the ``id_field`` drives
        the upsert key.
        """

    @abc.abstractmethod
    def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]:
        """Return all records matching every filter as plain dicts.

        Use ``query_as`` to deserialise into a Pydantic model.
        """

    @abc.abstractmethod
    def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None:
        """Delete all records matching every filter.

        ``None`` or ``[]`` deletes the entire collection's contents.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def query_as(
        self,
        collection: str,
        filters: list[Filter] | None,
        model: type[M],
    ) -> list[M]:
        """Typed wrapper around ``query`` — deserialises results into ``model``."""
        return [model.model_validate(row) for row in self.query(collection, filters)]


class VectorStoreBase(abc.ABC):
    """Contract for any vector store backend."""

    @abc.abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None: ...

    @abc.abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]: ...

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None: ...
