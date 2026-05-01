"""Abstract adapter contract for structured stores."""

import abc
from typing import TypeVar

from pydantic import BaseModel

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
    (return / delete all).  How each filter is evaluated is adapter-defined -
    backends with native JSON support (e.g. PostgreSQL) may push JSON-column
    filters to the engine; others may evaluate them in Python after the fetch.
    """

    @abc.abstractmethod
    async def create_collection(self, schema: CollectionSchema) -> None:
        """Declare a collection. Idempotent - safe to call on every startup."""

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

    @abc.abstractmethod
    async def list_collections(self) -> list[str]:
        """Return the names of all registered collections."""

    async def query_as(
        self,
        collection: str,
        filters: list[Filter] | None,
        model: type[M],
        fields: list[str] | None = None,
    ) -> list[M]:
        """Typed wrapper around ``query`` - deserialises results into ``model``."""
        return [
            model.model_validate(row)
            for row in await self.query(collection, filters, fields)
        ]
