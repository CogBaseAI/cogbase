"""NamespacedStructuredStore — a transparent wrapper that prefixes collection names.

Allows multiple applications to share a single physical store backend (e.g. one
SQLite file or one Postgres database) without collection-name collisions.

Every public ``StructuredStoreBase`` method that takes a ``collection`` argument
applies the prefix transparently, so pack code never needs to know about it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from cogbase.stores.base import StructuredStoreBase
from cogbase.stores.filters import Filter
from cogbase.stores.schema import CollectionSchema


class NamespacedStructuredStore(StructuredStoreBase):
    """Wraps a shared ``StructuredStoreBase``, prefixing all collection names.

    Collection names are prefixed as ``{namespace}__{collection}`` where the
    namespace is derived from the application name with non-alphanumeric
    characters replaced by underscores.

    Args:
        store:     The underlying shared store instance.
        namespace: String used to prefix all collection names; typically the
                   application name.
    """

    def __init__(self, store: StructuredStoreBase, namespace: str) -> None:
        self._store = store
        # Sanitize: keep only letters, digits, and underscores.
        self._prefix = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)

    def _ns(self, collection: str) -> str:
        return f"{self._prefix}__{collection}"

    async def create_collection(self, schema: CollectionSchema) -> None:
        ns_schema = schema.model_copy(update={"name": self._ns(schema.name)})
        await self._store.create_collection(ns_schema)

    async def save(self, collection: str, records: list[BaseModel]) -> None:
        await self._store.save(self._ns(collection), records)

    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        return await self._store.query(self._ns(collection), filters, fields)

    async def update_collection(self, schema: CollectionSchema) -> None:
        ns_schema = schema.model_copy(update={"name": self._ns(schema.name)})
        await self._store.update_collection(ns_schema)

    async def delete_records(
        self, collection: str, filters: list[Filter] | None = None
    ) -> None:
        await self._store.delete_records(self._ns(collection), filters)
