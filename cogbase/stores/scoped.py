"""Scoped proxy wrappers for shared store instances.

Used by ``VectorStoreBase.with_scope``, ``StructuredStoreBase.with_scope``, and
``DocumentStoreBase.with_scope`` to let a shared (system-level) store be used by
multiple applications without collection-name collisions.  All public methods
delegate to the inner store after prefixing the collection name via ``_c()``.
"""

from __future__ import annotations

from cogbase.core.models import Chunk
from cogbase.stores.document.base import DocumentStoreBase
from cogbase.stores.filters import Filter
from cogbase.stores.schema import CollectionSchema
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.base import StructuredStoreBase
from cogbase.stores.vector.base import VectorCollectionSchema, VectorStoreBase


class ScopedVectorStore(VectorStoreBase):
    """Vector store proxy that transparently prefixes every collection name."""

    def __init__(self, inner: VectorStoreBase, scope: AppScope) -> None:
        super().__init__(scope)
        self._inner = inner

    async def create_collection(self, schema: VectorCollectionSchema) -> None:
        scoped = schema.model_copy(update={"name": self._c(schema.name)})
        await self._inner.create_collection(scoped)

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        await self._inner.upsert(self._c(collection), chunks)

    async def search(
        self,
        collection: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[Chunk]:
        return await self._inner.search(
            self._c(collection), query, query_embedding, top_k, filters, fields
        )

    async def delete_collection(self, collection: str) -> None:
        await self._inner.delete_collection(self._c(collection))

    async def delete(self, collection: str, doc_id: str) -> None:
        await self._inner.delete(self._c(collection), doc_id)


class ScopedStructuredStore(StructuredStoreBase):
    """Structured store proxy that transparently prefixes every collection name."""

    def __init__(self, inner: StructuredStoreBase, scope: AppScope) -> None:
        super().__init__(scope)
        self._inner = inner

    def _scoped_schema(self, schema: CollectionSchema) -> CollectionSchema:
        return CollectionSchema(
            name=self._c(schema.name),
            description=schema.description,
            primary_fields=schema.primary_fields,
            fields=schema.fields,
        )

    def register_schema(self, schema: CollectionSchema) -> None:
        self._schemas[schema.name] = schema
        self._inner.register_schema(self._scoped_schema(schema))

    async def create_collection(self, schema: CollectionSchema) -> None:
        self._schemas[schema.name] = schema
        await self._inner.create_collection(self._scoped_schema(schema))

    async def _save(self, collection: str, records: list[dict]) -> None:
        await self._inner._save(self._c(collection), records)

    async def query(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        return await self._inner.query(self._c(collection), filters, fields)

    async def update_collection(self, schema: CollectionSchema) -> None:
        self._schemas[schema.name] = schema
        await self._inner.update_collection(self._scoped_schema(schema))

    async def delete_collection(self, collection: str) -> None:
        self._schemas.pop(collection, None)
        await self._inner.delete_collection(self._c(collection))

    async def delete_records(
        self, collection: str, filters: list[Filter] | None = None
    ) -> None:
        await self._inner.delete_records(self._c(collection), filters)


class ScopedDocumentStore(DocumentStoreBase):
    """Document store proxy that transparently prefixes every collection name."""

    def __init__(self, inner: DocumentStoreBase, scope: AppScope) -> None:
        super().__init__(scope)
        self._inner = inner

    async def save(self, collection: str, doc_id: str, content: str) -> None:
        await self._inner.save(self._c(collection), doc_id, content)

    async def load(self, collection: str, doc_id: str) -> str:
        return await self._inner.load(self._c(collection), doc_id)

    async def delete(self, collection: str, doc_id: str) -> None:
        await self._inner.delete(self._c(collection), doc_id)

    async def exists(self, collection: str, doc_id: str) -> bool:
        return await self._inner.exists(self._c(collection), doc_id)

    async def delete_collection(self, collection: str) -> None:
        await self._inner.delete_collection(self._c(collection))

    async def save_bytes(self, collection: str, doc_id: str, content: bytes) -> None:
        await self._inner.save_bytes(self._c(collection), doc_id, content)

    async def load_bytes(self, collection: str, doc_id: str) -> bytes:
        return await self._inner.load_bytes(self._c(collection), doc_id)
