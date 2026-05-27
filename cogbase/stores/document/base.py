"""Abstract contract for object-based document stores."""

from __future__ import annotations

import abc

from cogbase.stores.scope import AppScope


class DocumentStoreBase(abc.ABC):
    """Stores and retrieves full document text keyed by ``collection`` + ``doc_id``.

    The *collection* parameter lets multiple applications share the same store
    instance without key collisions — each application uses its own collection
    name (typically the application name).

    All methods are async.  Implementations that call blocking I/O must
    wrap it with ``asyncio.get_event_loop().run_in_executor``.

    Example::

        store = LocalFSDocumentStore("/var/cogbase/docs")
        await store.save("legal", "contract-001", full_text)
        text = await store.load("legal", "contract-001")
        await store.delete("legal", "contract-001")
    """

    def __init__(self, scope: AppScope | None = None) -> None:
        self._scope = scope

    def _c(self, collection: str) -> str:
        """Return the backend-internal name for *collection* (bare name → scoped name)."""
        prefix = self._scope.prefix() if self._scope else None
        return f"{prefix}__{collection}" if prefix else collection

    def with_scope(self, scope: AppScope) -> "DocumentStoreBase":
        """Return a scoped proxy that prefixes all collection names with *scope*."""
        from cogbase.stores.scoped import ScopedDocumentStore
        return ScopedDocumentStore(self, scope)

    @abc.abstractmethod
    async def save(self, collection: str, doc_id: str, content: str) -> None:
        """Persist *content* for *doc_id* in *collection*, overwriting any previous version."""

    @abc.abstractmethod
    async def load(self, collection: str, doc_id: str) -> str:
        """Return the stored text for *doc_id* in *collection*.

        Raises ``KeyError`` if the document does not exist.
        """

    @abc.abstractmethod
    async def delete(self, collection: str, doc_id: str) -> None:
        """Delete the stored document.  No-op if it does not exist."""

    @abc.abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Delete all documents in *collection*.  No-op if the collection does not exist."""

    @abc.abstractmethod
    async def exists(self, collection: str, doc_id: str) -> bool:
        """Return ``True`` if *doc_id* is present in *collection*."""

    async def save_bytes(self, collection: str, doc_id: str, content: bytes) -> None:
        """Persist raw *content* bytes for *doc_id* in *collection*.

        Subclasses should override for efficient binary storage.  The default
        implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support save_bytes")

    async def load_bytes(self, collection: str, doc_id: str) -> bytes:
        """Return the stored bytes for *doc_id* in *collection*.

        Raises ``KeyError`` if the document does not exist.  Subclasses should
        override for efficient binary storage.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support load_bytes")
