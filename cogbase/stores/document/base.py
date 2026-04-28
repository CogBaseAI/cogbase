"""Abstract contract for object-based document stores."""

import abc


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
    async def exists(self, collection: str, doc_id: str) -> bool:
        """Return ``True`` if *doc_id* is present in *collection*."""
