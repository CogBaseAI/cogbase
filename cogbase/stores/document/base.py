"""Abstract contract for object-based document stores."""

import abc


class DocumentStoreBase(abc.ABC):
    """Stores and retrieves full document text keyed by ``doc_id``.

    All methods are async.  Implementations that call blocking I/O must
    wrap it with ``asyncio.get_event_loop().run_in_executor``.

    Example::

        store = LocalFSDocumentStore("/var/cogbase/docs")
        await store.save("contract-001", full_text)
        text = await store.load("contract-001")
        await store.delete("contract-001")
    """

    @abc.abstractmethod
    async def save(self, doc_id: str, content: str) -> None:
        """Persist *content* for *doc_id*, overwriting any previous version."""

    @abc.abstractmethod
    async def load(self, doc_id: str) -> str:
        """Return the stored text for *doc_id*.

        Raises ``KeyError`` if the document does not exist.
        """

    @abc.abstractmethod
    async def delete(self, doc_id: str) -> None:
        """Delete the stored document.  No-op if it does not exist."""

    @abc.abstractmethod
    async def exists(self, doc_id: str) -> bool:
        """Return ``True`` if *doc_id* is present in the store."""
