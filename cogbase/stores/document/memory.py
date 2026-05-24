from cogbase.stores.document.base import DocumentStoreBase


class InMemoryDocumentStore(DocumentStoreBase):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    async def save(self, collection: str, doc_id: str, content: str) -> None:
        self._store[(collection, doc_id)] = content

    async def load(self, collection: str, doc_id: str) -> str:
        try:
            return self._store[(collection, doc_id)]
        except KeyError:
            raise KeyError(f"{doc_id!r} not found in collection {collection!r}")

    async def delete(self, collection: str, doc_id: str) -> None:
        self._store.pop((collection, doc_id), None)

    async def exists(self, collection: str, doc_id: str) -> bool:
        return (collection, doc_id) in self._store
