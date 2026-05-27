from cogbase.stores.document.base import DocumentStoreBase
from cogbase.stores.scope import AppScope


class InMemoryDocumentStore(DocumentStoreBase):
    def __init__(self, scope: AppScope | None = None) -> None:
        super().__init__(scope)
        self._store: dict[tuple[str, str], str] = {}

    async def save(self, collection: str, doc_id: str, content: str) -> None:
        self._store[(self._c(collection), doc_id)] = content

    async def load(self, collection: str, doc_id: str) -> str:
        try:
            return self._store[(self._c(collection), doc_id)]
        except KeyError:
            raise KeyError(f"{doc_id!r} not found in collection {collection!r}")

    async def delete(self, collection: str, doc_id: str) -> None:
        self._store.pop((self._c(collection), doc_id), None)

    async def delete_collection(self, collection: str) -> None:
        scoped = self._c(collection)
        for key in [k for k in self._store if k[0] == scoped]:
            del self._store[key]

    async def exists(self, collection: str, doc_id: str) -> bool:
        return (self._c(collection), doc_id) in self._store
