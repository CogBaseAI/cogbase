import pytest
from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.stores import AppScope, CollectionSchema, Filter, StructuredStoreBase, VectorStoreBase


def test_structured_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        StructuredStoreBase()  # type: ignore[abstract]


def test_vector_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VectorStoreBase()  # type: ignore[abstract]


def test_incomplete_structured_subclass_raises():
    class Partial(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def save(self, collection: str, records: list[BaseModel]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: return []
        # delete_records intentionally missing

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_complete_structured_subclass_ok():
    class Minimal(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def delete_collection(self, collection: str) -> None: ...
        async def _save(self, collection: str, records: list[dict]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: return []
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    assert Minimal() is not None


def test_structured_store_scope_stored():
    class Minimal(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def delete_collection(self, collection: str) -> None: ...
        async def _save(self, collection: str, records: list[dict]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: return []
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    scope = AppScope(app_id="myapp")
    store = Minimal(scope=scope)
    assert store._scope is scope
    assert store._c("col") == "myapp__col"


def test_structured_store_no_scope_bare_name():
    class Minimal(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def delete_collection(self, collection: str) -> None: ...
        async def _save(self, collection: str, records: list[dict]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]: return []
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    store = Minimal()
    assert store._c("col") == "col"


async def test_query_as_uses_query():
    class Stub(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def delete_collection(self, collection: str) -> None: ...
        async def _save(self, collection: str, records: list[dict]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None, fields: list[str] | None = None) -> list[dict]:
            return [{"x": 1}]
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    class M(BaseModel):
        x: int

    assert await Stub().query_as("col", None, M) == [M(x=1)]


async def test_vector_store_query_is_optional_and_says_so():
    """``query`` is concrete-with-a-raise rather than abstract on purpose.

    Most backends have filter-only reads natively (pgvector is SQL, Qdrant ``scroll``,
    Milvus ``query``, Chroma ``get(where=)``), but some are vector-first and genuinely
    do not — Pinecone's ``query`` requires a vector and offers only ``fetch`` by id and
    ``list`` by id prefix. Making it abstract would force such a backend to emulate it
    with a zero-vector query and a guessed ``top_k``, which is the silent truncation
    the method exists to remove. So the default refuses, loudly, and existing
    out-of-tree implementations keep working.
    """
    class VectorFirst(VectorStoreBase):
        async def create_collection(self, schema) -> None: ...
        async def upsert(self, collection: str, chunks: list[Chunk]) -> None: ...
        async def search(self, collection, query, query_embedding, top_k, filters=None, fields=None): return []
        async def delete_collection(self, collection: str) -> None: ...
        async def delete(self, collection: str, chunk_ids: list[str]) -> None: ...
        async def delete_doc(self, collection: str, doc_id: str) -> None: ...

    store = VectorFirst()
    assert store is not None, "a store without query must still be instantiable"
    with pytest.raises(NotImplementedError, match="without a query vector"):
        await store.query("col")
