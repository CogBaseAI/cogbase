import pytest
from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.filters import Filter
from cogbase.stores.schema import CollectionSchema


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
        async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]: return []
        # delete_records intentionally missing

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_complete_structured_subclass_ok():
    class Minimal(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def save(self, collection: str, records: list[BaseModel]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]: return []
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    assert Minimal() is not None


async def test_query_as_uses_query():
    class Stub(StructuredStoreBase):
        async def create_collection(self, schema: CollectionSchema) -> None: ...
        async def update_collection(self, schema: CollectionSchema) -> None: ...
        async def save(self, collection: str, records: list[BaseModel]) -> None: ...
        async def query(self, collection: str, filters: list[Filter] | None = None) -> list[dict]:
            return [{"x": 1}]
        async def delete_records(self, collection: str, filters: list[Filter] | None = None) -> None: ...

    class M(BaseModel):
        x: int

    assert await Stub().query_as("col", None, M) == [M(x=1)]
