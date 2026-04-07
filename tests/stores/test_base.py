import pytest

from cogbase.core.models import Chunk, Contradiction, Event, Fact
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase


def test_structured_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        StructuredStoreBase()  # type: ignore[abstract]


def test_vector_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VectorStoreBase()  # type: ignore[abstract]


def test_incomplete_structured_subclass_raises():
    class Partial(StructuredStoreBase):
        def save_facts(self, facts: list[Fact]) -> None: ...
        def query_facts(self, filters: dict) -> list[Fact]: return []
        def save_timeline(self, events: list[Event]) -> None: ...
        def query_timeline(self, session_id: str) -> list[Event]: return []
        # save_contradiction and query_contradictions intentionally missing

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_complete_structured_subclass_ok():
    class InMemory(StructuredStoreBase):
        def save_facts(self, facts: list[Fact]) -> None: ...
        def query_facts(self, filters: dict) -> list[Fact]: return []
        def save_timeline(self, events: list[Event]) -> None: ...
        def query_timeline(self, session_id: str) -> list[Event]: return []
        def save_contradiction(self, c: Contradiction) -> None: ...
        def query_contradictions(self, filters: dict) -> list[Contradiction]: return []

    store = InMemory()
    assert store is not None
