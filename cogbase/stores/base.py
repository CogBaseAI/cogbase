"""Abstract adapter contracts for structured and vector stores.

Implement these to add a new backend. The ``filters`` dict on query methods
is treated as a set of equality matchers (e.g., ``{"type": "notice_period", "doc_id": "abc123"}``).
More complex filter semantics are adapter-defined.
"""

import abc

from cogbase.core.models import Chunk, Contradiction, Event, Fact


class StructuredStoreBase(abc.ABC):
    """Contract for any structured (relational/document) store backend."""

    @abc.abstractmethod
    def save_facts(self, facts: list[Fact]) -> None: ...

    @abc.abstractmethod
    def query_facts(self, filters: dict) -> list[Fact]: ...

    @abc.abstractmethod
    def save_timeline(self, events: list[Event]) -> None: ...

    @abc.abstractmethod
    def query_timeline(self, session_id: str) -> list[Event]: ...

    @abc.abstractmethod
    def save_contradiction(self, c: Contradiction) -> None: ...

    @abc.abstractmethod
    def query_contradictions(self, filters: dict) -> list[Contradiction]: ...


class VectorStoreBase(abc.ABC):
    """Contract for any vector store backend."""

    @abc.abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None: ...

    @abc.abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Chunk]: ...

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None: ...
