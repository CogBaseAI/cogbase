"""Shared behavioral contract tests for all VectorStore implementations.

Import ``assert_non_core_fields_roundtrip`` into each vector store test module
and call it with a store that already has *collection* registered.
"""

from __future__ import annotations

from cogbase.core.models import Chunk
from cogbase.stores.filters import Col
from cogbase.stores.vector.base import VectorStoreBase


async def assert_non_core_fields_roundtrip(
    store: VectorStoreBase,
    collection: str,
    dim: int,
) -> None:
    """Non-core Chunk fields survive upsert → search without store changes.

    This is the guardrail: adding a field to Chunk must not require touching
    any store implementation for it to round-trip correctly.
    """
    embedding = [0.1] * dim
    chunk = Chunk(
        chunk_id="contract_c1",
        doc_id="contract_d1",
        text="hello",
        embedding=embedding,
        char_offset=10,
        char_length=5,
    )
    await store.upsert(collection, [chunk])
    result = (await store.search(collection, "hello", embedding, top_k=1))[0]
    assert result.char_offset == 10
    assert result.char_length == 5


async def assert_non_core_fields_none_roundtrip(
    store: VectorStoreBase,
    collection: str,
    dim: int,
) -> None:
    """Non-core Chunk fields that are None survive upsert → search as None."""
    embedding = [0.1] * dim
    chunk = Chunk(
        chunk_id="contract_c2",
        doc_id="contract_d2",
        text="hello",
        embedding=embedding,
    )
    await store.upsert(collection, [chunk])
    result = (await store.search(collection, "hello", embedding, top_k=1))[0]
    assert result.char_offset is None
    assert result.char_length is None


async def assert_query_enumerates(
    store: VectorStoreBase,
    collection: str,
    dim: int,
) -> None:
    """``query`` returns every match, in ``chunk_id`` order, without a query vector.

    The contract that makes the method worth having, and it has to hold identically
    on every backend that implements it: a caller enumerating through ``search``
    guesses a ``top_k`` and gets silent distance-ordered truncation when the guess is
    low, which is the failure ``query`` exists to remove. A backend where ``query``
    quietly capped results would put that failure straight back.
    """
    embedding = [0.1] * dim
    chunks = [
        Chunk(
            chunk_id=f"enum_c{i:02d}",
            doc_id="enum_d1" if i % 2 == 0 else "enum_d2",
            text=f"row {i}",
            embedding=embedding,
            metadata={"parity": "even" if i % 2 == 0 else "odd"},
        )
        for i in range(30)
    ]
    await store.upsert(collection, chunks)

    everything = await store.query(collection, [Col("chunk_id").like("enum_c%")])
    assert [c.chunk_id for c in everything] == [c.chunk_id for c in chunks]

    scoped = await store.query(collection, [Col("doc_id") == "enum_d1"])
    assert {c.chunk_id for c in scoped} == {c.chunk_id for c in chunks if c.doc_id == "enum_d1"}

    by_metadata = await store.query(collection, [Col("metadata.parity") == "odd"])
    assert {c.chunk_id for c in by_metadata} == {
        c.chunk_id for c in chunks if c.metadata["parity"] == "odd"
    }

    limited = await store.query(collection, [Col("chunk_id").like("enum_c%")], limit=5)
    assert [c.chunk_id for c in limited] == [c.chunk_id for c in chunks[:5]]

    projected = await store.query(
        collection, [Col("chunk_id") == "enum_c00"], fields=["chunk_id", "doc_id", "text"]
    )
    assert projected[0].embedding is None

    assert await store.query(collection, [Col("doc_id") == "enum_absent"]) == []
