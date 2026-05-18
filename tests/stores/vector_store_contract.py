"""Shared behavioral contract tests for all VectorStore implementations.

Import ``assert_non_core_fields_roundtrip`` into each vector store test module
and call it with a store that already has *collection* registered.
"""

from __future__ import annotations

from cogbase.core.models import Chunk
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
