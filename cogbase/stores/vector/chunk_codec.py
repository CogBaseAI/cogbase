"""Shared encode/decode/projection logic for all vector store implementations.

Every store maps a Chunk to five physical columns plus a metadata bag:

    chunk_id  TEXT PRIMARY KEY
    doc_id    TEXT NOT NULL
    text      TEXT NOT NULL
    embedding vector / float[]
    metadata  JSONB / dict   ← carries non-core Chunk fields automatically

Stores call ``to_store_record`` on write and ``from_store_record`` on read.
This ensures that any field added to Chunk round-trips through all stores
without per-store changes — unless the field needs indexed querying, in which
case it is promoted to a real column explicitly.
"""

from __future__ import annotations

from cogbase.core.models import Chunk, _CHUNK_CORE_FIELDS

# Columns with dedicated physical storage that support direct SQL / field filtering.
# Fields outside this set are stored in the metadata bag via to_storable_metadata().
FILTERABLE_COLUMNS: frozenset[str] = frozenset({"chunk_id", "doc_id", "text"})


def to_store_record(chunk: Chunk) -> tuple[str, str, str, list[float] | None, dict]:
    """Serialize *chunk* to the five physical store columns.

    The returned metadata dict already contains any non-core Chunk fields
    (e.g. ``char_offset``, ``char_length``) merged in via
    ``Chunk.to_storable_metadata()``.  Stores are responsible for any
    backend-specific type conversion (numpy arrays, JSON strings, etc.).
    """
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.text,
        chunk.embedding,
        chunk.to_storable_metadata(),
    )


def from_store_record(
    chunk_id: str,
    doc_id: str,
    text: str,
    embedding: list[float] | None,
    metadata: dict,
) -> Chunk:
    """Reconstruct a ``Chunk`` from the five physical store columns."""
    return Chunk.from_stored(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        embedding=embedding,
        metadata=metadata,
    )


def project_chunk(chunk: Chunk, fields: list[str] | None) -> Chunk:
    """Return a copy of *chunk* with only the requested *fields* populated.

    Core fields ``chunk_id``, ``doc_id``, and ``text`` are always included.
    ``embedding`` defaults to ``None`` and ``metadata`` defaults to ``{}``
    when not requested.  Any future non-core Chunk field is forwarded if it
    appears in *fields*, keeping projection correct without manual updates.
    """
    if not fields:
        return chunk
    field_set = set(fields)
    data: dict = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "embedding": chunk.embedding if "embedding" in field_set else None,
        "metadata": chunk.metadata if "metadata" in field_set else {},
    }
    for k in Chunk.model_fields:
        if k not in _CHUNK_CORE_FIELDS and k in field_set:
            data[k] = getattr(chunk, k)
    return Chunk(**data)
