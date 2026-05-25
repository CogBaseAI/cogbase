"""Shared data primitives used across all layers of CogBase."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class DocWorkflowStatus(str, Enum):
    READY   = "ready"    # manual trigger — applicable but no task created yet
    PENDING = "pending"  # task queued (after_ingest trigger)
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"

_CHUNK_CORE_FIELDS = frozenset({"chunk_id", "doc_id", "text", "embedding", "metadata"})


class Document(BaseModel):
    """A raw document to be ingested into CogBase.

    ``doc_id`` is the stable identifier used to correlate chunks, extracted
    facts, and vector embeddings back to their source.  ``metadata`` is an
    open bag of string key-value pairs for caller-defined attributes (e.g.
    filename, source URL, author).
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk from a document, optionally carrying its embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str # doc_id_{chunk_index} such as doc_id_0, doc_id_1
    doc_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)
    # char_offset and char_length are optional. They will be set for document chunks.
    # Chunk could be used by other cases, such as document summary (one chunk per
    # document), where char offset and length are not meaningful.
    char_offset: int | None = None  # start character position in the source document
    char_length: int | None = None  # number of characters in this chunk

    def to_storable_metadata(self) -> dict:
        """Return metadata dict with extra (non-core) fields merged in for storage.

        Stores that have a fixed column schema plus a JSONB/dict metadata bag
        should call this instead of ``self.metadata`` so that any future fields
        added to ``Chunk`` automatically round-trip without store changes.
        """
        extras = {
            k: getattr(self, k)
            for k in type(self).model_fields
            if k not in _CHUNK_CORE_FIELDS and getattr(self, k) is not None
        }
        return {**self.metadata, **extras}

    @classmethod
    def from_stored(cls, *, metadata: dict, **core_fields) -> "Chunk":
        """Reconstruct a ``Chunk`` from stored core fields and a metadata dict.

        Pops any extra (non-core) ``Chunk`` fields back out of ``metadata``
        so callers don't need to know which fields were spilled.
        """
        metadata = dict(metadata)
        extras = {
            k: metadata.pop(k)
            for k in cls.model_fields
            if k not in _CHUNK_CORE_FIELDS and k in metadata
        }
        return cls(**core_fields, metadata=metadata, **extras)
