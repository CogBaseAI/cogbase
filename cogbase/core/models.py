"""Shared data primitives used across all layers of CogBase."""

from pydantic import BaseModel, ConfigDict, Field


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
    char_offset: int | None = None  # start character position in the source document
    char_length: int | None = None  # number of characters in this chunk
