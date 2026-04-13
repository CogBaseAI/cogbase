"""Shared data primitives used across all layers of CogBase."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Fact(BaseModel):
    """An extracted, typed fact from a source document.

    ``raw_text`` is preserved verbatim as the citation string.
    ``confidence`` is in [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    value: str
    raw_text: str
    doc_id: str
    page: int | None = None
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


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

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class Event(BaseModel):
    """A timeline event recording an action taken within a session."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str
    action: str
    payload: dict = Field(default_factory=dict)


class Contradiction(BaseModel):
    """A detected conflict between two facts.

    ``conflict_type`` is a plain string so domain packs can define their own types.
    The built-in conventions are: ``"date"``, ``"numeric"``, ``"statement"``.
    """

    model_config = ConfigDict(frozen=True)

    contradiction_id: str = Field(default_factory=lambda: str(uuid4()))
    fact_a: Fact
    fact_b: Fact
    conflict_type: str
    resolved: bool = False
    resolution_note: str | None = None
