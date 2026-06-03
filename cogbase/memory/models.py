"""Data models for the memory layer.

These types are shared by the short-term tier today and are shaped so the
planned episodic/long-term tiers and a unifying ``MemoryManager`` can reuse
them.  Everything here is plain Pydantic, consistent with
:mod:`cogbase.core.models` and :mod:`cogbase.core.session`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRole(str, Enum):
    """Role of a turn stored in short-term memory."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MemoryMessage(BaseModel):
    """A single conversational turn held in a session's working context."""

    role: MemoryRole
    content: str
    created_at: datetime = Field(default_factory=_utcnow)
    # Rough token cost, filled in when appended so context assembly can budget
    # without re-estimating every message on every call.
    token_estimate: int = 0


class RetrievalKind(str, Enum):
    """What a :class:`RetrievedItem` refers to."""

    CHUNK = "chunk"     # a passage from vector_search
    RECORD = "record"   # a structured_lookup record
    SLICE = "slice"     # a read_document slice


class RetrievedItem(BaseModel):
    """A piece of evidence retrieved during a session.

    Kept separately from messages so the session can reason about (and later
    re-rank or expire) retrieved evidence without parsing the transcript.
    """

    kind: RetrievalKind
    ref_id: str | None = None      # chunk_id / doc_id / primary key, when available
    text: str = ""
    score: float | None = None
    source: str | None = None      # the tool that produced it, e.g. "vector_search"
    created_at: datetime = Field(default_factory=_utcnow)


class SessionState(BaseModel):
    """The full working context for one active session.

    Short-term memory owns one of these per ``session_id``.  It is intentionally
    not a source of truth — it decides what belongs in the next LLM call and is
    allowed to drop or compact its own contents.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    app_name: str | None = None
    user_id: str | None = None
    # Explicit scope (session / user / app / project / org / global) per
    # docs/memory.md; carried so multi-tenant callers can isolate sessions.
    scope: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    messages: list[MemoryMessage] = Field(default_factory=list)
    retrievals: list[RetrievedItem] = Field(default_factory=list)
    # Compacted summary of turns that no longer fit the raw transcript.
    summary: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    last_active_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _utcnow()) >= self.expires_at
