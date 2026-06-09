"""Data models for the memory layer.

These types serve the short-term and episodic tiers and are shaped so the
planned long-term tier and a unifying ``MemoryManager`` can reuse them.  The
short-term :class:`SessionState` / :class:`MemoryMessage` are a *projection* of
the episodic :class:`MemoryEvent` log, not an independent store.  Everything
here is plain Pydantic, consistent with :mod:`cogbase.core.models` and
:mod:`cogbase.core.session`.
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
    """A single conversational turn projected from the episodic log.

    Short-term memory builds these by projecting the continuity events
    (``user_message`` / ``final_answer``) of a session's log; ``seq`` is the
    source event's per-session sequence number, retained so compaction can
    record the ``replaces_through`` watermark it covers.  Messages not sourced
    from a log event (e.g. the current turn's pending input) carry ``seq=None``.
    """

    role: MemoryRole
    content: str
    # Source event's per-session seq; None for messages not (yet) in the log.
    seq: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    # Rough token cost, filled in when projected so context assembly can budget
    # without re-estimating every message on every call.
    token_estimate: int = 0


class SessionState(BaseModel):
    """A projected view of one session's working context.

    Short-term memory rebuilds one of these per ``session_id`` by projecting the
    episodic log (see :mod:`cogbase.memory.short_term`).  It is intentionally not
    a source of truth — the log is — so it holds only what belongs in the next
    LLM call: the recent continuity thread plus the running compaction summary.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    app_id: str | None = None
    user_id: str | None = None
    # Explicit scope (session / user / app / project / org / global) per
    # docs/memory.md; carried so multi-tenant callers can isolate sessions.
    scope: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    messages: list[MemoryMessage] = Field(default_factory=list)
    # Running summary from the latest ``session_compacted`` event covering the
    # turns folded out of ``messages``.
    summary: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    last_active_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _utcnow()) >= self.expires_at


# ---------------------------------------------------------------------------
# Episodic memory: the append-only event log
#
# These types model the durable per-session event log (see
# docs/episodic-memory.md).  Unlike the short-term types above — which are a
# transient projection of this log — a :class:`MemoryEvent` is immutable once
# appended: the log only ever grows.  Short-term memory rehydrates its
# :class:`SessionState` from these events.
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """The episodic event types (docs/episodic-memory.md#event-payloads).

    The *continuity* tier (``USER_MESSAGE``, ``FINAL_ANSWER``,
    ``SESSION_COMPACTED``) is what short-term rehydrate reconstructs a thread
    from and must be durably flushed before a turn is acknowledged; the rest are
    best-effort observability that may be lost without corrupting continuity.
    """

    SESSION_STARTED = "session_started"
    USER_MESSAGE = "user_message"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    RETRIEVAL_RESULT = "retrieval_result"
    FINAL_ANSWER = "final_answer"
    FEEDBACK = "feedback"
    SESSION_COMPACTED = "session_compacted"


# Continuity-critical events: at-least-once durability required before turn-ack.
CONTINUITY_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.USER_MESSAGE, EventType.FINAL_ANSWER, EventType.SESSION_COMPACTED}
)


class EventRef(BaseModel):
    """A self-locating reference to another event — the identity *triplet*.

    Resolving a reference is a log seek, not an index lookup: locate the line by
    ``(session_id, seq)``, then verify ``ulid`` matches the line found.  A
    mismatch catches a ``seq`` reuse at read time (see
    docs/episodic-memory.md#event-identity).  Used for ``parent_event_id``,
    ``final_answer.cited_ids``, and ``feedback.target``.
    """

    session_id: str
    seq: int
    ulid: str


class MemoryEvent(BaseModel):
    """One immutable event in a session's append-only log.

    The envelope (everything but ``payload``/``metadata``) is fixed across event
    types; the per-type ``payload`` contracts live in the ``*Payload`` models
    below.  ``seq`` and ``ulid`` are the identity fields stamped by
    :class:`~cogbase.memory.episodic.EpisodicMemory` at ``record`` time — they
    default to the unstamped sentinels (``seq=-1``, ``ulid=""``) on a
    freshly-built event and are assigned by the single writer for the session.

    - ``seq`` — per-session monotonic integer; the *authority* for ordering and
      gap detection.
    - ``ulid`` — globally-unique, time-sortable; the idempotency/dedupe key for
      retried appends and an independent witness for ``seq``.
    """

    session_id: str
    seq: int = -1
    ulid: str = ""
    event_type: EventType
    created_at: datetime = Field(default_factory=_utcnow)
    app_id: str | None = None
    user_id: str | None = None
    # Causal link to a prior event in the same session (e.g. tool_result →
    # tool_called); stored as the full triplet so it resolves by log seek.
    parent_event_id: EventRef | None = None
    payload: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @property
    def is_stamped(self) -> bool:
        """True once the writer has assigned ``seq`` and ``ulid``."""
        return self.seq >= 0 and bool(self.ulid)

    @property
    def ref(self) -> EventRef:
        """The identity triplet for threading into references."""
        return EventRef(session_id=self.session_id, seq=self.seq, ulid=self.ulid)

    def to_ndjson(self) -> str:
        """Serialize to a single NDJSON line (no trailing newline; the log store
        owns framing)."""
        return self.model_dump_json()

    @classmethod
    def from_ndjson(cls, line: str) -> "MemoryEvent":
        return cls.model_validate_json(line)


# -- Per-type payload contracts (docs/episodic-memory.md#event-payloads) -----
#
# Payloads are deliberately minimal with an open ``metadata`` dict for
# extension; the envelope's ``metadata`` carries cross-cutting extension instead.
# These models exist to make the contract explicit and validated at construction
# time; they are dumped to plain dicts into ``MemoryEvent.payload``.


class SessionStartedPayload(BaseModel):
    # client / app-config-version and similar start-of-session context
    metadata: dict = Field(default_factory=dict)


class UserMessagePayload(BaseModel):
    text: str
    attachments: list[dict] = Field(default_factory=list)


class FinalAnswerPayload(BaseModel):
    text: str
    cited_ids: list[EventRef] = Field(default_factory=list)


class SessionCompactedPayload(BaseModel):
    summary: str
    # last ``seq`` the summary covers; rehydrate loads this summary plus every
    # event after ``replaces_through``.
    replaces_through: int
    token_stats: dict = Field(default_factory=dict)


class ToolCalledPayload(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolResultPayload(BaseModel):
    tool_call_id: str
    ok: bool
    # exactly one of result / error is meaningful, keyed by ``ok``
    result: object | None = None
    error: str | None = None
    latency_ms: float | None = None


class RetrievalHit(BaseModel):
    id: str
    score: float | None = None


class RetrievalResultPayload(BaseModel):
    collection: str
    query: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    top_k: int | None = None


class FeedbackPayload(BaseModel):
    target: EventRef
    rating: str
    comment: str | None = None
