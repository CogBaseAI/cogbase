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
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType


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


# ---------------------------------------------------------------------------
# Long-term memory: curated, durable cross-session knowledge
#
# These model the records distilled out of session logs (see
# docs/long-term-memory.md).  Unlike the episodic log (append-only) and the
# short-term projection (transient), long-term records are *mutable*:
# reconciliation reinforces, revises, or supersedes them in place so the store
# stays curated rather than append-only.  They live physically in the system
# stores but are partitioned per app for RBAC (the ``app_id`` partition key,
# realized by the scoped store layout — see
# docs/long-term-memory.md#the-app-is-the-partition-boundary).
# ---------------------------------------------------------------------------


def normalize_entities(values: list[str]) -> list[str]:
    """Normalize entity mentions: lowercase, strip, drop empties, dedupe.

    Entities are an *index* over claims, not records of their own — matching is
    exact on the normalized form, so both write (distill) and read (lookup,
    reconcile) paths must normalize through this one function.
    """
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        norm = v.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


class MemoryKind(str, Enum):
    """What a long-term record is about (docs/long-term-memory.md#record-shape)."""

    PREFERENCE = "preference"
    FACT = "fact"
    CORRECTION = "correction"
    RETRIEVAL_HINT = "retrieval_hint"


class MemoryStatus(str, Enum):
    """Lifecycle state of a long-term record.

    Only ``active`` records are recalled into the query path; ``pending_review``
    gates behaviour-affecting kinds until a reviewer accepts them, and
    ``superseded`` marks records retracted by a contradicting observation.
    """

    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    SUPERSEDED = "superseded"


class ReconcileOp(str, Enum):
    """The operation reconciliation applies a candidate against belief with.

    The crux of long-term memory and the one step with no analog in the
    document pipeline (docs/long-term-memory.md#pipeline): a new observation is
    merged against accumulated belief rather than upserted by primary key.
    """

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class LongTermRecord(BaseModel):
    """One curated, durable memory promoted out of session logs.

    The record is *self-contained*: ``evidence_snapshot`` copies the deciding
    cited text so the record stays valid after the source session log expires,
    while ``source_event_ids`` keep the audit trail into the log while it lives
    (docs/long-term-memory.md#pipeline, step 5).
    """

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    # Partition key — the outer RBAC fence.  Physically the records live in the
    # shared system stores; isolation is realized by the scoped store layout
    # (each app addresses its own prefixed collections), so this field is for
    # self-containment / audit, not the enforcement path.  Named ``app_id`` to
    # match the rest of the memory layer (the stable internal id, not the
    # mutable client-facing name).
    app_id: str | None = None
    kind: MemoryKind = MemoryKind.FACT
    content: str = ""
    # Normalized entity mentions (people, projects, systems) the claim is about.
    # An index over claims for exact-match lookup and reconcile candidate
    # retrieval — not entity records; the claim stays the unit of memory.
    entities: list[str] = Field(default_factory=list)
    # 0..1; reinforced on repeat observation, weighed in reconciliation.
    confidence: float = 0.5
    status: MemoryStatus = MemoryStatus.ACTIVE
    # Directed edges to other records this memory relates to (same entity/topic, a
    # follow-up event, an update, or a contradiction).  A lightweight memory graph
    # over the curated store — recall expands the neighborhood of its hits along
    # these edges (see docs/long-term-memory.md).  Stored as ``memory_id`` strings;
    # edges to a superseded/missing record are simply skipped on traversal, so no
    # referential integrity is enforced.
    linked_memory_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[EventRef] = Field(default_factory=list)
    evidence_snapshot: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _utcnow()) >= self.expires_at

    @classmethod
    def collection_schema(cls, collection_name: str) -> CollectionSchema:
        """The structured-store schema for this record, under *collection_name*.

        Co-located with the model so the persisted schema and the fields above
        stay in lockstep: every field a row carries is declared here.  Scalar
        attributes are stored as strings (enum values and ISO timestamps are
        dumped as text); ``confidence`` is the one numeric column; the two
        provenance attributes are nested structures stored as JSON.
        """
        s = lambda **kw: FieldSchema(type=FieldType.STRING, **kw)  # noqa: E731
        return CollectionSchema(
            name=collection_name,
            description=(
                "Curated long-term memory records (facts, preferences, "
                "corrections, retrieval hints) with provenance and lifecycle."
            ),
            primary_fields=["memory_id"],
            fields={
                "memory_id": s(nullable=False, index=True),
                "app_id": s(),
                "kind": s(index=True),
                "content": s(),
                "entities": FieldSchema(type=FieldType.JSON),
                "confidence": FieldSchema(type=FieldType.FLOAT, index=True),
                "status": s(index=True),
                "linked_memory_ids": FieldSchema(type=FieldType.JSON),
                "source_event_ids": FieldSchema(type=FieldType.JSON),
                "evidence_snapshot": FieldSchema(type=FieldType.JSON),
                "created_at": s(index=True),
                "updated_at": s(index=True),
                "expires_at": s(index=True),
            },
        )

    # The record fields projected into the content vector's metadata so recall
    # and reconcile can filter on them.  Kept adjacent to :meth:`vector_metadata`
    # so the indexed field list and the written values stay in lockstep, and
    # used to build the vector collection's ``metadata_fields``.
    VECTOR_METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "kind", "status", "entities",
    )

    def vector_metadata(self) -> dict:
        """The subset of fields indexed on this record's content vector.

        Enum fields are dumped to their values to match how the structured row
        stores them.  The keys are exactly :attr:`VECTOR_METADATA_FIELDS`.
        """
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "entities": list(self.entities),
        }


class MemoryCandidate(BaseModel):
    """A distiller's proposed memory, before reconciliation decides its fate.

    Carries everything reconciliation needs to ADD / UPDATE / DELETE / NOOP it
    against accumulated belief, including the provenance to snapshot on
    promotion.  ``confidence`` is required (0..1): the distiller scores every
    candidate, and it also gates promotion — a record auto-activates only when
    its score clears the kind's auto-promote threshold (see
    :data:`cogbase.memory.long_term.AUTO_PROMOTE_CONFIDENCE`), otherwise it waits
    for review.
    """

    content: str
    kind: MemoryKind = MemoryKind.FACT
    # Normalized entity mentions the claim is about (see LongTermRecord.entities).
    entities: list[str] = Field(default_factory=list)
    # Resolved ``memory_id``s of existing records this candidate relates to, lifted
    # from the extractor's links against the existing-memory context and carried
    # onto the promoted record (see LongTermRecord.linked_memory_ids).
    linked_memory_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[EventRef] = Field(default_factory=list)
    evidence_snapshot: dict = Field(default_factory=dict)
    # 0..1 — how strongly the source supports the claim; weighed in reconciliation.
    confidence: float


# ---------------------------------------------------------------------------
# Promotion review: the pending_review -> active gate for behaviour-affecting
# kinds (docs/long-term-memory.md, build order step 6).  A reviewer clears the
# pending queue in batches; these model one decision and its applied outcome.
# ---------------------------------------------------------------------------


class ReviewDecision(BaseModel):
    """A reviewer's verdict on one gated record: accept (-> active) or reject."""

    memory_id: str
    # True accepts (promotes to ``active``); False rejects (marks ``superseded``).
    accept: bool


class ReviewOutcome(str, Enum):
    """What :meth:`LongTermMemory.review` did with one decision.

    ``accepted`` / ``rejected`` are the two terminal transitions; ``skipped``
    and ``not_found`` make the batch path honest about decisions that did not
    apply (a record already decided, or one that no longer exists) instead of
    pretending the whole batch is atomic — there is no cross-record transaction
    across the structured + vector stores anyway.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    # The record exists but is not pending_review (already accepted/rejected) —
    # the guard that keeps review idempotent and can't resurrect a superseded one.
    SKIPPED = "skipped"
    NOT_FOUND = "not_found"


class ReviewResult(BaseModel):
    """The applied outcome for one :class:`ReviewDecision` in a batch."""

    memory_id: str
    outcome: ReviewOutcome
