"""Episodic memory: the durable append-only event writer.

``EpisodicMemory`` is the thin writer over :class:`~cogbase.stores.log.base.LogStoreBase`
that backs the per-session event log (see ``docs/episodic-memory.md``).  Unlike a
cache it never compacts or evicts — it only appends.

Two design points from the doc shape the implementation:

- **Appends are batched per turn, not per event.**  Each ``record_*`` stamps an
  event with its ``seq`` + ``ulid`` and buffers it in the session's in-memory
  cache; the buffer is flushed to the log as *one* multi-line append at the turn
  boundary via :meth:`flush`.  A turn therefore costs one store write, and the
  buffer doubles as the retry buffer until the append lands.
- **Identity is assigned by the single writer.**  ``seq`` is a per-session
  monotonic integer (the ordering authority + gap-detection signal); ``ulid`` is
  a globally-unique, time-sortable idempotency key and an independent witness for
  ``seq``.  On a cold start the next ``seq`` is recovered from the log tail, so a
  new process that takes over a session continues the sequence without
  duplicating it.
- **The append is fenced on the log's byte offset.**  Recovering ``seq`` from the
  tail does not by itself stop two cold-starting owners from both stamping the
  same ``seq`` — they read the same tail.  So the writer also tracks the log's
  byte length and passes it as the store's ``expected_offset``: the first owner's
  flush advances the object, the second's offset goes stale, and its
  compare-and-append is rejected with :class:`LogFenced` before the colliding
  ``seq`` is ever persisted.  A fenced writer is *deposed* — :meth:`flush`
  relinquishes the session rather than retrying (retrying is the bug).  The
  offset, not affinity, is the correctness mechanism (see
  ``docs/episodic-memory.md`` — single-writer and append safety).

Read-back (:meth:`replay`, :meth:`tail`) dedupes by ``ulid`` so a retried append
that double-wrote a line surfaces once.  Quarantining a ``seq``-colliding
out-of-order straggler is left to short-term rehydrate (build-order step 5),
which is the consumer that threads events into a conversation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from ulid import ULID

from cogbase.memory.models import (
    CONTINUITY_EVENT_TYPES,
    EventRef,
    EventType,
    FeedbackPayload,
    FinalAnswerPayload,
    MemoryEvent,
    RetrievalHit,
    RetrievalResultPayload,
    SessionCompactedPayload,
    SessionDistilledPayload,
    SessionStartedPayload,
    ToolCalledPayload,
    ToolResultPayload,
    UserMessagePayload,
)
from cogbase.stores.log.base import LogFenced, LogStoreBase

logger = logging.getLogger(__name__)

# The log family this writer appends to within the (shared) log store.
DEFAULT_LOG_TYPE = "episodic"


class EpisodicMemory:
    """Append-only writer over a :class:`LogStoreBase`.

    Args:
        log_store: The append-only log backing every session's event stream.
        log_type:  Log family namespace within the store (default ``"episodic"``).
    """

    def __init__(
        self, log_store: LogStoreBase, *, log_type: str = DEFAULT_LOG_TYPE
    ) -> None:
        self._log = log_store
        self._log_type = log_type
        # Per-session in-memory state.  All of it is process-local: it is rebuilt
        # from the log on a cold start, so losing it costs at most a re-read.
        self._buffers: dict[str, list[MemoryEvent]] = {}   # unflushed events, in seq order
        self._next_seq: dict[str, int] = {}                # next seq to assign
        self._expected_offset: dict[str, int] = {}         # log byte length this writer last observed
        self._app_ids: dict[str, str | None] = {}          # app_id per session
        # One lock per session keeps cross-session appends concurrent while
        # serializing a session's own record/flush (it has a single writer).
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------------
    # Recording (buffer + stamp)
    # ------------------------------------------------------------------

    def bind_app(
        self,
        session_id: str,
        *,
        app_id: str | None = None,
    ) -> None:
        """Register a session's app attribution without emitting an event.

        Later recorded events inherit ``app_id`` so callers need not re-pass it
        on every record.  Idempotent and process-local (rebuilt on a cold
        start), so it is safe — and expected — to call once per turn.  Use this
        where ``record_session_started`` would be wrong: the per-turn query
        runner does not own session creation and must not log a
        ``session_started`` event on every turn.
        """
        self._app_ids[session_id] = app_id

    async def record(self, event: MemoryEvent) -> EventRef:
        """Stamp *event* with its ``seq`` + ``ulid`` and buffer it for the next flush.

        Returns the event's identity triplet so callers can thread it into a
        ``parent_event_id`` / reference.  Does not touch the log store — durability
        comes from :meth:`flush` at the turn boundary.
        """
        lock = await self._lock_for(event.session_id)
        async with lock:
            await self._ensure_session_locked(event.session_id)
            self._fill_app_locked(event)
            event.seq = self._next_seq[event.session_id]
            event.ulid = str(ULID())
            self._next_seq[event.session_id] += 1
            self._buffers[event.session_id].append(event)
            return event.ref

    async def record_session_started(
        self,
        *,
        session_id: str,
        app_id: str | None = None,
        metadata: dict | None = None,
        observation_date: datetime | None = None,
    ) -> EventRef:
        # Establish the session's app attribution so later events inherit it
        # without the caller re-passing app_id on every record.
        self._app_ids[session_id] = app_id
        logger.info(
            "[episodic] app=%s session=%s session started", app_id, session_id
        )
        # When the conversation happened in the past (e.g. replaying an external
        # dialogue into memory), pin the event's timestamp to that date so the
        # distiller anchors relative time references correctly — distillation runs
        # offline, so wall-clock "now" would be the wrong anchor.  The distiller
        # anchors on the timestamps of the turns it distills (see
        # ``_thread_observation_date``), which the replayed messages also carry.
        event = MemoryEvent(
            session_id=session_id,
            event_type=EventType.SESSION_STARTED,
            app_id=app_id,
            payload=SessionStartedPayload(metadata=metadata or {}).model_dump(),
        )
        if observation_date is not None:
            event.created_at = observation_date
        return await self.record(event)

    async def record_user_message(
        self,
        *,
        session_id: str,
        content: str,
        attachments: list[dict] | None = None,
        observation_date: datetime | None = None,
    ) -> EventRef:
        # ``observation_date`` pins the turn's timestamp when replaying a past
        # dialogue (see ``record_session_started``); the distiller dates each
        # promoted memory from its source turns, so a turn replayed for a past
        # conversation must carry that conversation's date, not wall-clock now.
        event = MemoryEvent(
            session_id=session_id,
            event_type=EventType.USER_MESSAGE,
            payload=UserMessagePayload(
                text=content, attachments=attachments or []
            ).model_dump(),
        )
        if observation_date is not None:
            event.created_at = observation_date
        return await self.record(event)

    async def record_tool_call(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict,
        parent_event_id: EventRef | None = None,
    ) -> EventRef:
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                parent_event_id=parent_event_id,
                payload=ToolCalledPayload(
                    tool_call_id=tool_call_id, name=name, arguments=arguments
                ).model_dump(),
            )
        )

    async def record_tool_result(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        result: object | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
        parent_event_id: EventRef | None = None,
    ) -> EventRef:
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.TOOL_RESULT,
                parent_event_id=parent_event_id,
                payload=ToolResultPayload(
                    tool_call_id=tool_call_id,
                    ok=error is None,
                    result=result,
                    error=error,
                    latency_ms=latency_ms,
                ).model_dump(),
            )
        )

    async def record_retrieval_result(
        self,
        *,
        session_id: str,
        collection: str,
        query: str,
        hits: list[dict] | list[RetrievalHit],
        top_k: int | None = None,
        parent_event_id: EventRef | None = None,
    ) -> EventRef:
        """Record a typed retrieval projection for low-score mining.

        Emitted *in addition to* ``tool_result`` only once the gap detector wants
        score-filtered reads; until then retrieval folds into ``tool_result`` to
        avoid double-logging every search.
        """
        norm = [h if isinstance(h, RetrievalHit) else RetrievalHit(**h) for h in hits]
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.RETRIEVAL_RESULT,
                parent_event_id=parent_event_id,
                payload=RetrievalResultPayload(
                    collection=collection, query=query, hits=norm, top_k=top_k
                ).model_dump(),
            )
        )

    async def record_final_answer(
        self,
        *,
        session_id: str,
        answer: str,
        cited_ids: list[EventRef] | None = None,
        observation_date: datetime | None = None,
    ) -> EventRef:
        # See ``record_user_message`` re: ``observation_date`` — a replayed past
        # turn carries its conversation's date so its memories are dated correctly.
        event = MemoryEvent(
            session_id=session_id,
            event_type=EventType.FINAL_ANSWER,
            payload=FinalAnswerPayload(
                text=answer, cited_ids=cited_ids or []
            ).model_dump(),
        )
        if observation_date is not None:
            event.created_at = observation_date
        return await self.record(event)

    # TODO not used yet. need the end-to-end feedback mechanism from client.
    async def record_feedback(
        self,
        *,
        session_id: str,
        target: EventRef,
        rating: str,
        comment: str | None = None,
    ) -> EventRef:
        """Record feedback into the *current* session's log.

        Feedback carries the target event's triplet rather than being appended
        into the targeted session's log — writing into another session's log from
        this process would create a second writer for it and break the
        single-writer invariant.
        """
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.FEEDBACK,
                payload=FeedbackPayload(
                    target=target, rating=rating, comment=comment
                ).model_dump(),
            )
        )

    async def record_compaction(
        self,
        *,
        session_id: str,
        summary: str,
        replaces_through: int,
        token_stats: dict | None = None,
    ) -> EventRef:
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.SESSION_COMPACTED,
                payload=SessionCompactedPayload(
                    summary=summary,
                    replaces_through=replaces_through,
                    token_stats=token_stats or {},
                ).model_dump(),
            )
        )

    async def record_distillation(
        self,
        *,
        session_id: str,
        distilled_through: int,
        memory_count: int = 0,
    ) -> EventRef:
        """Record how far the offline distiller has extracted durable memories.

        The watermark counterpart of :meth:`record_compaction`: the distiller
        appends one ``session_distilled`` event per pass recording the last turn
        ``seq`` it covered, so a later distill of a resumed/re-closed session
        skips turns already distilled rather than re-reconciling them (see
        ``SessionDistilledPayload``).  Not continuity-critical — if lost, the
        worst case is a re-distill examining those turns again.
        """
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.SESSION_DISTILLED,
                payload=SessionDistilledPayload(
                    distilled_through=distilled_through,
                    memory_count=memory_count,
                ).model_dump(),
            )
        )

    # ------------------------------------------------------------------
    # Flush (turn-boundary durable append)
    # ------------------------------------------------------------------

    async def flush(self, session_id: str) -> None:
        """Append the session's buffered events to the log as one ordered write.

        Called at the turn boundary.  On success the buffer is cleared and the
        tracked ``expected_offset`` advances to the log's new length.  On a
        transient store failure the buffer is left intact (it *is* the retry
        buffer) and the error re-raised so the caller can retry or — for
        continuity-critical events — surface/alert rather than acknowledge the
        turn.  An empty buffer is a no-op.

        :class:`LogFenced` is *not* retryable: it means another owner appended to
        this session after a handoff, so this writer is deposed.  We drop the
        session's in-memory state (so a later turn re-resolves the tail cleanly if
        affinity legitimately returns here) and re-raise; the caller must fail the
        turn rather than acknowledge it.  Retrying would re-stamp the same ``seq``
        the fence just rejected — the very corruption the offset exists to stop.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            buffered = self._buffers.get(session_id)
            if not buffered:
                return
            lines = [e.to_ndjson() for e in buffered]
            # Append under the lock: a session has a single writer, so serializing
            # its own appends costs no cross-session concurrency.  The offset fences
            # a deposed co-owner whose flush would otherwise inject a colliding seq.
            try:
                new_offset = await self._log.append(
                    self._log_type,
                    session_id,
                    lines,
                    expected_offset=self._expected_offset.get(session_id),
                )
            except LogFenced:
                logger.critical(
                    "[episodic] app=%s session=%s FENCED on flush of %d event(s); "
                    "another writer owns this session — relinquishing",
                    self._app_ids.get(session_id),
                    session_id,
                    len(buffered),
                )
                self._drop_session_locked(session_id)
                raise
            self._expected_offset[session_id] = new_offset
            logger.info(
                "[episodic] app=%s session=%s flushed %d event(s) [%s] up to seq=%d (offset=%d)",
                self._app_ids.get(session_id),
                session_id,
                len(buffered),
                ", ".join(sorted({e.event_type.value for e in buffered})),
                buffered[-1].seq,
                new_offset,
            )
            self._buffers[session_id] = []

    def has_pending(self, session_id: str) -> bool:
        """True if the session has buffered events not yet flushed.

        Lets the caller decide whether a turn carrying continuity events is safe
        to acknowledge (see the durability invariant in docs/episodic-memory.md).
        """
        return bool(self._buffers.get(session_id))

    def pending_continuity(self, session_id: str) -> bool:
        """True if any *continuity-critical* event is still unflushed."""
        return any(
            e.event_type in CONTINUITY_EVENT_TYPES
            for e in self._buffers.get(session_id, ())
        )

    # ------------------------------------------------------------------
    # Read-back
    # ------------------------------------------------------------------

    async def replay(self, *, session_id: str) -> list[MemoryEvent]:
        """Return the whole session log in order (replay, debug, distillation)."""
        lines = await self._log.load_lines(self._log_type, session_id)
        return self._parse_dedup(lines)

    async def replay_since(
        self, *, session_id: str, offset: int
    ) -> tuple[list[MemoryEvent], int]:
        """Return events appended at/after byte *offset*, plus the log's new size.

        The incremental counterpart of :meth:`replay`: a consumer that has already
        folded the log up to *offset* re-reads only the tail past it, so a long,
        tool-heavy session no longer re-parses its whole log every turn (short-term
        memory's projection cache — see ``cogbase.memory.short_term``).  *offset*
        must be a record boundary (a prior size); ``replay_since(offset=0)`` is
        equivalent to :meth:`replay` plus the size.  A returned size below *offset*
        means the log shrank under the caller (deleted/recreated), who should then
        rebuild from ``0``.  Dedups the returned slice by ``ulid`` as :meth:`replay`
        does; cross-slice idempotency (a retry whose twin was already folded) is the
        consumer's watermark concern, since the duplicate's twin is not in view.
        """
        lines, size = await self._log.read_since(self._log_type, session_id, offset)
        return self._parse_dedup(lines), size

    async def tail(self, *, session_id: str, limit: int) -> list[MemoryEvent]:
        """Return the last *limit* events (short-term rehydrate)."""
        lines = await self._log.load_lines(self._log_type, session_id, tail=limit)
        return self._parse_dedup(lines)

    async def delete(self, *, session_id: str) -> None:
        """Delete the session's whole log and drop its in-memory state.

        Whole-object delete is the only supported mutation (retention/TTL,
        per-session erasure).
        """
        lock = await self._lock_for(session_id)
        async with lock:
            logger.info(
                "[episodic] app=%s session=%s deleting whole log",
                self._app_ids.get(session_id),
                session_id,
            )
            await self._log.delete(self._log_type, session_id)
            self._drop_session_locked(session_id)
            self._app_ids.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dedup(lines: list[str]) -> list[MemoryEvent]:
        """Parse NDJSON lines to events, dropping ``ulid`` duplicates (retries).

        Keeps the first occurrence so order is preserved.  Unparseable lines are
        skipped with a warning rather than failing the whole read.
        """
        events: list[MemoryEvent] = []
        seen: set[str] = set()
        for line in lines:
            if not line:
                continue
            try:
                event = MemoryEvent.from_ndjson(line)
            except Exception:
                logger.warning("[episodic] skipping unparseable log line", exc_info=True)
                continue
            if event.ulid and event.ulid in seen:
                continue
            seen.add(event.ulid)
            events.append(event)
        return events

    def _drop_session_locked(self, session_id: str) -> None:
        """Forget this writer's process-local state for *session_id*.

        Used when the session is deleted or when a flush is fenced (this writer
        was deposed).  Clearing ``_next_seq``/``_expected_offset`` forces the next
        ``record`` to re-resolve seq and offset from the log tail, so if affinity
        legitimately returns the session here it continues cleanly; if not, the
        next flush simply fences again — never corrupts the log.  Assumes the
        per-session lock is held.  ``_app_ids`` is attribution, not ordering
        state, so the caller pops it separately only on a true delete.
        """
        self._buffers.pop(session_id, None)
        self._next_seq.pop(session_id, None)
        self._expected_offset.pop(session_id, None)

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    async def _ensure_session_locked(self, session_id: str) -> None:
        """Initialize the session's seq counter from the log tail on first use.

        On a cold start (new process taking over a session) the next ``seq`` is
        the last persisted ``seq`` + 1, so the sequence continues without
        duplication.  Assumes the per-session lock is held.
        """
        if session_id in self._next_seq:
            return
        self._buffers.setdefault(session_id, [])
        last = await self._log.load_lines(self._log_type, session_id, tail=1)
        # Seed the fencing offset from the log's current byte length: the first
        # flush conditions on it, so a co-owner that advanced the log in the
        # meantime fences us instead of letting both writers stamp the same seq.
        self._expected_offset[session_id] = await self._log.size(
            self._log_type, session_id
        )
        next_seq = 0
        if last:
            try:
                next_seq = MemoryEvent.from_ndjson(last[-1]).seq + 1
            except Exception:
                logger.warning(
                    "[episodic] could not parse seq from log tail for %s; starting at 0",
                    session_id,
                    exc_info=True,
                )
        self._next_seq[session_id] = next_seq
        if next_seq > 0:
            logger.info(
                "[episodic] app=%s session=%s resumed from log; next seq=%d",
                self._app_ids.get(session_id),
                session_id,
                next_seq,
            )

    def _fill_app_locked(self, event: MemoryEvent) -> None:
        """Inherit app_id from the session's recorded attribution."""
        if event.app_id is None:
            event.app_id = self._app_ids.get(event.session_id)
