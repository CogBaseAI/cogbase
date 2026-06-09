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

Read-back (:meth:`replay`, :meth:`tail`) dedupes by ``ulid`` so a retried append
that double-wrote a line surfaces once.  Quarantining a ``seq``-colliding
out-of-order straggler is left to short-term rehydrate (build-order step 5),
which is the consumer that threads events into a conversation.
"""

from __future__ import annotations

import asyncio
import logging

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
    SessionStartedPayload,
    ToolCalledPayload,
    ToolResultPayload,
    UserMessagePayload,
)
from cogbase.stores.log.base import LogStoreBase

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
        self._scope: dict[str, dict] = {}                  # app_id / user_id per session
        # One lock per session keeps cross-session appends concurrent while
        # serializing a session's own record/flush (it has a single writer).
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------------
    # Recording (buffer + stamp)
    # ------------------------------------------------------------------

    def bind_scope(
        self,
        session_id: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Register a session's attribution scope without emitting an event.

        Later recorded events inherit ``app_id`` / ``user_id`` so callers need
        not re-pass them on every record.  Idempotent and process-local (rebuilt
        on a cold start), so it is safe — and expected — to call once per turn.
        Use this where ``record_session_started`` would be wrong: the per-turn
        query runner does not own session creation and must not log a
        ``session_started`` event on every turn.
        """
        self._scope[session_id] = {"app_id": app_id, "user_id": user_id}

    async def record(self, event: MemoryEvent) -> EventRef:
        """Stamp *event* with its ``seq`` + ``ulid`` and buffer it for the next flush.

        Returns the event's identity triplet so callers can thread it into a
        ``parent_event_id`` / reference.  Does not touch the log store — durability
        comes from :meth:`flush` at the turn boundary.
        """
        lock = await self._lock_for(event.session_id)
        async with lock:
            await self._ensure_session_locked(event.session_id)
            self._fill_scope_locked(event)
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
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> EventRef:
        # Establish the session's scope so later events inherit it without the
        # caller re-passing app_id / user_id on every record.
        self._scope[session_id] = {"app_id": app_id, "user_id": user_id}
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.SESSION_STARTED,
                app_id=app_id,
                user_id=user_id,
                payload=SessionStartedPayload(metadata=metadata or {}).model_dump(),
            )
        )

    async def record_user_message(
        self,
        *,
        session_id: str,
        content: str,
        attachments: list[dict] | None = None,
    ) -> EventRef:
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.USER_MESSAGE,
                payload=UserMessagePayload(
                    text=content, attachments=attachments or []
                ).model_dump(),
            )
        )

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
    ) -> EventRef:
        return await self.record(
            MemoryEvent(
                session_id=session_id,
                event_type=EventType.FINAL_ANSWER,
                payload=FinalAnswerPayload(
                    text=answer, cited_ids=cited_ids or []
                ).model_dump(),
            )
        )

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

    # ------------------------------------------------------------------
    # Flush (turn-boundary durable append)
    # ------------------------------------------------------------------

    async def flush(self, session_id: str) -> None:
        """Append the session's buffered events to the log as one ordered write.

        Called at the turn boundary.  On success the buffer is cleared; on a
        store failure the buffer is left intact (it *is* the retry buffer) and the
        error re-raised so the caller can retry or — for continuity-critical
        events — surface/alert rather than acknowledge the turn.  An empty buffer
        is a no-op.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            buffered = self._buffers.get(session_id)
            if not buffered:
                return
            lines = [e.to_ndjson() for e in buffered]
            # Append under the lock: a session has a single writer, so serializing
            # its own appends costs no cross-session concurrency.
            await self._log.append(self._log_type, session_id, lines)
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
            await self._log.delete(self._log_type, session_id)
            self._buffers.pop(session_id, None)
            self._next_seq.pop(session_id, None)
            self._scope.pop(session_id, None)

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

    def _fill_scope_locked(self, event: MemoryEvent) -> None:
        """Inherit app_id / user_id from the session's recorded scope."""
        scope = self._scope.get(event.session_id)
        if not scope:
            return
        if event.app_id is None:
            event.app_id = scope.get("app_id")
        if event.user_id is None:
            event.user_id = scope.get("user_id")
