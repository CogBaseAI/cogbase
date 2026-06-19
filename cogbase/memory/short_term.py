"""Short-term memory: a projection over the episodic log.

Short-term memory owns the working context for an active session: it decides
what belongs in the *next* LLM call.  It is **not** a store of its own — the
episodic event log is its persistence (see ``docs/episodic-memory.md#short-term
-memory-rides-on-the-same-log``).  Its durable record is the ``user_message`` /
``final_answer`` continuity events the query runner already appends each turn,
plus the ``session_compacted`` summaries this layer appends when the thread
nears the model window.

The three operations the layer performs:

- **Rehydrate.** :meth:`build_context` rebuilds the conversational thread by
  reading the (small) whole log and projecting it: the latest
  ``session_compacted`` summary plus every continuity event after the ``seq`` it
  covers (``replaces_through``).  On a cold start or a process handoff this is a
  full reconstruction; any process can serve any session because the log is the
  source of truth.
- **Append-a-summary compaction.** When the projected thread approaches the
  model-context budget, the overflow (oldest turns) is folded into the running
  summary and persisted as a *new* ``session_compacted`` event — the log is
  append-only, so compaction never rewrites history.  The event rides the
  current turn's flush, so it lands durably alongside the turn's
  ``final_answer``.  A lost in-memory summary is self-healing: the raw turns are
  still in the log, so the next rehydrate re-derives it.
- **Near pass-through assembly.** Because compaction keeps the thread under the
  budget, context assembly is "summary header + the projected turns + the
  current input" — no per-turn newest-first budget walk.

Rehydrate does not re-read the whole log every turn.  A per-session projection
cache holds the folded thread (summary + turns) plus the byte ``offset`` it has
consumed; each turn reads only the events appended past that watermark
(``replay_since``) and folds them in, so a long, tool-heavy session no longer
pays an ever-growing parse cost to rebuild a bounded context.  The cache is a
pure optimization: a cold process (empty cache) folds from ``offset=0``, which is
a full replay, and the byte offset is reconciled against the log's real size
every turn — so a warm and a cold ``build_context`` still produce the same
result.  Compaction (the rare, over-budget path) drops the cache so the next turn
rebuilds cleanly rather than reasoning about a half-applied summary.

The cache also holds session metadata (app) and TTL, evicted together.

Concurrency is per session: each session has its own lock, so compacting one
session (including its slow LLM summary, which runs outside the lock) never
blocks context builds for other sessions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from cogbase.llms.compaction import (
    CONVERSATION_SUMMARY_PROMPT,
    context_budget_tokens,
    estimate_tokens,
    summarise_chunk_tokens,
    summarize_transcript,
)
from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.models import (
    EventType,
    MemoryEvent,
    MemoryMessage,
    SessionState,
)
from cogbase.memory.projection import (
    continuity_role,
    latest_compaction,
    message_from_event,
    project_thread,
)

# Fallback working-context budget when no LLM is configured (so the model window
# is unknown and no compaction runs anyway). When an LLM *is* present the budget
# is derived from its context window via ``context_budget_tokens`` — a fraction
# of the real window, so it tracks the deployed model and can never exceed it.
# Equals DEFAULT_CONTEXT_WINDOW * CONTEXT_BUDGET_RATIO.
DEFAULT_COMPACTION_TOKEN_BUDGET = 96_000

# Default session time-to-live for the metadata cache.  None means never expire.
# This only bounds the in-memory cache; the durable log has its own retention
# clock (see docs/episodic-memory.md#retention-deletion-and-redaction).
DEFAULT_TTL_SECONDS = 3600

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _CompactionPlan:
    """The folding decision made under the lock, summarised outside it.

    Captures everything :meth:`ShortTermMemory._commit_compaction` needs to
    persist the result after the (slow, lock-free) LLM summary returns: the
    overflow turns being folded, the last ``seq`` the new summary will cover, the
    transcript handed to the summariser, and the pre-fold thread size (for stats).
    """

    overflow: list[MemoryMessage]
    covered: int
    transcript: str
    thread_tokens: int


@dataclass
class _ProjectionCache:
    """The folded projection of a session's log, plus the watermark it covers.

    Lets :meth:`ShortTermMemory._rehydrate` re-read only the events appended since
    the last turn instead of replaying the whole log.  ``offset`` is the log byte
    length already folded (a record boundary); ``folded_seq`` is the highest event
    ``seq`` folded, the dedup watermark that drops a straggler or a non-fenced
    retry whose twin is already in the projection.  The remaining fields are the
    projection itself — identical to what a full :meth:`_project` would yield.
    """

    offset: int
    folded_seq: int
    summary: str | None
    replaces_through: int
    messages: list[MemoryMessage]


class ShortTermMemory:
    """Per-session working context, projected from the episodic log.

    Args:
        episodic:                The append-only event log this layer rides on;
                                 the source of truth for every session's thread.
                                 Must be the same instance the query runner
                                 records into, so compaction events ride the
                                 turn's flush.
        ttl_seconds:             Idle lifetime of the in-memory metadata cache;
                                 ``None`` disables expiry.  Does not affect the
                                 durable log.
        compaction_token_budget: Estimated-token threshold above which the
                                 rehydrated thread is compacted into a summary.
                                 ``None`` (the default) derives it from ``llm``'s
                                 context window — a fraction that tracks the
                                 deployed model — falling back to a constant when
                                 no LLM is configured.
        llm:                     LLM used to summarise overflow turns during
                                 compaction.  When ``None`` no compaction runs
                                 and the full thread is assembled as-is.
    """

    def __init__(
        self,
        *,
        episodic: EpisodicMemory,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
        compaction_token_budget: int | None = None,
        llm: LLMBase | None = None,
    ) -> None:
        self._episodic = episodic
        self._ttl_seconds = ttl_seconds
        self._llm = llm
        # Budget tracks the answering model's window when an LLM is present; an
        # explicit value still wins for deployments that want to pin it.
        if compaction_token_budget is not None:
            self._compaction_token_budget = compaction_token_budget
        elif llm is not None:
            self._compaction_token_budget = context_budget_tokens(llm)
        else:
            self._compaction_token_budget = DEFAULT_COMPACTION_TOKEN_BUDGET
        # Metadata cache (app + TTL); the thread itself is projected from the log.
        self._sessions: dict[str, SessionState] = {}
        # Per-session projection cache: the folded thread + the log offset it
        # covers, so a rehydrate re-reads only the tail appended since (see
        # ``_rehydrate``).  Evicted alongside ``_sessions`` — same lifetime.
        self._projections: dict[str, _ProjectionCache] = {}
        # One lock per session serializes that session's own rehydrate/compaction
        # while leaving other sessions concurrent — a slow compaction no longer
        # stalls every session.  The slow LLM summary itself runs *outside* the
        # lock (see ``build_context``).  Mutations of ``_sessions`` are otherwise
        # synchronous (no await), so they need no separate guard; ``_locks_guard``
        # only protects lazy creation of the per-session locks.  Mirrors
        # :class:`EpisodicMemory`'s per-session locking.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(
        self,
        *,
        app_id: str | None = None,
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create (or resume) a session and return its id.

        Seeds the metadata cache only; the conversational thread lives in the
        log and is materialised on the first ``build_context``.  Resuming an
        existing ``session_id`` is idempotent.
        """
        # Resuming a known session serializes with that session's other ops; a
        # brand-new session has no id yet, so there is nothing to contend with.
        if session_id is not None:
            lock = await self._lock_for(session_id)
            async with lock:
                return self._resume_or_create(app_id, metadata, session_id)
        return self._resume_or_create(app_id, metadata, None)

    def _resume_or_create(
        self,
        app_id: str | None,
        metadata: dict | None,
        session_id: str | None,
    ) -> str:
        """Synchronous core of :meth:`start_session` (no await, so atomic)."""
        self._sweep_expired()
        if session_id is not None and session_id in self._sessions:
            state = self._sessions[session_id]
            self._touch(state)
            logger.info(
                "[short_term] app=%s session=%s resumed (cached)",
                state.app_id, state.session_id,
            )
            return state.session_id
        state = SessionState(
            app_id=app_id,
            metadata=metadata or {},
            **({"session_id": session_id} if session_id else {}),
        )
        self._touch(state)
        self._sessions[state.session_id] = state
        logger.info(
            "[short_term] app=%s session=%s started", app_id, state.session_id
        )
        return state.session_id

    async def get(self, session_id: str) -> SessionState | None:
        """Return the projected session state, or ``None`` if it has no history.

        Rehydrates the conversational thread from the log so the returned state
        reflects every durably-recorded turn — including turns recorded by
        another process.  Returns ``None`` only when the session is neither
        cached nor present in the log.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            self._sweep_expired()
            cached = self._get_live(session_id)
            summary, _, messages = await self._rehydrate(session_id)
            proj = self._projections[session_id]
            if cached is None and proj.offset == 0 and not messages and summary is None:
                # Neither cached metadata nor any durable history → unknown session.
                self._projections.pop(session_id, None)
                return None
            state = cached or SessionState(session_id=session_id)
            state.messages = messages
            state.summary = summary
            self._touch(state)
            self._sessions[session_id] = state
            return state

    async def end_session(self, session_id: str) -> None:
        """Drop the cached metadata for a session.

        This evicts only the in-memory cache; the durable log is untouched (its
        deletion is episodic memory's concern — retention / erasure).
        """
        lock = await self._lock_for(session_id)
        async with lock:
            state = self._sessions.pop(session_id, None)
            self._projections.pop(session_id, None)
            logger.info(
                "[short_term] app=%s session=%s cached context evicted",
                state.app_id if state else None, session_id,
            )

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    async def build_context(
        self,
        *,
        session_id: str,
        current_user_message: str | None = None,
        token_budget: int | None = None,
    ) -> list[ChatMessage]:
        """Assemble the message list for the next LLM call.

        Rehydrates the thread from the log, compacts it if it has grown past the
        budget (appending a ``session_compacted`` event that rides the turn's
        flush), then assembles: an optional summary header, the projected turns,
        and ``current_user_message`` as the final user turn.

        ``current_user_message`` is the turn's input.  The runner records it into
        the episodic log separately; it is passed here because that record is
        still buffered (not yet flushed) when this runs, so the projection — which
        reads the flushed log — would not otherwise include it.
        """
        budget = token_budget or self._compaction_token_budget
        lock = await self._lock_for(session_id)

        # Phase 1 — rehydrate the thread and decide whether to compact.  Cheap (a
        # log read plus a token tally), so it is fine to hold the session lock.
        async with lock:
            self._sweep_expired()
            # Re-reads only the log tail appended since last turn and folds it into
            # the cached projection (full replay only on a cold cache); see
            # ``_rehydrate``.
            summary, replaces_through, messages = await self._rehydrate(session_id)
            plan = self._plan_compaction(summary, replaces_through, messages, budget)

        # Phase 2 — summarise the overflow *outside* the lock.  This is the slow
        # step (an LLM call); holding the lock across it would let one session's
        # compaction block every other session.  On failure the full phase-1
        # thread is served unchanged — never truncated without a covering summary.
        summarized = 0
        if plan is not None:
            try:
                new_summary = await summarize_transcript(
                    self._llm,
                    plan.transcript,
                    chunk_tokens=summarise_chunk_tokens(self._llm),
                    prior_summary=summary,
                    compress_prompt=CONVERSATION_SUMMARY_PROMPT,
                )
            except Exception:
                logger.warning(
                    "[short_term] compaction failed; serving the full thread",
                    exc_info=True,
                )
            else:
                # Phase 3 — re-acquire the lock, recheck the log, then persist.
                async with lock:
                    summary, messages, summarized = await self._commit_compaction(
                        session_id, plan, new_summary, summary, budget
                    )

        async with lock:
            if plan is not None:
                # Compaction (the rare, over-budget path) reshaped the thread and
                # may have raced a concurrent compaction that flushed mid-summary;
                # drop the projection cache so the next turn rebuilds from the log
                # rather than folding new events onto a now-stale watermark.
                self._projections.pop(session_id, None)
            state = self._get_live(session_id) or SessionState(session_id=session_id)
            state.messages = messages
            state.summary = summary
            state.metadata["last_context"] = {
                "budget": budget,
                "included_messages": len(messages),
                "summarized_messages": summarized,
                "has_summary": summary is not None,
            }
            self._touch(state)
            self._sessions[session_id] = state

        logger.info(
            "[short_term] app=%s session=%s context built: %d message(s), "
            "summarized=%d, has_summary=%s, budget=%d",
            state.app_id, session_id, len(messages), summarized,
            summary is not None, budget,
        )

        context: list[ChatMessage] = []
        if summary:
            context.append(
                {
                    "role": "system",
                    "content": (
                        "Summary of earlier conversation in this session:\n"
                        f"{summary}"
                    ),
                }
            )
        for msg in messages:
            context.append({"role": msg.role.value, "content": msg.content})
        if current_user_message:
            context.append({"role": "user", "content": current_user_message})
        return context

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    async def _rehydrate(
        self, session_id: str
    ) -> tuple[str | None, int, list[MemoryMessage]]:
        """Project the thread, re-reading only the log tail since the last turn.

        Folds the events appended past the cached watermark into the cached
        projection; on a cold cache — or after the log shrank under us — it folds
        from ``offset=0``, which is a full replay.  Refreshes the projection cache
        and returns ``(summary, replaces_through, messages)``: the same triple a
        full :meth:`_project` yields, so a warm and a cold rehydrate agree.

        Assumes the per-session lock is held (it mutates the projection cache).
        """
        cache = self._projections.get(session_id)
        start = cache.offset if cache else 0
        events, size = await self._episodic.replay_since(
            session_id=session_id, offset=start
        )
        if cache is not None and size < cache.offset:
            # The watermark points into a log that no longer exists (deleted /
            # recreated under us): discard the cache and fold from scratch.
            cache = None
            events, size = await self._episodic.replay_since(
                session_id=session_id, offset=0
            )
        if cache is None:
            summary, replaces_through, messages = self._project(events)
            folded_seq = max((e.seq for e in events), default=-1)
        else:
            summary, replaces_through, messages, folded_seq = self._fold(cache, events)
        self._projections[session_id] = _ProjectionCache(
            offset=size,
            folded_seq=folded_seq,
            summary=summary,
            replaces_through=replaces_through,
            messages=messages,
        )
        return summary, replaces_through, messages

    @staticmethod
    def _fold(
        cache: _ProjectionCache, new_events: list[MemoryEvent]
    ) -> tuple[str | None, int, list[MemoryMessage], int]:
        """Fold events appended past the cache's watermark into its projection.

        Mirrors :meth:`_project` incrementally: a ``session_compacted`` advances
        the running summary and drops the turns it now covers; a continuity event
        after the current ``replaces_through`` is appended as a turn.
        ``folded_seq`` gates re-folding — an event at/below it is a straggler or a
        non-fenced retry whose twin is already projected, so first occurrence
        wins (as :func:`project_thread` dedupes by seq).  This leans on the log's
        single-writer guarantee that ``seq`` is non-decreasing in log order;
        ``replaces_through`` is likewise monotonic, so a turn dropped by one
        compaction is never re-added by a later read.  Returns a fresh message
        list so the cached projection is never mutated through the returned value.
        """
        summary = cache.summary
        replaces_through = cache.replaces_through
        folded_seq = cache.folded_seq
        messages = list(cache.messages)
        for event in new_events:
            if event.seq <= folded_seq:
                continue
            folded_seq = event.seq
            if event.event_type is EventType.SESSION_COMPACTED:
                summary = event.payload.get("summary")
                replaces_through = int(
                    event.payload.get("replaces_through", replaces_through)
                )
                messages = [
                    m for m in messages if m.seq is None or m.seq > replaces_through
                ]
            elif (
                continuity_role(event.event_type) is not None
                and event.seq > replaces_through
            ):
                messages.append(message_from_event(event))
        return summary, replaces_through, messages, folded_seq

    @staticmethod
    def _project(
        events: list[MemoryEvent],
    ) -> tuple[str | None, int, list[MemoryMessage]]:
        """Project a session's events into (summary, replaces_through, messages).

        Takes the latest ``session_compacted`` summary and the ``seq`` it covers,
        then every continuity event after that ``seq`` in order.  The thread
        projection is the shared :mod:`cogbase.memory.projection` helper (also
        used by the distiller); this method layers the summary lookup on top.
        """
        summary, replaces_through = latest_compaction(events)
        messages = project_thread(events, since_seq=replaces_through)
        return summary, replaces_through, messages

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _plan_compaction(
        self,
        summary: str | None,
        replaces_through: int,
        messages: list[MemoryMessage],
        budget: int,
    ) -> _CompactionPlan | None:
        """Decide which oldest turns to fold when the thread is over *budget*.

        Pure and synchronous: it picks the overflow and builds the transcript but
        never calls the LLM, so it can run under the session lock while the actual
        (slow) summary runs outside it.  Returns ``None`` when there is no LLM,
        the thread fits, or there is nothing to fold.
        """
        thread_tokens = (estimate_tokens(summary) if summary else 0) + sum(
            m.token_estimate for m in messages
        )
        if self._llm is None or thread_tokens <= budget:
            return None

        # Walk newest-first, keeping turns that fit half the budget (leaving room
        # for the summary header and the current turn); the rest overflow.
        overflow_rev: list[MemoryMessage] = []
        running = 0
        keep_budget = budget // 2
        kept_any = False
        for msg in reversed(messages):
            cost = msg.token_estimate or estimate_tokens(msg.content)
            if running + cost <= keep_budget or not kept_any:
                kept_any = True
                running += cost
            else:
                overflow_rev.append(msg)

        overflow = list(reversed(overflow_rev))
        if not overflow:
            return None

        # The last seq the new summary covers: the highest folded seq, but never
        # below the prior summary's coverage.
        covered = max(replaces_through, max(m.seq for m in overflow if m.seq is not None))
        transcript = "\n".join(f"[{m.role.value}] {m.content}" for m in overflow)
        return _CompactionPlan(
            overflow=overflow,
            covered=covered,
            transcript=transcript,
            thread_tokens=thread_tokens,
        )

    async def _commit_compaction(
        self,
        session_id: str,
        plan: _CompactionPlan,
        new_summary: str | None,
        prior_summary: str | None,
        budget: int,
    ) -> tuple[str | None, list[MemoryMessage], int]:
        """Persist the summary from *plan* and return the retained turns.

        Runs under the session lock after the LLM summary completes.  Re-reads the
        log first (the lock was released for the summary): if a compaction already
        covers ``plan.covered`` it serves that projection instead of double-folding;
        otherwise it records the new summary and re-projects so any turns appended
        meanwhile are kept.  Never truncates without a covering summary.
        """
        events = await self._episodic.replay(session_id=session_id)
        latest_summary, latest_through = latest_compaction(events)
        if latest_through >= plan.covered:
            # A compaction already covers these turns (e.g. one flushed while the
            # LLM ran); honour it rather than fold again.
            return latest_summary, project_thread(events, since_seq=latest_through), 0

        new_summary = new_summary or prior_summary
        if not new_summary:
            # Summariser produced nothing and there was no prior summary: keep the
            # turns rather than drop them without a covering summary.
            return latest_summary, project_thread(events, since_seq=latest_through), 0

        # Persist the summary as a new event so future rehydrates honour it.  It
        # is buffered now and flushed with the turn's continuity events, so a
        # crash before the flush only costs a re-compaction (the raw turns are
        # still in the log) — the durability invariant holds.
        await self._episodic.record_compaction(
            session_id=session_id,
            summary=new_summary,
            replaces_through=plan.covered,
            token_stats={"thread_tokens": plan.thread_tokens, "budget": budget},
        )
        kept = project_thread(events, since_seq=plan.covered)
        state = self._get_live(session_id)
        logger.info(
            "[short_term] app=%s session=%s compacted %d turn(s) into summary "
            "(thread_tokens=%d > budget=%d), replaces_through=%d, %d turn(s) kept",
            state.app_id if state else None, session_id, len(plan.overflow),
            plan.thread_tokens, budget, plan.covered, len(kept),
        )
        return new_summary, kept, len(plan.overflow)

    # ------------------------------------------------------------------
    # Internal helpers (synchronous: no await, so atomic without a lock)
    # ------------------------------------------------------------------

    def _get_live(self, session_id: str) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        if state.is_expired():
            self._sessions.pop(session_id, None)
            return None
        return state

    def _touch(self, state: SessionState) -> None:
        now = _utcnow()
        state.last_active_at = now
        if self._ttl_seconds is not None:
            state.expires_at = now + timedelta(seconds=self._ttl_seconds)

    def _sweep_expired(self) -> None:
        now = _utcnow()
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(now)]
        for sid in expired:
            self._sessions.pop(sid, None)
            # The projection cache shares the session's lifetime — drop it too so
            # an idle session does not pin its folded thread in memory.
            self._projections.pop(sid, None)
