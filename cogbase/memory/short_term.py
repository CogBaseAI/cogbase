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

A small per-session cache holds only session metadata (app/user/scope) and TTL;
the conversational thread is always projected fresh from the log, so a
``build_context`` from a warm or a cold process produces the same result.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from cogbase.llms.compaction import (
    CONVERSATION_SUMMARY_PROMPT,
    DEFAULT_CHUNK_TOKENS,
    estimate_tokens,
    summarize_transcript,
)
from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.models import (
    EventType,
    MemoryEvent,
    MemoryMessage,
    MemoryRole,
    SessionState,
)

# Default working-context budget, in estimated tokens, that triggers compaction.
#
# Compaction fires on *model-context* pressure — when the rehydrated thread
# approaches a fixed fraction of the LLM window — not on a small per-turn budget.
# Keeping it large (a fraction of a 128k-token window, leaving room for the
# system prompt, retrieval, skills, and output) keeps compaction rare and each
# persisted ``session_compacted`` summary worth keeping.  Override per-instance
# via ``compaction_token_budget``.
DEFAULT_COMPACTION_TOKEN_BUDGET = 96_000

# Default session time-to-live for the metadata cache.  None means never expire.
# This only bounds the in-memory cache; the durable log has its own retention
# clock (see docs/episodic-memory.md#retention-deletion-and-redaction).
DEFAULT_TTL_SECONDS = 3600

# Continuity events short-term threads into the conversation, mapped to the role
# they project to.  Tool calls/results are intra-turn scratch and never threaded.
_CONTINUITY_ROLE: dict[EventType, MemoryRole] = {
    EventType.USER_MESSAGE: MemoryRole.USER,
    EventType.FINAL_ANSWER: MemoryRole.ASSISTANT,
}

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
        llm:                     LLM used to summarise overflow turns during
                                 compaction.  When ``None`` no compaction runs
                                 and the full thread is assembled as-is.
    """

    def __init__(
        self,
        *,
        episodic: EpisodicMemory,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
        compaction_token_budget: int = DEFAULT_COMPACTION_TOKEN_BUDGET,
        llm: LLMBase | None = None,
    ) -> None:
        self._episodic = episodic
        self._ttl_seconds = ttl_seconds
        self._compaction_token_budget = compaction_token_budget
        self._llm = llm
        # Cache holds only session metadata + TTL; the thread is projected fresh
        # from the log on every read, so a warm and a cold process agree.
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(
        self,
        *,
        app_name: str | None = None,
        user_id: str | None = None,
        scope: dict | None = None,
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create (or resume) a session and return its id.

        Seeds the metadata cache only; the conversational thread lives in the
        log and is materialised on the first ``build_context``.  Resuming an
        existing ``session_id`` is idempotent.
        """
        async with self._lock:
            self._sweep_expired_locked()
            if session_id is not None and session_id in self._sessions:
                state = self._sessions[session_id]
                self._touch_locked(state)
                return state.session_id
            state = SessionState(
                app_name=app_name,
                user_id=user_id,
                scope=scope or {},
                metadata=metadata or {},
                **({"session_id": session_id} if session_id else {}),
            )
            self._touch_locked(state)
            self._sessions[state.session_id] = state
            return state.session_id

    async def get(self, session_id: str) -> SessionState | None:
        """Return the projected session state, or ``None`` if it has no history.

        Rehydrates the conversational thread from the log so the returned state
        reflects every durably-recorded turn — including turns recorded by
        another process.  Returns ``None`` only when the session is neither
        cached nor present in the log.
        """
        async with self._lock:
            self._sweep_expired_locked()
            cached = self._get_live_locked(session_id)
            events = await self._episodic.replay(session_id=session_id)
            if cached is None and not events:
                return None
            summary, _, messages = self._project(events)
            state = cached or SessionState(session_id=session_id)
            state.messages = messages
            state.summary = summary
            self._touch_locked(state)
            self._sessions[session_id] = state
            return state

    async def end_session(self, session_id: str) -> None:
        """Drop the cached metadata for a session.

        This evicts only the in-memory cache; the durable log is untouched (its
        deletion is episodic memory's concern — retention / erasure).
        """
        async with self._lock:
            self._sessions.pop(session_id, None)

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
        async with self._lock:
            self._sweep_expired_locked()
            # replay reads the session log to make sure every query sees the latest messages.
            # TODO add a cache layer.
            events = await self._episodic.replay(session_id=session_id)
            summary, replaces_through, messages = self._project(events)

            summary, messages, summarized = await self._maybe_compact_locked(
                session_id, summary, replaces_through, messages, budget
            )

            state = self._get_live_locked(session_id) or SessionState(session_id=session_id)
            state.messages = messages
            state.summary = summary
            state.metadata["last_context"] = {
                "budget": budget,
                "included_messages": len(messages),
                "summarized_messages": summarized,
                "has_summary": summary is not None,
            }
            self._touch_locked(state)
            self._sessions[session_id] = state

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

    @staticmethod
    def _project(
        events: list[MemoryEvent],
    ) -> tuple[str | None, int, list[MemoryMessage]]:
        """Project a session's events into (summary, replaces_through, messages).

        Takes the latest ``session_compacted`` summary and the ``seq`` it covers,
        then every continuity event after that ``seq`` in order.  ``events`` are
        already ulid-deduped and in log order; a ``seq`` is kept on first
        occurrence, so an out-of-order straggler that reuses a ``seq`` does not
        displace the active writer's event (see
        docs/episodic-memory.md#single-writer-and-append-safety).
        """
        summary: str | None = None
        replaces_through = -1
        for event in events:
            if event.event_type is EventType.SESSION_COMPACTED:
                summary = event.payload.get("summary")
                replaces_through = int(event.payload.get("replaces_through", -1))

        messages: list[MemoryMessage] = []
        seen_seqs: set[int] = set()
        for event in events:
            role = _CONTINUITY_ROLE.get(event.event_type)
            if role is None or event.seq <= replaces_through or event.seq in seen_seqs:
                continue
            seen_seqs.add(event.seq)
            text = event.payload.get("text", "")
            messages.append(
                MemoryMessage(
                    role=role,
                    content=text,
                    seq=event.seq,
                    token_estimate=estimate_tokens(text),
                )
            )
        return summary, replaces_through, messages

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _maybe_compact_locked(
        self,
        session_id: str,
        summary: str | None,
        replaces_through: int,
        messages: list[MemoryMessage],
        budget: int,
    ) -> tuple[str | None, list[MemoryMessage], int]:
        """Fold the oldest turns into the running summary when over *budget*.

        Returns the (possibly updated) summary, the retained messages, and the
        number of messages summarised.  On no LLM, or a transient LLM failure, or
        nothing to fold, the thread is returned unchanged — never truncated
        without a durable summary covering the dropped turns (the durability
        invariant).  Assumes the instance lock is held.
        """
        thread_tokens = (estimate_tokens(summary) if summary else 0) + sum(
            m.token_estimate for m in messages
        )
        if self._llm is None or thread_tokens <= budget:
            return summary, messages, 0

        # Walk newest-first, keeping turns that fit half the budget (leaving room
        # for the summary header and the current turn); the rest overflow.
        kept_rev: list[MemoryMessage] = []
        overflow_rev: list[MemoryMessage] = []
        running = 0
        keep_budget = budget // 2
        for msg in reversed(messages):
            cost = msg.token_estimate or estimate_tokens(msg.content)
            if (running + cost <= keep_budget or not kept_rev):
                kept_rev.append(msg)
                running += cost
            else:
                overflow_rev.append(msg)

        overflow = list(reversed(overflow_rev))
        if not overflow:
            return summary, messages, 0
        kept = list(reversed(kept_rev))

        # The last seq the new summary covers: the highest folded seq, but never
        # below the prior summary's coverage.
        covered = max(replaces_through, max(m.seq for m in overflow if m.seq is not None))
        transcript = "\n".join(f"[{m.role.value}] {m.content}" for m in overflow)
        try:
            new_summary = await summarize_transcript(
                self._llm,
                transcript,
                chunk_tokens=DEFAULT_CHUNK_TOKENS,
                prior_summary=summary,
                compress_prompt=CONVERSATION_SUMMARY_PROMPT,
            )
        except Exception:
            logger.warning(
                "[short_term] compaction failed; serving the full thread", exc_info=True
            )
            return summary, messages, 0

        new_summary = new_summary or summary
        if not new_summary:
            # Summariser produced nothing and there was no prior summary: keep the
            # turns rather than drop them without a covering summary.
            return summary, messages, 0

        # Persist the summary as a new event so future rehydrates honour it.  It
        # is buffered now and flushed with the turn's continuity events, so a
        # crash before the flush only costs a re-compaction (the raw turns are
        # still in the log) — the durability invariant holds.
        await self._episodic.record_compaction(
            session_id=session_id,
            summary=new_summary,
            replaces_through=covered,
            token_stats={"thread_tokens": thread_tokens, "budget": budget},
        )
        return new_summary, kept, len(overflow)

    # ------------------------------------------------------------------
    # Internal helpers (assume the lock is held)
    # ------------------------------------------------------------------

    def _get_live_locked(self, session_id: str) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        if state.is_expired():
            self._sessions.pop(session_id, None)
            return None
        return state

    def _touch_locked(self, state: SessionState) -> None:
        now = _utcnow()
        state.last_active_at = now
        if self._ttl_seconds is not None:
            state.expires_at = now + timedelta(seconds=self._ttl_seconds)

    def _sweep_expired_locked(self) -> None:
        now = _utcnow()
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(now)]
        for sid in expired:
            self._sessions.pop(sid, None)
