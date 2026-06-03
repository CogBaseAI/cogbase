"""Short-term memory: session-local working context.

Short-term memory owns the working context for an active session.  Its single
responsibility is to decide what belongs in the *next* LLM call: it holds the
recent transcript plus retrieved evidence, compacts older turns into a summary
when the raw transcript exceeds a token budget, and assembles a bounded message
list on demand.

The initial implementation is in-memory (a dict keyed by ``session_id`` with
lazy TTL expiry).  Redis or another expiring cache can replace the backing
store later for multi-worker deployments without changing this interface — all
public methods are ``async`` for exactly that reason.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.memory.models import (
    MemoryMessage,
    MemoryRole,
    RetrievedItem,
    SessionState,
)

# Default working-context budget, in estimated tokens, used when a caller does
# not specify one for build_context().
DEFAULT_CONTEXT_TOKEN_BUDGET = 4000

# Default session time-to-live.  None means sessions never expire.
DEFAULT_TTL_SECONDS = 3600


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token), matching the query runner heuristic."""
    return len(text) // 4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ShortTermMemory:
    """In-memory store of per-session working context.

    Args:
        ttl_seconds:           Session idle lifetime; refreshed on every access.
                               ``None`` disables expiry.
        max_context_tokens:    Default token budget used by ``build_context`` and
                               the threshold above which compaction kicks in.
        llm:                   Optional LLM used to summarise overflow turns during
                               compaction.  When ``None``, compaction falls back to
                               a terse textual concatenation (no LLM call).
    """

    def __init__(
        self,
        *,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        llm: LLMBase | None = None,
    ) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds
        self._max_context_tokens = max_context_tokens
        self._llm = llm
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
        """Create a session and return its id.

        If ``session_id`` is supplied and already exists it is returned as-is
        (idempotent resume); otherwise a new session is created.
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
        """Return the (non-expired) session state, or ``None`` if absent/expired."""
        async with self._lock:
            return self._get_live_locked(session_id)

    async def end_session(self, session_id: str) -> None:
        """Drop a session and its working context."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        role: MemoryRole | str,
        content: str,
    ) -> None:
        """Append a conversational turn, lazily creating the session if needed."""
        if not content:
            return
        async with self._lock:
            state = self._ensure_locked(session_id)
            state.messages.append(
                MemoryMessage(
                    role=MemoryRole(role),
                    content=content,
                    token_estimate=estimate_tokens(content),
                )
            )
            self._touch_locked(state)

    async def append_retrievals(
        self,
        session_id: str,
        items: list[RetrievedItem],
    ) -> None:
        """Record retrieved evidence for the session (deduped by ``ref_id``)."""
        if not items:
            return
        async with self._lock:
            state = self._ensure_locked(session_id)
            seen = {r.ref_id for r in state.retrievals if r.ref_id is not None}
            for item in items:
                if item.ref_id is not None and item.ref_id in seen:
                    continue
                state.retrievals.append(item)
                if item.ref_id is not None:
                    seen.add(item.ref_id)
            self._touch_locked(state)

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    async def build_context(
        self,
        *,
        session_id: str,
        query: str,
        token_budget: int | None = None,
    ) -> list[ChatMessage]:
        """Assemble the message list for the next LLM call within ``token_budget``.

        Newest turns are kept verbatim; once the transcript exceeds the budget,
        the oldest overflow turns are folded into the session ``summary`` and a
        single system message carrying that summary is prepended.  The selection
        decision (budget, included, summarised) is recorded on the session
        metadata under ``last_context``.

        Returns an empty list for an unknown/expired session so callers can fall
        back to stateless behaviour.
        """
        budget = token_budget or self._max_context_tokens
        async with self._lock:
            state = self._get_live_locked(session_id)
            if state is None:
                return []

            # Reserve room for the summary header so the running summary never
            # crowds out the live transcript entirely.
            summary_reserve = estimate_tokens(state.summary) if state.summary else 0
            transcript_budget = max(budget - summary_reserve, budget // 2)

            # Walk newest-first, keeping messages that fit; the rest overflow.
            kept_rev: list[MemoryMessage] = []
            running = 0
            overflow: list[MemoryMessage] = []
            for msg in reversed(state.messages):
                cost = msg.token_estimate or estimate_tokens(msg.content)
                if running + cost <= transcript_budget or not kept_rev:
                    # Always keep at least the most recent turn (the live query).
                    kept_rev.append(msg)
                    running += cost
                else:
                    overflow.append(msg)

            overflow.reverse()  # back to chronological order
            kept = list(reversed(kept_rev))

            if overflow:
                await self._compact_into_summary_locked(state, overflow)
                # Drop the compacted turns from the live transcript.
                state.messages = kept

            state.metadata["last_context"] = {
                "budget": budget,
                "included_messages": len(kept),
                "summarized_messages": len(overflow),
                "has_summary": state.summary is not None,
            }
            self._touch_locked(state)

            context: list[ChatMessage] = []
            if state.summary:
                context.append(
                    {
                        "role": "system",
                        "content": (
                            "Summary of earlier conversation in this session:\n"
                            f"{state.summary}"
                        ),
                    }
                )
            for msg in kept:
                context.append({"role": msg.role.value, "content": msg.content})
            return context

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _compact_into_summary_locked(
        self,
        state: SessionState,
        overflow: list[MemoryMessage],
    ) -> None:
        """Fold ``overflow`` turns into ``state.summary`` (LLM or textual fallback)."""
        transcript = "\n".join(
            f"[{m.role.value}] {m.content[:500]}" for m in overflow
        )
        prior = f"Existing summary:\n{state.summary}\n\n" if state.summary else ""

        summary: str | None = None
        if self._llm is not None:
            try:
                result = await self._llm.complete(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Update the running summary of a conversation so it preserves "
                                "all key facts, decisions, retrieved evidence, and conclusions. "
                                "Be terse; output only the updated summary.\n\n"
                                f"{prior}New turns to fold in:\n{transcript}"
                            ),
                        }
                    ]
                )
                summary = result.get("content") or None
            except Exception:
                summary = None  # fall back below

        if summary is None:
            # No LLM (or it failed): keep a bounded textual trail.
            combined = (state.summary + "\n" if state.summary else "") + transcript
            summary = combined[-2000:]

        state.summary = summary

    # ------------------------------------------------------------------
    # Internal helpers (assume the lock is held)
    # ------------------------------------------------------------------

    def _ensure_locked(self, session_id: str) -> SessionState:
        state = self._get_live_locked(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
        return state

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
