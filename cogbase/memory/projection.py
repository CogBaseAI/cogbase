"""Shared projection of an episodic log into a conversational thread.

Both consumers of the per-session log project it into the same continuity
thread before reasoning over it: short-term memory rehydrates the working
context (dropping turns a ``session_compacted`` summary already covers), and the
distiller reads the *whole* thread to extract durable memories.  This module
holds the one projection both use, so the rule for what counts as a
conversational turn lives in a single place (the plan's open question on where
the shared helper lives — docs/long-term-memory-implementation-plan.md).

The projection mirrors the log's single-writer / append-safety guarantees: a
``seq`` is kept on first occurrence, so an out-of-order straggler that reuses a
``seq`` never displaces the active writer's event (see
docs/episodic-memory.md#single-writer-and-append-safety).
"""

from __future__ import annotations

from cogbase.llms.compaction import estimate_tokens
from cogbase.memory.models import (
    EventType,
    MemoryEvent,
    MemoryMessage,
    MemoryRole,
)

# Continuity events threaded into the conversation, mapped to the role they
# project to.  Tool calls/results are intra-turn scratch and never threaded.
_CONTINUITY_ROLE: dict[EventType, MemoryRole] = {
    EventType.USER_MESSAGE: MemoryRole.USER,
    EventType.FINAL_ANSWER: MemoryRole.ASSISTANT,
}


def continuity_role(event_type: EventType) -> MemoryRole | None:
    """Return the thread role an event projects to, or ``None`` if it is scratch.

    The single authority for "what counts as a conversational turn", shared so an
    incremental folder (short-term's projection cache) classifies events exactly
    as :func:`project_thread` does.
    """
    return _CONTINUITY_ROLE.get(event_type)


def message_from_event(event: MemoryEvent) -> MemoryMessage:
    """Project one continuity event into a :class:`MemoryMessage`.

    The caller must have established the event is a continuity turn (a non-``None``
    :func:`continuity_role`).  Token cost is estimated once here so context
    assembly never re-estimates a projected turn — matching :func:`project_thread`.
    """
    text = event.payload.get("text", "")
    return MemoryMessage(
        role=_CONTINUITY_ROLE[event.event_type],
        content=text,
        seq=event.seq,
        token_estimate=estimate_tokens(text),
    )


def latest_compaction(events: list[MemoryEvent]) -> tuple[str | None, int]:
    """Return the latest ``session_compacted`` summary and the seq it covers.

    ``events`` are in log order; the last compaction wins.  Returns
    ``(None, -1)`` when the session has never been compacted.
    """
    summary: str | None = None
    replaces_through = -1
    for event in events:
        if event.event_type is EventType.SESSION_COMPACTED:
            summary = event.payload.get("summary")
            replaces_through = int(event.payload.get("replaces_through", -1))
    return summary, replaces_through


def project_thread(
    events: list[MemoryEvent], *, since_seq: int = -1
) -> list[MemoryMessage]:
    """Project the continuity events into an ordered list of turns.

    Only ``user_message`` / ``final_answer`` events become turns; everything
    after ``since_seq`` is included.  Pass the prior summary's
    ``replaces_through`` as ``since_seq`` to drop turns already folded into a
    summary (short-term rehydrate); leave the default ``-1`` to project the whole
    thread (distillation, which reads every turn the log still holds).
    """
    messages: list[MemoryMessage] = []
    seen_seqs: set[int] = set()
    for event in events:
        if (
            continuity_role(event.event_type) is None
            or event.seq <= since_seq
            or event.seq in seen_seqs
        ):
            continue
        seen_seqs.add(event.seq)
        messages.append(message_from_event(event))
    return messages
