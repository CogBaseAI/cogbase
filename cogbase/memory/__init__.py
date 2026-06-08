"""Memory layer for CogBase.

The short-term tier and the episodic event writer are implemented; short-term
memory rides on the episodic log as a projection over it (rehydrate, append-a-
``session_compacted`` compaction, near-pass-through assembly).  Long-term memory
and a unifying ``MemoryManager`` are planned.
"""

from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.models import (
    CONTINUITY_EVENT_TYPES,
    EventRef,
    EventType,
    FeedbackPayload,
    FinalAnswerPayload,
    MemoryEvent,
    MemoryMessage,
    MemoryRole,
    RetrievalHit,
    RetrievalResultPayload,
    SessionCompactedPayload,
    SessionStartedPayload,
    SessionState,
    ToolCalledPayload,
    ToolResultPayload,
    UserMessagePayload,
)
from cogbase.memory.short_term import ShortTermMemory, estimate_tokens

__all__ = [
    # short-term
    "MemoryMessage",
    "MemoryRole",
    "SessionState",
    "ShortTermMemory",
    "estimate_tokens",
    # episodic
    "EpisodicMemory",
    "MemoryEvent",
    "EventType",
    "EventRef",
    "CONTINUITY_EVENT_TYPES",
    "SessionStartedPayload",
    "UserMessagePayload",
    "FinalAnswerPayload",
    "SessionCompactedPayload",
    "ToolCalledPayload",
    "ToolResultPayload",
    "RetrievalResultPayload",
    "RetrievalHit",
    "FeedbackPayload",
]
