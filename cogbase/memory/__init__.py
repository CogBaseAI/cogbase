"""Memory layer for CogBase.

The short-term tier and the episodic event writer are implemented.  Short-term
memory is being refactored to ride on the episodic log (docs/memory.md
build-order step 5); long-term memory and a unifying ``MemoryManager`` are
planned.
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
    RetrievalKind,
    RetrievalResultPayload,
    RetrievedItem,
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
    "RetrievalKind",
    "RetrievedItem",
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
