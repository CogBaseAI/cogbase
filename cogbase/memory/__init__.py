"""Memory layer for CogBase.

The short-term tier and the episodic event writer are implemented; short-term
memory rides on the episodic log as a projection over it (rehydrate, append-a-
``session_compacted`` compaction, near-pass-through assembly).  The long-term
tier — the ``LongTermMemory`` recall/reconcile/promote service and the offline
``Distiller`` that promotes durable records out of session logs — is implemented
in ``long_term.py`` / ``distill.py``.  A unifying ``MemoryManager`` is planned.
"""

from cogbase.memory.distill import Distiller
from cogbase.memory.episodic import EpisodicMemory
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import (
    CONTINUITY_EVENT_TYPES,
    EventRef,
    EventType,
    FeedbackPayload,
    FinalAnswerPayload,
    LongTermRecord,
    MemoryCandidate,
    MemoryEvent,
    MemoryKind,
    MemoryMessage,
    MemoryRole,
    MemoryStatus,
    ReconcileOp,
    ReviewDecision,
    ReviewOutcome,
    ReviewResult,
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
    # long-term
    "LongTermMemory",
    "Distiller",
    "LongTermRecord",
    "MemoryCandidate",
    "MemoryKind",
    "MemoryStatus",
    "ReconcileOp",
    "ReviewDecision",
    "ReviewOutcome",
    "ReviewResult",
]
