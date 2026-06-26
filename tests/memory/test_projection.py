"""Unit tests for cogbase.memory.projection.

These cover the two pure helpers that turn an episodic event list into a
conversational thread: ``latest_compaction`` (find the running summary and the
seq it covers) and ``project_thread`` (project continuity events into turns).
Both are synchronous and store-free, so the tests build ``MemoryEvent`` objects
directly rather than recording through ``EpisodicMemory``.
"""

from __future__ import annotations

from cogbase.llms.summarization import estimate_tokens
from cogbase.memory.models import EventType, MemoryEvent, MemoryRole
from cogbase.memory.projection import (
    latest_compaction,
    latest_distillation,
    project_thread,
)


def _event(event_type: EventType, seq: int, **payload) -> MemoryEvent:
    return MemoryEvent(
        session_id="s1",
        seq=seq,
        ulid=f"ulid-{seq}",
        event_type=event_type,
        payload=payload,
    )


def _user(seq: int, text: str) -> MemoryEvent:
    return _event(EventType.USER_MESSAGE, seq, text=text)


def _answer(seq: int, text: str) -> MemoryEvent:
    return _event(EventType.FINAL_ANSWER, seq, text=text)


def _compaction(seq: int, summary: str, replaces_through: int) -> MemoryEvent:
    return _event(
        EventType.SESSION_COMPACTED,
        seq,
        summary=summary,
        replaces_through=replaces_through,
    )


# ---------------------------------------------------------------------------
# latest_compaction
# ---------------------------------------------------------------------------

def test_latest_compaction_none_when_never_compacted():
    events = [_user(0, "hi"), _answer(1, "hello")]
    assert latest_compaction(events) == (None, -1)


def test_latest_compaction_empty_log():
    assert latest_compaction([]) == (None, -1)


def test_latest_compaction_returns_summary_and_watermark():
    events = [_user(0, "hi"), _answer(1, "hello"), _compaction(2, "SUMMARY", 1)]
    assert latest_compaction(events) == ("SUMMARY", 1)


def test_latest_compaction_last_one_wins():
    events = [
        _compaction(2, "OLD", 1),
        _user(3, "more"),
        _answer(4, "ok"),
        _compaction(5, "NEW", 4),
    ]
    assert latest_compaction(events) == ("NEW", 4)


def test_latest_compaction_defaults_replaces_through_when_missing():
    # A malformed compaction payload missing replaces_through falls back to -1.
    event = _event(EventType.SESSION_COMPACTED, 2, summary="S")
    assert latest_compaction([event]) == ("S", -1)


# ---------------------------------------------------------------------------
# latest_distillation
# ---------------------------------------------------------------------------

def _distillation(seq: int, distilled_through: int) -> MemoryEvent:
    return _event(
        EventType.SESSION_DISTILLED, seq, distilled_through=distilled_through
    )


def test_latest_distillation_minus_one_when_never_distilled():
    events = [_user(0, "hi"), _answer(1, "hello")]
    assert latest_distillation(events) == -1


def test_latest_distillation_empty_log():
    assert latest_distillation([]) == -1


def test_latest_distillation_returns_watermark():
    events = [_user(0, "hi"), _answer(1, "hello"), _distillation(2, 1)]
    assert latest_distillation(events) == 1


def test_latest_distillation_takes_the_highest_watermark():
    # Monotonic in practice, but an out-of-order straggler must not drag it back.
    events = [_distillation(2, 3), _distillation(5, 1)]
    assert latest_distillation(events) == 3


def test_latest_distillation_defaults_when_missing():
    event = _event(EventType.SESSION_DISTILLED, 2)  # no distilled_through
    assert latest_distillation([event]) == -1


# ---------------------------------------------------------------------------
# project_thread
# ---------------------------------------------------------------------------

def test_project_thread_empty():
    assert project_thread([]) == []


def test_project_thread_maps_roles_and_preserves_order():
    events = [_user(0, "q1"), _answer(1, "a1"), _user(2, "q2"), _answer(3, "a2")]
    messages = project_thread(events)

    assert [m.role for m in messages] == [
        MemoryRole.USER,
        MemoryRole.ASSISTANT,
        MemoryRole.USER,
        MemoryRole.ASSISTANT,
    ]
    assert [m.content for m in messages] == ["q1", "a1", "q2", "a2"]
    assert [m.seq for m in messages] == [0, 1, 2, 3]


def test_project_thread_fills_token_estimate():
    messages = project_thread([_user(0, "some text here")])
    assert messages[0].token_estimate == estimate_tokens("some text here")


def test_project_thread_ignores_non_continuity_events():
    events = [
        _user(0, "q1"),
        _event(EventType.TOOL_CALLED, 1, name="search"),
        _event(EventType.TOOL_RESULT, 2, ok=True),
        _event(EventType.RETRIEVAL_RESULT, 3, collection="c"),
        _answer(4, "a1"),
    ]
    messages = project_thread(events)
    assert [m.content for m in messages] == ["q1", "a1"]


def test_project_thread_since_seq_drops_covered_turns():
    events = [_user(0, "q1"), _answer(1, "a1"), _user(2, "q2"), _answer(3, "a2")]
    messages = project_thread(events, since_seq=1)
    assert [m.seq for m in messages] == [2, 3]
    assert [m.content for m in messages] == ["q2", "a2"]


def test_project_thread_since_seq_is_exclusive():
    # since_seq itself is dropped; the next seq survives.
    events = [_user(0, "q1"), _answer(1, "a1")]
    assert [m.seq for m in project_thread(events, since_seq=0)] == [1]


def test_project_thread_dedupes_on_first_seq():
    # An out-of-order straggler reusing a seq must not displace the first event.
    events = [_user(0, "original"), _user(0, "straggler")]
    messages = project_thread(events)
    assert len(messages) == 1
    assert messages[0].content == "original"


def test_project_thread_missing_text_becomes_empty():
    event = _event(EventType.USER_MESSAGE, 0)  # no text in payload
    messages = project_thread([event])
    assert messages[0].content == ""
