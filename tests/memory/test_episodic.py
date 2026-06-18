"""Tests for the episodic-memory event writer."""

import pytest

from cogbase.memory import EpisodicMemory, EventRef, EventType, MemoryEvent
from cogbase.stores.log.base import LogFenced
from cogbase.stores.log.local_fs import LocalFSLogStore

SESSION = "session-abc"


@pytest.fixture
def episodic(tmp_path):
    return EpisodicMemory(LocalFSLogStore(tmp_path))


# -- stamping & buffering ---------------------------------------------------


async def test_record_stamps_seq_and_ulid(episodic):
    ref = await episodic.record_user_message(session_id=SESSION, content="hi")
    assert ref.session_id == SESSION
    assert ref.seq == 0
    assert ref.ulid  # stamped


async def test_seq_is_monotonic_within_session(episodic):
    refs = [
        await episodic.record_user_message(session_id=SESSION, content=f"m{i}")
        for i in range(3)
    ]
    assert [r.seq for r in refs] == [0, 1, 2]
    assert len({r.ulid for r in refs}) == 3  # ulids are unique


async def test_seq_is_per_session(episodic):
    a = await episodic.record_user_message(session_id="s-a", content="x")
    b = await episodic.record_user_message(session_id="s-b", content="y")
    assert a.seq == 0 and b.seq == 0


async def test_records_buffer_until_flush(episodic):
    await episodic.record_user_message(session_id=SESSION, content="hi")
    assert episodic.has_pending(SESSION)
    # Nothing is durable yet.
    assert await episodic.replay(session_id=SESSION) == []
    await episodic.flush(SESSION)
    assert not episodic.has_pending(SESSION)
    events = await episodic.replay(session_id=SESSION)
    assert [e.event_type for e in events] == [EventType.USER_MESSAGE]


async def test_flush_empty_buffer_is_noop(episodic):
    await episodic.flush(SESSION)  # no records yet
    assert await episodic.replay(session_id=SESSION) == []


async def test_flush_writes_a_turn_as_one_append(episodic, tmp_path):
    await episodic.record_user_message(session_id=SESSION, content="q")
    await episodic.record_tool_call(
        session_id=SESSION, tool_call_id="t1", name="vector_search", arguments={"q": "x"}
    )
    await episodic.record_final_answer(session_id=SESSION, answer="a")
    await episodic.flush(SESSION)

    raw = (tmp_path / "episodic" / SESSION).read_text()
    assert raw.count("\n") == 3  # three NDJSON lines, one append


# -- read-back --------------------------------------------------------------


async def test_replay_returns_events_in_order(episodic):
    await episodic.record_user_message(session_id=SESSION, content="q")
    await episodic.record_final_answer(session_id=SESSION, answer="a")
    await episodic.flush(SESSION)

    events = await episodic.replay(session_id=SESSION)
    assert [e.seq for e in events] == [0, 1]
    assert events[0].payload["text"] == "q"
    assert events[1].payload["text"] == "a"


async def test_tail_returns_last_n(episodic):
    for i in range(5):
        await episodic.record_user_message(session_id=SESSION, content=f"m{i}")
    await episodic.flush(SESSION)

    tail = await episodic.tail(session_id=SESSION, limit=2)
    assert [e.seq for e in tail] == [3, 4]


async def test_replay_dedupes_by_ulid(episodic, tmp_path):
    # Simulate a retried append that double-wrote the same line.
    event = MemoryEvent(
        session_id=SESSION, seq=0, ulid="01J0SAMEULID0000000000000", event_type=EventType.USER_MESSAGE
    )
    line = event.to_ndjson()
    store = LocalFSLogStore(tmp_path)
    await store.append("episodic", SESSION, [line, line])

    events = await episodic.replay(session_id=SESSION)
    assert len(events) == 1


async def test_replay_skips_unparseable_lines(episodic, tmp_path):
    store = LocalFSLogStore(tmp_path)
    good = MemoryEvent(
        session_id=SESSION, seq=0, ulid="01J0GOODULID00000000000000", event_type=EventType.USER_MESSAGE
    ).to_ndjson()
    await store.append("episodic", SESSION, ["{not json", good])

    events = await episodic.replay(session_id=SESSION)
    assert len(events) == 1


# -- cold-start seq recovery ------------------------------------------------


async def test_seq_resumes_from_log_on_cold_start(tmp_path):
    first = EpisodicMemory(LocalFSLogStore(tmp_path))
    await first.record_user_message(session_id=SESSION, content="m0")
    await first.record_user_message(session_id=SESSION, content="m1")
    await first.flush(SESSION)

    # A fresh writer (new process) with empty in-memory state continues the seq.
    second = EpisodicMemory(LocalFSLogStore(tmp_path))
    ref = await second.record_user_message(session_id=SESSION, content="m2")
    assert ref.seq == 2

    await second.flush(SESSION)
    events = await second.replay(session_id=SESSION)
    assert [e.seq for e in events] == [0, 1, 2]


# -- dual-writer fencing ----------------------------------------------------


async def test_concurrent_cold_start_owner_is_fenced_on_flush(tmp_path):
    # Two processes cold-start the same session and both compute next seq=0.
    store = LocalFSLogStore(tmp_path)
    owner = EpisodicMemory(store)
    intruder = EpisodicMemory(store)

    a = await owner.record_user_message(session_id=SESSION, content="from-owner")
    b = await intruder.record_user_message(session_id=SESSION, content="from-intruder")
    assert a.seq == b.seq == 0  # the seq collision the offset must catch

    await owner.flush(SESSION)  # owner lands first, advancing the log

    # The intruder's offset is now stale; its flush is fenced, not retried into
    # success — so its colliding seq=0 event never reaches the log.
    with pytest.raises(LogFenced):
        await intruder.flush(SESSION)

    events = await owner.replay(session_id=SESSION)
    assert len(events) == 1
    assert events[0].payload["text"] == "from-owner"


async def test_fenced_writer_relinquishes_and_can_re_resume(tmp_path):
    store = LocalFSLogStore(tmp_path)
    owner = EpisodicMemory(store)
    intruder = EpisodicMemory(store)

    await intruder.record_user_message(session_id=SESSION, content="doomed")
    await owner.record_user_message(session_id=SESSION, content="owner-0")
    await owner.flush(SESSION)

    with pytest.raises(LogFenced):
        await intruder.flush(SESSION)

    # After fencing, the deposed writer dropped its session state: a fresh record
    # re-resolves seq + offset from the log, so it continues cleanly (no longer
    # colliding) if affinity legitimately returns the session to it.
    ref = await intruder.record_user_message(session_id=SESSION, content="intruder-1")
    assert ref.seq == 1  # resumed from the owner's seq=0, not a stale 0
    await intruder.flush(SESSION)

    events = await owner.replay(session_id=SESSION)
    assert [e.payload["text"] for e in events] == ["owner-0", "intruder-1"]


# -- app attribution inheritance & payload contracts -------------------------


async def test_session_app_id_is_inherited_by_later_events(episodic):
    await episodic.record_session_started(session_id=SESSION, app_id="legal")
    await episodic.record_user_message(session_id=SESSION, content="q")
    await episodic.flush(SESSION)

    events = await episodic.replay(session_id=SESSION)
    user_msg = next(e for e in events if e.event_type == EventType.USER_MESSAGE)
    assert user_msg.app_id == "legal"


async def test_tool_result_marks_ok_and_error(episodic):
    await episodic.record_tool_result(session_id=SESSION, tool_call_id="t1", result={"n": 1})
    await episodic.record_tool_result(session_id=SESSION, tool_call_id="t2", error="boom")
    await episodic.flush(SESSION)

    events = await episodic.replay(session_id=SESSION)
    assert events[0].payload["ok"] is True
    assert events[1].payload["ok"] is False
    assert events[1].payload["error"] == "boom"


async def test_parent_event_id_threads_a_causal_chain(episodic):
    call_ref = await episodic.record_tool_call(
        session_id=SESSION, tool_call_id="t1", name="vector_search", arguments={}
    )
    res_ref = await episodic.record_tool_result(
        session_id=SESSION, tool_call_id="t1", result={}, parent_event_id=call_ref
    )
    await episodic.flush(SESSION)

    events = await episodic.replay(session_id=SESSION)
    result_event = next(e for e in events if e.seq == res_ref.seq)
    assert result_event.parent_event_id is not None
    assert result_event.parent_event_id.seq == call_ref.seq


async def test_final_answer_carries_cited_triplets(episodic):
    cite = EventRef(session_id=SESSION, seq=0, ulid="01J0CITED00000000000000000")
    await episodic.record_final_answer(session_id=SESSION, answer="a", cited_ids=[cite])
    await episodic.flush(SESSION)

    events = await episodic.replay(session_id=SESSION)
    assert events[0].payload["cited_ids"][0]["seq"] == 0


async def test_pending_continuity_flags_only_continuity_events(episodic):
    await episodic.record_tool_call(
        session_id=SESSION, tool_call_id="t1", name="x", arguments={}
    )
    assert episodic.has_pending(SESSION)
    assert not episodic.pending_continuity(SESSION)

    await episodic.record_final_answer(session_id=SESSION, answer="a")
    assert episodic.pending_continuity(SESSION)


# -- deletion ---------------------------------------------------------------


async def test_delete_removes_log_and_state(episodic):
    await episodic.record_user_message(session_id=SESSION, content="q")
    await episodic.flush(SESSION)
    await episodic.delete(session_id=SESSION)

    assert await episodic.replay(session_id=SESSION) == []
    # seq restarts after a delete (fresh stream).
    ref = await episodic.record_user_message(session_id=SESSION, content="new")
    assert ref.seq == 0
