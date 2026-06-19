"""Unit tests for cogbase.memory.short_term.ShortTermMemory.

ShortTermMemory is a projection over the episodic log: it has no store of its
own, so these tests back it with a real ``EpisodicMemory`` over a local-fs log
store and seed the session's history by recording continuity events the way the
query runner would.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.memory import EpisodicMemory, EventType, MemoryRole, ShortTermMemory
from cogbase.memory.short_term import estimate_tokens
from cogbase.stores.log.local_fs import LocalFSLogStore


def _summarizing_llm(summary: str = "SUMMARY") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": summary, "tool_calls": None})
    # A real context window; otherwise MagicMock's int() defaults to 1, collapsing
    # the summariser's chunk budget to 1 token and fanning the transcript into
    # hundreds of single-token chunks.
    llm.context_window = MagicMock(return_value=128_000)
    return llm


@pytest.fixture
def episodic(tmp_path) -> EpisodicMemory:
    return EpisodicMemory(LocalFSLogStore(tmp_path))


async def _seed_turn(ep: EpisodicMemory, session_id: str, question: str, answer: str) -> None:
    """Record + durably flush one user/assistant turn, as the runner would."""
    await ep.record_user_message(session_id=session_id, content=question)
    await ep.record_final_answer(session_id=session_id, answer=answer)
    await ep.flush(session_id)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_session_returns_id_and_is_resumable(episodic):
    mem = ShortTermMemory(episodic=episodic)
    sid = await mem.start_session(app_id="acme")
    assert sid

    # Resume with the same id is idempotent (no new session created).
    same = await mem.start_session(session_id=sid)
    assert same == sid
    state = await mem.get(sid)
    assert state is not None
    assert state.app_id == "acme"


@pytest.mark.asyncio
async def test_get_unknown_session_returns_none(episodic):
    mem = ShortTermMemory(episodic=episodic)
    assert await mem.get("nope") is None


@pytest.mark.asyncio
async def test_end_session_drops_cache(episodic):
    mem = ShortTermMemory(episodic=episodic)
    sid = await mem.start_session()
    await mem.end_session(sid)
    # No cached metadata and no log history → gone.
    assert await mem.get(sid) is None


# ---------------------------------------------------------------------------
# TTL / expiry (of the metadata cache)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_session_is_not_returned(episodic):
    mem = ShortTermMemory(episodic=episodic, ttl_seconds=3600)
    sid = await mem.start_session()
    state = await mem.get(sid)
    state.expires_at = state.expires_at - timedelta(hours=2)
    # Expired cache entry with no durable history → None.
    assert await mem.get(sid) is None


@pytest.mark.asyncio
async def test_ttl_none_never_expires(episodic):
    mem = ShortTermMemory(episodic=episodic, ttl_seconds=None)
    sid = await mem.start_session()
    state = await mem.get(sid)
    assert state.expires_at is None


# ---------------------------------------------------------------------------
# Projection / rehydrate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_rehydrates_thread_from_log(episodic):
    mem = ShortTermMemory(episodic=episodic)
    await _seed_turn(episodic, "s1", "what is the capital of France?", "Paris.")

    state = await mem.get("s1")
    assert state is not None
    assert [(m.role, m.content) for m in state.messages] == [
        (MemoryRole.USER, "what is the capital of France?"),
        (MemoryRole.ASSISTANT, "Paris."),
    ]
    # seq is carried from the source events.
    assert [m.seq for m in state.messages] == [0, 1]


@pytest.mark.asyncio
async def test_build_context_no_history_returns_current_message(episodic):
    mem = ShortTermMemory(episodic=episodic)
    ctx = await mem.build_context(session_id="fresh", current_user_message="hi")
    assert ctx == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_build_context_threads_prior_turns_plus_current(episodic):
    mem = ShortTermMemory(episodic=episodic)
    await _seed_turn(episodic, "s1", "first?", "answer-one")

    ctx = await mem.build_context(session_id="s1", current_user_message="second?")
    assert [m["content"] for m in ctx] == ["first?", "answer-one", "second?"]
    assert [m["role"] for m in ctx] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_only_continuity_events_are_threaded(episodic):
    # Tool calls/results are intra-turn scratch and must not be rehydrated.
    await episodic.record_user_message(session_id="s1", content="use a tool")
    await episodic.record_tool_call(
        session_id="s1", tool_call_id="t1", name="vector_search", arguments={"q": "x"}
    )
    await episodic.record_tool_result(session_id="s1", tool_call_id="t1", result="hits")
    await episodic.record_final_answer(session_id="s1", answer="done")
    await episodic.flush("s1")

    mem = ShortTermMemory(episodic=episodic)
    ctx = await mem.build_context(session_id="s1", current_user_message="next")
    assert [m["content"] for m in ctx] == ["use a tool", "done", "next"]


# ---------------------------------------------------------------------------
# Incremental projection cache (re-read only the tail since the last turn)
# ---------------------------------------------------------------------------


class _RecordingLogStore(LocalFSLogStore):
    """Local-fs store that records the offsets ``read_since`` is called with."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.read_since_offsets: list[int] = []

    async def read_since(self, log_type, log_id, offset):
        self.read_since_offsets.append(offset)
        return await super().read_since(log_type, log_id, offset)


@pytest.mark.asyncio
async def test_build_context_reads_only_the_tail_after_first_turn(tmp_path):
    store = _RecordingLogStore(tmp_path)
    ep = EpisodicMemory(store)
    mem = ShortTermMemory(episodic=ep)

    await _seed_turn(ep, "s1", "q1", "a1")
    await mem.build_context(session_id="s1", current_user_message="x")
    # Cold cache: the first rehydrate folds from offset 0 (a full read).
    assert store.read_since_offsets == [0]
    watermark = await store.size("episodic", "s1")

    await _seed_turn(ep, "s1", "q2", "a2")
    store.read_since_offsets.clear()
    await mem.build_context(session_id="s1", current_user_message="y")
    # Warm cache: it re-reads only past last turn's watermark — never from 0 again,
    # so a long session stops re-parsing its whole log every turn.
    assert store.read_since_offsets == [watermark]


@pytest.mark.asyncio
async def test_incremental_projection_matches_cold_rebuild(tmp_path):
    # A warm cache folded turn-by-turn must equal a cold full replay of the log.
    ep = EpisodicMemory(LocalFSLogStore(tmp_path))
    warm = ShortTermMemory(episodic=ep)
    for i in range(4):
        await _seed_turn(ep, "s1", f"q{i}", f"a{i}")
        await warm.build_context(session_id="s1", current_user_message="cur")

    warm_state = await warm.get("s1")
    cold_state = await ShortTermMemory(episodic=ep).get("s1")
    assert [(m.role, m.content, m.seq) for m in warm_state.messages] == [
        (m.role, m.content, m.seq) for m in cold_state.messages
    ]
    assert [(m.role, m.content) for m in warm_state.messages] == [
        (MemoryRole.USER, "q0"), (MemoryRole.ASSISTANT, "a0"),
        (MemoryRole.USER, "q1"), (MemoryRole.ASSISTANT, "a1"),
        (MemoryRole.USER, "q2"), (MemoryRole.ASSISTANT, "a2"),
        (MemoryRole.USER, "q3"), (MemoryRole.ASSISTANT, "a3"),
    ]


@pytest.mark.asyncio
async def test_rehydrate_rebuilds_after_log_shrinks(tmp_path):
    # If the log is wiped from under a warm cache (retention/erasure), the stale
    # folded turns must not survive — the smaller size signals a rebuild from 0.
    ep = EpisodicMemory(LocalFSLogStore(tmp_path))
    mem = ShortTermMemory(episodic=ep)
    await _seed_turn(ep, "s1", "q1", "a1")
    ctx1 = await mem.build_context(session_id="s1", current_user_message="x")
    assert [m["content"] for m in ctx1] == ["q1", "a1", "x"]

    await ep.delete(session_id="s1")
    ctx2 = await mem.build_context(session_id="s1", current_user_message="y")
    assert ctx2 == [{"role": "user", "content": "y"}]


# ---------------------------------------------------------------------------
# Compaction (append-a-session_compacted)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_context_compacts_overflow_into_session_compacted(episodic):
    llm = _summarizing_llm("COMPACTED")
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=llm)

    big = "x" * 200  # ~50 tokens each, exceeds the budget
    await _seed_turn(episodic, "s1", big + " one", big + " two")
    await _seed_turn(episodic, "s1", big + " three", big + " four")

    ctx = await mem.build_context(
        session_id="s1", current_user_message="latest", token_budget=20
    )

    # The running summary is prepended as a system message; the LLM was called.
    assert ctx[0]["role"] == "system"
    assert "COMPACTED" in ctx[0]["content"]
    llm.complete.assert_awaited()
    # The current turn is always the final message.
    assert ctx[-1]["content"] == "latest"

    # A session_compacted event was appended (buffered for the turn flush).
    await episodic.flush("s1")
    types = [e.event_type for e in await episodic.replay(session_id="s1")]
    assert EventType.SESSION_COMPACTED in types

    # A fresh projection honours the summary: it no longer threads the folded
    # turns verbatim, and the summary header is present.
    fresh = ShortTermMemory(episodic=episodic, llm=llm)
    ctx2 = await fresh.build_context(session_id="s1", current_user_message="again")
    assert ctx2[0]["role"] == "system" and "COMPACTED" in ctx2[0]["content"]
    assert "one" not in " ".join(m["content"] for m in ctx2[1:])


@pytest.mark.asyncio
async def test_warm_cache_honours_compaction_on_next_turn(episodic):
    # After compaction the projection cache is dropped, so the next turn on the
    # same (warm) instance rebuilds and honours the now-flushed summary rather
    # than folding new events onto a pre-compaction watermark.
    llm = _summarizing_llm("COMPACTED")
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=llm)

    big = "x" * 200
    await _seed_turn(episodic, "s1", big + " one", big + " two")
    await _seed_turn(episodic, "s1", big + " three", big + " four")

    ctx = await mem.build_context(
        session_id="s1", current_user_message="latest", token_budget=20
    )
    assert "COMPACTED" in ctx[0]["content"]
    await episodic.flush("s1")  # the turn flush lands the session_compacted

    ctx2 = await mem.build_context(
        session_id="s1", current_user_message="again", token_budget=20
    )
    assert ctx2[0]["role"] == "system" and "COMPACTED" in ctx2[0]["content"]
    # The folded turns are not re-threaded verbatim.
    assert "one" not in " ".join(m["content"] for m in ctx2[1:])


@pytest.mark.asyncio
async def test_no_compaction_without_llm(episodic):
    # No LLM → no compaction; the full thread is assembled as-is.
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=None)
    big = "y" * 200
    await _seed_turn(episodic, "s1", big + " old", big + " older")

    ctx = await mem.build_context(session_id="s1", current_user_message="new")
    assert all(m["role"] != "system" for m in ctx)  # no summary
    assert ctx[-1]["content"] == "new"
    # Nothing was compacted into the log.
    assert not episodic.has_pending("s1")


@pytest.mark.asyncio
async def test_compaction_failure_serves_full_thread(episodic):
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("boom"))
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=llm)
    big = "x" * 200
    await _seed_turn(episodic, "s1", big + " old", "recent")

    # A failed summarisation must not crash build_context, and must not drop the
    # turns it failed to summarise (no covering summary persisted).
    ctx = await mem.build_context(session_id="s1", current_user_message="new")
    assert ctx[-1]["content"] == "new"
    assert all(m["role"] != "system" for m in ctx)
    assert not episodic.has_pending("s1")  # no session_compacted appended


# ---------------------------------------------------------------------------
# Per-session locking (a slow compaction must not stall other sessions)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slow_compaction_does_not_block_other_sessions(episodic):
    # The summary LLM call must run *outside* the per-session lock, and the lock
    # is per session — so one session stuck mid-summary cannot stall another's
    # context build.
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_complete(*_a, **_k):
        entered.set()
        await release.wait()
        return {"content": "COMPACTED", "tool_calls": None}

    llm = _summarizing_llm()
    llm.complete = AsyncMock(side_effect=blocking_complete)
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=llm)

    big = "x" * 200  # ~50 tokens each, over the budget → session "slow" compacts
    await _seed_turn(episodic, "slow", big + " one", big + " two")
    await _seed_turn(episodic, "slow", big + " three", big + " four")

    slow = asyncio.create_task(
        mem.build_context(session_id="slow", current_user_message="a", token_budget=20)
    )
    # Wait until "slow" is parked inside the LLM — at which point its session lock
    # is released (the summary runs outside it).
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert not slow.done()

    # A different session builds context to completion while "slow" is still
    # blocked. With a global lock this would deadlock until release.
    ctx_other = await asyncio.wait_for(
        mem.build_context(session_id="other", current_user_message="b"), timeout=1
    )
    assert ctx_other[-1]["content"] == "b"

    release.set()
    ctx_slow = await asyncio.wait_for(slow, timeout=1)
    assert "COMPACTED" in ctx_slow[0]["content"]


@pytest.mark.asyncio
async def test_commit_rechecks_log_and_skips_double_compaction(episodic):
    # If a compaction lands (and flushes) while the LLM summary is running, the
    # recheck in _commit_compaction must honour it rather than fold a second time.
    async def inject_then_summarise(*_a, **_k):
        # Simulate a concurrent compaction covering the whole thread landing
        # durably while this summary call is in flight.
        await episodic.record_compaction(
            session_id="s1", summary="OTHER", replaces_through=999, token_stats={}
        )
        await episodic.flush("s1")
        return {"content": "MINE", "tool_calls": None}

    llm = _summarizing_llm()
    llm.complete = AsyncMock(side_effect=inject_then_summarise)
    mem = ShortTermMemory(episodic=episodic, compaction_token_budget=20, llm=llm)

    big = "x" * 200
    await _seed_turn(episodic, "s1", big + " one", big + " two")
    await _seed_turn(episodic, "s1", big + " three", big + " four")

    ctx = await mem.build_context(
        session_id="s1", current_user_message="latest", token_budget=20
    )

    # The already-landed summary wins; this build does not append its own.
    assert "OTHER" in ctx[0]["content"]
    assert all("MINE" not in m["content"] for m in ctx)
    assert not episodic.has_pending("s1")  # no second session_compacted buffered
    compactions = [
        e for e in await episodic.replay(session_id="s1")
        if e.event_type is EventType.SESSION_COMPACTED
    ]
    assert len(compactions) == 1
