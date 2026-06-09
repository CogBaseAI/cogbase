"""Unit tests for cogbase.memory.short_term.ShortTermMemory.

ShortTermMemory is a projection over the episodic log: it has no store of its
own, so these tests back it with a real ``EpisodicMemory`` over a local-fs log
store and seed the session's history by recording continuity events the way the
query runner would.
"""

from __future__ import annotations

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
    sid = await mem.start_session(app_id="acme", user_id="u1")
    assert sid

    # Resume with the same id is idempotent (no new session created).
    same = await mem.start_session(session_id=sid)
    assert same == sid
    state = await mem.get(sid)
    assert state is not None
    assert state.app_id == "acme"
    assert state.user_id == "u1"


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
