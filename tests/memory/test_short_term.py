"""Unit tests for cogbase.memory.short_term.ShortTermMemory."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.memory import MemoryRole, RetrievalKind, RetrievedItem, ShortTermMemory
from cogbase.memory.short_term import estimate_tokens


def _summarizing_llm(summary: str = "SUMMARY") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": summary, "tool_calls": None})
    return llm


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_session_returns_id_and_is_resumable():
    mem = ShortTermMemory()
    sid = await mem.start_session(app_name="acme", user_id="u1")
    assert sid

    # Resume with the same id is idempotent (no new session created).
    same = await mem.start_session(session_id=sid)
    assert same == sid
    state = await mem.get(sid)
    assert state is not None
    assert state.app_name == "acme"
    assert state.user_id == "u1"


@pytest.mark.asyncio
async def test_append_message_lazily_creates_session():
    mem = ShortTermMemory()
    await mem.append_message("sess-x", MemoryRole.USER, "hello")
    state = await mem.get("sess-x")
    assert state is not None
    assert len(state.messages) == 1
    assert state.messages[0].role is MemoryRole.USER
    assert state.messages[0].content == "hello"
    assert state.messages[0].token_estimate == estimate_tokens("hello")


@pytest.mark.asyncio
async def test_empty_message_is_ignored():
    mem = ShortTermMemory()
    await mem.append_message("s", MemoryRole.USER, "")
    state = await mem.get("s")
    assert state is None  # nothing created for an empty turn


@pytest.mark.asyncio
async def test_end_session_drops_state():
    mem = ShortTermMemory()
    sid = await mem.start_session()
    await mem.end_session(sid)
    assert await mem.get(sid) is None


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_session_is_not_returned():
    mem = ShortTermMemory(ttl_seconds=3600)
    sid = await mem.start_session()
    state = await mem.get(sid)
    # Force expiry into the past.
    state.expires_at = state.expires_at - timedelta(hours=2)
    assert await mem.get(sid) is None


@pytest.mark.asyncio
async def test_ttl_none_never_expires():
    mem = ShortTermMemory(ttl_seconds=None)
    sid = await mem.start_session()
    state = await mem.get(sid)
    assert state.expires_at is None


# ---------------------------------------------------------------------------
# Retrievals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_retrievals_dedupes_by_ref_id():
    mem = ShortTermMemory()
    sid = await mem.start_session()
    items = [
        RetrievedItem(kind=RetrievalKind.CHUNK, ref_id="c1", text="a"),
        RetrievedItem(kind=RetrievalKind.CHUNK, ref_id="c1", text="a-dup"),
        RetrievedItem(kind=RetrievalKind.CHUNK, ref_id="c2", text="b"),
    ]
    await mem.append_retrievals(sid, items)
    state = await mem.get(sid)
    assert [r.ref_id for r in state.retrievals] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_context_unknown_session_returns_empty():
    mem = ShortTermMemory()
    assert await mem.build_context(session_id="nope", query="q") == []


@pytest.mark.asyncio
async def test_build_context_returns_chronological_messages():
    mem = ShortTermMemory()
    sid = await mem.start_session()
    await mem.append_message(sid, MemoryRole.USER, "first")
    await mem.append_message(sid, MemoryRole.ASSISTANT, "second")
    await mem.append_message(sid, MemoryRole.USER, "third")

    ctx = await mem.build_context(session_id=sid, query="third", token_budget=1000)
    assert [m["content"] for m in ctx] == ["first", "second", "third"]
    assert [m["role"] for m in ctx] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_build_context_compacts_overflow_into_summary():
    llm = _summarizing_llm("COMPACTED")
    # Tiny budget so older turns overflow and get summarized.
    mem = ShortTermMemory(max_context_tokens=20, llm=llm)
    sid = await mem.start_session()
    big = "x" * 200  # ~50 tokens each, exceeds the budget
    await mem.append_message(sid, MemoryRole.USER, big + " one")
    await mem.append_message(sid, MemoryRole.ASSISTANT, big + " two")
    await mem.append_message(sid, MemoryRole.USER, "latest question")

    ctx = await mem.build_context(session_id=sid, query="latest question", token_budget=20)

    # The summary is prepended as a system message and the LLM was asked to compact.
    assert ctx[0]["role"] == "system"
    assert "COMPACTED" in ctx[0]["content"]
    llm.complete.assert_awaited()

    # The most recent turn is always retained verbatim.
    assert ctx[-1]["content"] == "latest question"

    # Compacted turns are dropped from the live transcript; summary persisted.
    state = await mem.get(sid)
    assert state.summary == "COMPACTED"
    assert all("one" not in m.content and "two" not in m.content for m in state.messages)
    assert state.metadata["last_context"]["summarized_messages"] >= 1


@pytest.mark.asyncio
async def test_build_context_compaction_falls_back_without_llm():
    # No LLM configured → textual fallback, no crash.
    mem = ShortTermMemory(max_context_tokens=20, llm=None)
    sid = await mem.start_session()
    big = "y" * 200
    await mem.append_message(sid, MemoryRole.USER, big + " old")
    await mem.append_message(sid, MemoryRole.USER, "new")

    ctx = await mem.build_context(session_id=sid, query="new", token_budget=20)
    state = await mem.get(sid)
    assert state.summary is not None
    assert ctx[-1]["content"] == "new"
