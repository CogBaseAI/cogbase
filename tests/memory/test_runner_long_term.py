"""Integration tests for QueryRunner wired to long-term memory.

Verifies the recall seam: when a ``LongTermMemory`` is wired, ``run`` recalls
relevant records and injects them into the LLM context as a system block
marked memory-derived (kept distinct from document-backed evidence).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.query_runner import MemoryTiers, QueryRunner, RetrievalResources
from cogbase.llms.base import CompletionResult
from cogbase.memory.long_term import LongTermMemory
from cogbase.memory.models import LongTermRecord, MemoryCandidate, MemoryKind
from cogbase.stores.scope import AppScope
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.vector.faiss_store import FAISSMemoryVectorStore

from tests.memory.test_long_term import HashingEmbedding

_OBSERVED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _capturing_llm(answer: str) -> tuple[MagicMock, list]:
    """Fake LLM that records the messages of each completion and streams *answer*."""
    llm = MagicMock()
    captured: list = []

    async def _stream(messages, **kw):
        captured.append(messages)
        yield answer

    llm.complete_stream = MagicMock(side_effect=_stream)
    return llm, captured


async def _long_term(app_id="app1") -> LongTermMemory:
    return LongTermMemory(
        InMemoryStructuredStore().with_scope(AppScope(app_id=app_id)),
        FAISSMemoryVectorStore().with_scope(AppScope(app_id=app_id)),
        MagicMock(),
        HashingEmbedding(),
        app_id=app_id,
    )


async def _drain(runner, **kwargs):
    result = None
    async for item in runner.run(**kwargs):
        if not isinstance(item, str):
            result = item
    return result


@pytest.mark.asyncio
async def test_recall_injects_memory_block():
    lt = await _long_term()
    await lt.promote(
        candidate=MemoryCandidate(
            content="user prefers dark mode", kind=MemoryKind.PREFERENCE,
            confidence=0.7, observed_at=_OBSERVED_AT,
        ),
    )
    llm, captured = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )

    await _drain(runner, user_input="what theme do I like?")

    # The first completion's message list carries a memory-derived system block.
    system_blocks = [m["content"] for m in captured[0] if m["role"] == "system"]
    assert any("memory-derived" in b and "dark mode" in b for b in system_blocks)


@pytest.mark.asyncio
async def test_recall_block_surfaced_on_result_only_when_cited():
    """The recall block is one citable passage: its records reach the QueryResult
    only when the answer cites the block id, and as a whole when it does."""
    lt = await _long_term()
    await lt.promote(
        candidate=MemoryCandidate(
            content="user prefers dark mode", kind=MemoryKind.PREFERENCE,
            confidence=0.7, observed_at=_OBSERVED_AT,
        ),
    )

    # Answer cites the recall block id -> the block's records are surfaced.
    llm_cited, _ = _capturing_llm("You prefer dark mode [memory-1].")
    runner = QueryRunner(
        app_id="app1", llm=llm_cited,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )
    result = await _drain(runner, user_input="what theme do I like?")
    assert [m.content for m in result.memories] == ["user prefers dark mode"]

    # Same recall, but the answer cites nothing -> no memory on the result.
    llm_uncited, _ = _capturing_llm("I am not sure.")
    runner = QueryRunner(
        app_id="app1", llm=llm_uncited,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )
    result = await _drain(runner, user_input="what theme do I like?")
    assert result.memories == []


@pytest.mark.asyncio
async def test_no_recall_when_nothing_relevant_injects_no_block():
    lt = await _long_term()
    llm, captured = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )

    await _drain(runner, user_input="anything")
    system_blocks = " ".join(m["content"] for m in captured[0] if m["role"] == "system")
    assert "memory-derived" not in system_blocks


@pytest.mark.asyncio
async def test_recall_query_includes_previous_exchange_for_follow_ups():
    """A bare follow-up recalls via the prior exchange folded into the query."""
    lt = await _long_term()
    await lt.promote(
        candidate=MemoryCandidate(
            content="user prefers dark mode", kind=MemoryKind.PREFERENCE,
            confidence=0.7, observed_at=_OBSERVED_AT,
        ),
    )
    llm, captured = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )

    history = [
        {"role": "user", "content": "what theme do I like?"},
        {"role": "assistant", "content": "You prefer dark mode."},
    ]
    # On its own, "and on mobile?" recalls nothing under the hashing embedding.
    await _drain(runner, user_input="and on mobile?", history=history)

    system_blocks = [m["content"] for m in captured[0] if m["role"] == "system"]
    assert any("memory-derived" in b and "dark mode" in b for b in system_blocks)


def test_compose_recall_query_shapes():
    compose = QueryRunner._compose_recall_query
    # No prior conversation: the input passes through unchanged.
    assert compose("hello", []) == "hello"
    # Trailing current-input user message (build_context's shape) is dropped;
    # tool messages and tool-call-only assistant messages are skipped.
    prior = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": None},
        {"role": "tool", "content": "raw tool output"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up"},
    ]
    assert compose("follow-up", prior) == "first question\nfirst answer\nfollow-up"
    # Long previous turns are truncated; the current input never is.
    prior = [
        {"role": "user", "content": "q" * 1000},
        {"role": "assistant", "content": "a" * 1000},
    ]
    composed = compose("follow-up", prior)
    assert composed == "q" * 300 + "\n" + "a" * 500 + "\nfollow-up"


# ---------------------------------------------------------------------------
# memory_lookup tool (the pull path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_lookup_tool_registered_only_when_long_term_wired():
    lt = await _long_term()
    llm, _ = _capturing_llm("ok")
    with_lt = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
        enable_memory_lookup=True,
    )
    without_lt = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        enable_memory_lookup=True,
    )
    assert "memory_lookup" in [t["name"] for t in with_lt._tool_defs]
    assert "memory_lookup" not in [t["name"] for t in without_lt._tool_defs]


@pytest.mark.asyncio
async def test_memory_lookup_tool_disabled_by_default():
    """The pull tool is opt-in: a wired long-term tier alone does not expose it."""
    lt = await _long_term()
    llm, _ = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )
    assert "memory_lookup" not in [t["name"] for t in runner._tool_defs]


@pytest.mark.asyncio
async def test_memory_lookup_tool_withheld_when_disabled():
    lt = await _long_term()
    llm, _ = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
        enable_memory_lookup=False,
    )
    assert "memory_lookup" not in [t["name"] for t in runner._tool_defs]


@pytest.mark.asyncio
async def test_memory_lookup_tool_returns_matching_memories():
    lt = await _long_term()
    await lt.promote(
        candidate=MemoryCandidate(
            content="user works at Acme Corp", kind=MemoryKind.PREFERENCE,
            entities=["acme corp"], confidence=0.7, observed_at=_OBSERVED_AT,
        ),
    )
    llm, _ = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )

    output, memories = await runner._run_memory_lookup({"entities": ["Acme Corp"]})
    assert "user works at Acme Corp" in output
    assert "memory-derived" in output
    assert [m.content for m in memories] == ["user works at Acme Corp"]

    output, memories = await runner._run_memory_lookup({"query": "unrelated topic"})
    assert output is not None
    assert all(isinstance(m, LongTermRecord) for m in memories)


# ---------------------------------------------------------------------------
# Chronological rendering of recalled memories
#
# Both injection paths (recall block + memory_lookup tool) render the dated
# lines oldest -> newest so they read as a timeline, while the returned records
# keep the store's native order (relevance / recency) for the caller.
# ---------------------------------------------------------------------------

def _record(content: str, observed_at: datetime) -> LongTermRecord:
    return LongTermRecord(
        content=content, kind=MemoryKind.PREFERENCE,
        confidence=0.7, observed_at=observed_at,
    )


def _runner_with_long_term_stub() -> tuple[QueryRunner, MagicMock]:
    stub = MagicMock()
    runner = QueryRunner(
        app_id="app1", llm=_capturing_llm("ok")[0],
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=stub),
    )
    return runner, stub


def _rendered_dates(block: str) -> list[str]:
    return re.findall(r"as of (\d{4}-\d{2}-\d{2})", block)


@pytest.mark.asyncio
async def test_recall_block_renders_oldest_to_newest_but_returns_recall_order():
    runner, stub = _runner_with_long_term_stub()
    # Records handed back in a non-chronological (relevance) order.
    recall_order = [
        _record("newest", datetime(2024, 3, 1, tzinfo=timezone.utc)),
        _record("oldest", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _record("middle", datetime(2024, 2, 1, tzinfo=timezone.utc)),
    ]
    stub.recall = AsyncMock(return_value=recall_order)

    block, returned = await runner._recall_memory_block("q")

    # Rendered lines are chronological.
    assert _rendered_dates(block) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert block.index("oldest") < block.index("middle") < block.index("newest")
    # Returned records preserve recall's (relevance) order for the caller.
    assert returned is recall_order


@pytest.mark.asyncio
async def test_recall_block_renders_stable_within_same_date():
    runner, stub = _runner_with_long_term_stub()
    same = datetime(2024, 1, 1, tzinfo=timezone.utc)
    stub.recall = AsyncMock(return_value=[
        _record("first by relevance", same),
        _record("second by relevance", same),
    ])

    block, _ = await runner._recall_memory_block("q")

    # Equal observed_at -> stable sort keeps the incoming relevance order.
    assert block.index("first by relevance") < block.index("second by relevance")


@pytest.mark.asyncio
async def test_memory_lookup_renders_oldest_to_newest_but_returns_lookup_order():
    runner, stub = _runner_with_long_term_stub()
    lookup_order = [
        _record("newest", datetime(2024, 3, 1, tzinfo=timezone.utc)),
        _record("oldest", datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ]
    stub.lookup = AsyncMock(return_value=lookup_order)

    output, returned = await runner._run_memory_lookup({"query": "q"})

    assert _rendered_dates(output) == ["2024-01-01", "2024-03-01"]
    assert output.index("oldest") < output.index("newest")
    assert returned is lookup_order


@pytest.mark.asyncio
async def test_memory_lookup_tool_rejects_empty_and_bad_arguments():
    lt = await _long_term()
    llm, _ = _capturing_llm("ok")
    runner = QueryRunner(
        app_id="app1", llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(long_term=lt),
    )
    assert "error" in (await runner._run_memory_lookup({}))[0]
    assert "error" in (await runner._run_memory_lookup({"kind": "nonsense"}))[0]
