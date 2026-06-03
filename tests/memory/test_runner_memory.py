"""Integration tests for QueryRunner wired to short-term memory."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.llms.base import CompletionResult
from cogbase.memory import MemoryRole, ShortTermMemory


def _text_result(content: str) -> CompletionResult:
    return {"content": content, "tool_calls": None}


def _make_llm(*results: CompletionResult) -> MagicMock:
    """Fake LLM that streams queued results in order (mirrors test_runner.py)."""
    llm = MagicMock()
    queue = list(results)
    pos = [0]

    async def _stream_gen(result: CompletionResult):
        if result.get("content"):
            yield result["content"]
        if result.get("tool_calls"):
            yield result

    def _pop():
        r = queue[pos[0]]
        pos[0] += 1
        return r

    llm.complete_stream = MagicMock(side_effect=lambda *a, **kw: _stream_gen(_pop()))
    return llm


async def _drain(runner: QueryRunner, *args, **kwargs) -> tuple[list[str], QueryResult]:
    tokens: list[str] = []
    result: QueryResult | None = None
    async for item in runner.run(*args, **kwargs):
        if isinstance(item, str):
            tokens.append(item)
        else:
            result = item
    assert result is not None
    return tokens, result


def _runner(llm, short_term=None) -> QueryRunner:
    return QueryRunner(
        app_name="testapp",
        llm=llm,
        document_store=MagicMock(),
        short_term=short_term,
    )


@pytest.mark.asyncio
async def test_run_records_turns_into_session():
    llm = _make_llm(_text_result("hi there"))
    mem = ShortTermMemory()
    sid = await mem.start_session(app_name="testapp")
    runner = _runner(llm, short_term=mem)

    _, result = await _drain(runner, "hello", session_id=sid)
    assert result.answer.startswith("hi there")

    state = await mem.get(sid)
    roles = [(m.role, m.content) for m in state.messages]
    assert (MemoryRole.USER, "hello") in roles
    assert any(r is MemoryRole.ASSISTANT for r, _ in roles)


@pytest.mark.asyncio
async def test_second_turn_sees_prior_context_without_caller_history():
    # Turn 1
    mem = ShortTermMemory()
    sid = await mem.start_session()

    llm1 = _make_llm(_text_result("Paris is the capital of France."))
    await _drain(_runner(llm1, short_term=mem), "What is the capital of France?", session_id=sid)

    # Turn 2 — capture the messages the LLM is given (no caller history passed).
    captured = {}

    def _stream_gen(result):
        async def _gen():
            yield result["content"]
        return _gen()

    llm2 = MagicMock()
    def _side_effect(messages, *a, **kw):
        captured["messages"] = messages
        return _stream_gen(_text_result("It has about 2 million people."))
    llm2.complete_stream = MagicMock(side_effect=_side_effect)

    await _drain(_runner(llm2, short_term=mem), "How many people live there?", session_id=sid)

    contents = [m.get("content") for m in captured["messages"]]
    # Prior turn's question and answer are present in the assembled context.
    assert any("capital of France" in (c or "") for c in contents)
    assert any("Paris" in (c or "") for c in contents)
    assert any("How many people" in (c or "") for c in contents)


@pytest.mark.asyncio
async def test_no_memory_path_is_unchanged():
    # No short_term configured → caller history is used, nothing recorded.
    llm = _make_llm(_text_result("answer"))
    runner = _runner(llm, short_term=None)
    _, result = await _drain(
        runner,
        "q",
        history=[{"role": "user", "content": "earlier"}],
        session_id="ignored-when-no-memory",
    )
    assert result.answer.startswith("answer")
