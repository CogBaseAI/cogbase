"""Integration tests for QueryRunner wired to episodic memory."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.llms.base import CompletionResult, SystemTool
from cogbase.memory import EpisodicMemory, EventType
from cogbase.stores.log.local_fs import LocalFSLogStore


def _text_result(content: str) -> CompletionResult:
    return {"content": content, "tool_calls": None}


def _tool_call_result(call_id: str, name: str, arguments: dict) -> CompletionResult:
    return {
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": json.dumps(arguments)}],
    }


def _make_llm(*results: CompletionResult) -> MagicMock:
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


@pytest.fixture
def episodic(tmp_path):
    return EpisodicMemory(LocalFSLogStore(tmp_path))


def _runner(llm, episodic=None, system_tools=None) -> QueryRunner:
    return QueryRunner(
        app_id="testapp",
        llm=llm,
        document_store=MagicMock(),
        episodic=episodic,
        system_tools=system_tools,
    )


def _echo_tool() -> SystemTool:
    def handler(inputs: dict) -> str:
        return f"echo: {inputs.get('text', '')}"

    return SystemTool(
        definition={
            "name": "echo",
            "description": "Echo back the input text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        handler=handler,
    )


# -- basic turn recording ---------------------------------------------------


async def test_run_records_user_message_and_final_answer(episodic):
    llm = _make_llm(_text_result("hi there"))
    await _drain(_runner(llm, episodic=episodic), "hello", session_id="s1")

    events = await episodic.replay(session_id="s1")
    assert [e.event_type for e in events] == [
        EventType.USER_MESSAGE,
        EventType.FINAL_ANSWER,
    ]
    assert events[0].payload["text"] == "hello"
    assert events[1].payload["text"].startswith("hi there")


async def test_recorded_events_are_durable_after_the_turn(episodic):
    # The final-answer flush must persist the turn before run() returns.
    llm = _make_llm(_text_result("answer"))
    await _drain(_runner(llm, episodic=episodic), "q", session_id="s1")
    assert not episodic.has_pending("s1")  # flushed, nothing buffered


async def test_attribution_is_carried_onto_events(episodic):
    llm = _make_llm(_text_result("a"))
    await _drain(_runner(llm, episodic=episodic), "q", session_id="s1")

    events = await episodic.replay(session_id="s1")
    assert all(e.app_id == "testapp" for e in events)


# -- tool call / result recording -------------------------------------------


async def test_tool_call_and_result_are_recorded_with_causal_link(episodic):
    llm = _make_llm(
        _tool_call_result("call-1", "echo", {"text": "ping"}),
        _text_result("done"),
    )
    runner = _runner(llm, episodic=episodic, system_tools=[_echo_tool()])
    await _drain(runner, "use echo", session_id="s1")

    events = await episodic.replay(session_id="s1")
    types = [e.event_type for e in events]
    assert types == [
        EventType.USER_MESSAGE,
        EventType.TOOL_CALLED,
        EventType.TOOL_RESULT,
        EventType.FINAL_ANSWER,
    ]

    called = events[1]
    result = events[2]
    assert called.payload["name"] == "echo"
    assert called.payload["tool_call_id"] == "call-1"
    assert result.payload["result"] == "echo: ping"
    assert result.payload["ok"] is True
    assert result.payload["latency_ms"] is not None
    # tool_result threads back to its tool_called via the identity triplet.
    assert result.parent_event_id is not None
    assert result.parent_event_id.seq == called.seq


async def test_whole_turn_is_one_append(episodic, tmp_path):
    llm = _make_llm(
        _tool_call_result("call-1", "echo", {"text": "x"}),
        _text_result("done"),
    )
    runner = _runner(llm, episodic=episodic, system_tools=[_echo_tool()])
    await _drain(runner, "q", session_id="s1")

    # Four events, but the turn flushes once: each NDJSON line ends in "\n".
    raw = (tmp_path / "episodic" / "s1").read_text()
    assert raw.count("\n") == 4


# -- failure tiering --------------------------------------------------------


class _FailingAppendLogStore(LocalFSLogStore):
    """A log store whose append always fails — simulates the durable log down."""

    async def append(self, *args, **kwargs):
        raise RuntimeError("log store down")


async def test_continuity_flush_failure_fails_the_turn(tmp_path):
    # user_message / final_answer must be durable before the turn is acknowledged;
    # a persistent flush failure is surfaced, not swallowed.
    episodic = EpisodicMemory(_FailingAppendLogStore(tmp_path))
    runner = _runner(_make_llm(_text_result("answer")), episodic=episodic)
    with pytest.raises(RuntimeError, match="log store down"):
        await _drain(runner, "q", session_id="s1")
    # The events stay buffered (the retry buffer) for a later attempt.
    assert episodic.pending_continuity("s1")


async def test_best_effort_recording_failure_does_not_break_the_turn(episodic, monkeypatch):
    # A tool_called/tool_result recording failure is swallowed: losing it costs
    # only analytics, never continuity.
    async def _boom(*args, **kwargs):
        raise RuntimeError("telemetry hiccup")

    monkeypatch.setattr(episodic, "record_tool_call", _boom)
    monkeypatch.setattr(episodic, "record_tool_result", _boom)

    llm = _make_llm(
        _tool_call_result("call-1", "echo", {"text": "x"}),
        _text_result("done"),
    )
    runner = _runner(llm, episodic=episodic, system_tools=[_echo_tool()])
    _, result = await _drain(runner, "q", session_id="s1")

    assert result.answer.startswith("done")
    # The turn still committed its continuity events, just not the tool telemetry.
    types = [e.event_type for e in await episodic.replay(session_id="s1")]
    assert types == [EventType.USER_MESSAGE, EventType.FINAL_ANSWER]


# -- no-memory path ---------------------------------------------------------


async def test_no_episodic_path_records_nothing(episodic):
    llm = _make_llm(_text_result("answer"))
    # episodic wired, but no session_id → nothing recorded.
    await _drain(_runner(llm, episodic=episodic), "q")
    assert await episodic.replay(session_id="s1") == []


async def test_runner_without_episodic_is_unaffected():
    llm = _make_llm(_text_result("answer"))
    _, result = await _drain(_runner(llm, episodic=None), "q", session_id="s1")
    assert result.answer.startswith("answer")
