"""Tests for OpenAILLM."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.llms.base import LLMBase, ChatMessage, CompletionResult, ToolDefinition
from cogbase.llms.openai import OpenAILLM


def _make_non_stream_client(
    content,
    *,
    finish_reason: str = "stop",
    tool_calls=None,
) -> MagicMock:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _make_streaming_client(*deltas: str | None) -> MagicMock:
    async def _stream():
        for delta_text in deltas:
            delta = SimpleNamespace(content=delta_text, tool_calls=None)
            choice = SimpleNamespace(delta=delta)
            yield SimpleNamespace(choices=[choice])

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    return client


def _make_streaming_tool_call_client(tool_calls: list[dict]) -> MagicMock:
    """Client that streams tool call deltas (no text content)."""
    async def _stream():
        for i, tc in enumerate(tool_calls):
            tc_delta = SimpleNamespace(
                index=i,
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=""),
            )
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc_delta])
            )])
            if tc["arguments"]:
                tc_delta = SimpleNamespace(
                    index=i,
                    id=None,
                    function=SimpleNamespace(name=None, arguments=tc["arguments"]),
                )
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[tc_delta])
                )])

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    return client


def _make_tool_call_client(tool_calls: list) -> MagicMock:
    """Client that returns tool_calls with finish_reason='tool_calls'."""
    tc_objects = [
        SimpleNamespace(
            id=tc["id"],
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        )
        for tc in tool_calls
    ]
    return _make_non_stream_client(
        content=None,
        finish_reason="tool_calls",
        tool_calls=tc_objects,
    )


# ---------------------------------------------------------------------------
# Basic completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_response_text() -> None:
    client = _make_non_stream_client("hello world")
    llm = OpenAILLM(client, model="test-model")
    messages: list[ChatMessage] = [{"role": "user", "content": "hi"}]

    result = await llm.complete(messages, max_tokens=128, temperature=0.1)

    assert result["content"] == "hello world"
    assert result["tool_calls"] is None
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["messages"] == messages
    assert call["max_completion_tokens"] == 128
    assert call["temperature"] == 0.1
    assert call["stream"] is False


@pytest.mark.asyncio
async def test_complete_handles_content_parts() -> None:
    content = [
        {"type": "text", "text": "alpha "},
        {"type": "input_text", "text": "ignored"},
        {"type": "text", "text": "beta"},
    ]
    client = _make_non_stream_client(content)
    llm = OpenAILLM(client, model="test-model")

    result = await llm.complete([{"role": "user", "content": "hi"}])

    assert result["content"] == "alpha beta"
    assert result["tool_calls"] is None


@pytest.mark.asyncio
async def test_complete_stream_yields_delta_content() -> None:
    client = _make_streaming_client("hel", None, "lo", "", "!")
    llm = OpenAILLM(client, model="test-model")

    out = [part async for part in llm.complete_stream([{"role": "user", "content": "hi"}])]

    assert [p for p in out if isinstance(p, str)] == ["hel", "lo", "!"]
    assert isinstance(out[-1], dict)  # final CompletionResult
    call = client.chat.completions.create.call_args.kwargs
    assert call["stream"] is True


# ---------------------------------------------------------------------------
# Reasoning effort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_passes_reasoning_effort() -> None:
    client = _make_non_stream_client("hello")
    llm = OpenAILLM(client, model="gpt-5")

    await llm.complete([{"role": "user", "content": "think"}], reasoning_effort="high")

    call = client.chat.completions.create.call_args.kwargs
    assert call["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_instance_default_reasoning_effort_applies() -> None:
    client = _make_streaming_client("ok")
    llm = OpenAILLM(client, model="gpt-5-mini", reasoning_effort="medium")

    _ = [part async for part in llm.complete_stream([{"role": "user", "content": "hi"}])]

    call = client.chat.completions.create.call_args.kwargs
    assert call["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_call_reasoning_effort_overrides_instance_default() -> None:
    client = _make_non_stream_client("hello")
    llm = OpenAILLM(client, model="gpt-5", reasoning_effort="low")

    await llm.complete([{"role": "user", "content": "think"}], reasoning_effort="minimal")

    call = client.chat.completions.create.call_args.kwargs
    assert call["reasoning_effort"] == "minimal"


# ---------------------------------------------------------------------------
# Tool support
# ---------------------------------------------------------------------------

_WEATHER_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Return current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.asyncio
async def test_complete_passes_tools_in_openai_format() -> None:
    client = _make_non_stream_client("done")
    llm = OpenAILLM(client, model="test-model")

    await llm.complete([{"role": "user", "content": "weather?"}], tools=[_WEATHER_TOOL])

    call = client.chat.completions.create.call_args.kwargs
    assert "tools" in call
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return current weather for a city.",
                "parameters": _WEATHER_TOOL["parameters"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_complete_without_tools_omits_tools_key() -> None:
    client = _make_non_stream_client("done")
    llm = OpenAILLM(client, model="test-model")

    await llm.complete([{"role": "user", "content": "hi"}])

    call = client.chat.completions.create.call_args.kwargs
    assert "tools" not in call


@pytest.mark.asyncio
async def test_complete_returns_tool_calls_on_finish_reason_tool_calls() -> None:
    raw_args = json.dumps({"city": "London"})
    client = _make_tool_call_client(
        [{"id": "call_abc", "name": "get_weather", "arguments": raw_args}]
    )
    llm = OpenAILLM(client, model="test-model")

    result = await llm.complete(
        [{"role": "user", "content": "weather in London?"}],
        tools=[_WEATHER_TOOL],
    )

    assert result["content"] is None
    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "London"}


@pytest.mark.asyncio
async def test_complete_returns_content_when_no_tool_calls() -> None:
    client = _make_non_stream_client("It is sunny.", finish_reason="stop")
    llm = OpenAILLM(client, model="test-model")

    result = await llm.complete(
        [{"role": "user", "content": "weather?"}],
        tools=[_WEATHER_TOOL],
    )

    assert result["content"] == "It is sunny."
    assert result["tool_calls"] is None


@pytest.mark.asyncio
async def test_complete_returns_multiple_tool_calls() -> None:
    raw_args_1 = json.dumps({"city": "Paris"})
    raw_args_2 = json.dumps({"city": "Berlin"})
    client = _make_tool_call_client(
        [
            {"id": "call_1", "name": "get_weather", "arguments": raw_args_1},
            {"id": "call_2", "name": "get_weather", "arguments": raw_args_2},
        ]
    )
    llm = OpenAILLM(client, model="test-model")

    result = await llm.complete(
        [{"role": "user", "content": "weather in Paris and Berlin?"}],
        tools=[_WEATHER_TOOL],
    )

    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][1]["id"] == "call_2"


@pytest.mark.asyncio
async def test_complete_stream_passes_tools() -> None:
    client = _make_streaming_client("ok")
    llm = OpenAILLM(client, model="test-model")

    _ = [
        part
        async for part in llm.complete_stream(
            [{"role": "user", "content": "hi"}], tools=[_WEATHER_TOOL]
        )
    ]

    call = client.chat.completions.create.call_args.kwargs
    assert "tools" in call
    assert call["tools"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_complete_stream_yields_tool_calls() -> None:
    raw_args = json.dumps({"city": "Paris"})
    client = _make_streaming_tool_call_client([
        {"id": "call_1", "name": "get_weather", "arguments": raw_args},
    ])
    llm = OpenAILLM(client, model="test-model")

    chunks = [
        chunk
        async for chunk in llm.complete_stream(
            [{"role": "user", "content": "weather?"}], tools=[_WEATHER_TOOL]
        )
    ]

    assert len(chunks) == 1
    result = chunks[0]
    assert isinstance(result, dict)
    assert result["content"] is None
    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["arguments"] == raw_args


@pytest.mark.asyncio
async def test_complete_stream_yields_text_then_tool_calls() -> None:
    raw_args = json.dumps({"city": "Berlin"})

    async def _mixed_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Sure! ", tool_calls=None)
        )])
        tc_delta = SimpleNamespace(
            index=0, id="call_2",
            function=SimpleNamespace(name="get_weather", arguments=raw_args),
        )
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=[tc_delta])
        )])

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_mixed_stream())
    llm = OpenAILLM(client, model="test-model")

    chunks = [chunk async for chunk in llm.complete_stream(
        [{"role": "user", "content": "weather?"}], tools=[_WEATHER_TOOL]
    )]

    assert chunks[0] == "Sure! "
    result: CompletionResult = chunks[1]  # type: ignore[assignment]
    assert result["tool_calls"][0]["id"] == "call_2"
    assert result["tool_calls"][0]["arguments"] == raw_args


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_is_subclass() -> None:
    assert issubclass(OpenAILLM, LLMBase)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_uses_default_model_when_no_model_arg() -> None:
    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="gpt-4o", mini_model="gpt-4o-mini")

    await llm.complete([{"role": "user", "content": "hi"}])

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_complete_uses_mini_model_when_configured() -> None:
    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="gpt-4o", mini_model="gpt-4o-mini")

    await llm.complete([{"role": "user", "content": "hi"}], model="mini")

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_complete_falls_back_to_default_when_mini_not_configured() -> None:
    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="gpt-4o")

    await llm.complete([{"role": "user", "content": "hi"}], model="mini")

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_complete_uses_literal_model_override() -> None:
    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="gpt-4o", mini_model="gpt-4o-mini")

    await llm.complete([{"role": "user", "content": "hi"}], model="o3")

    assert client.chat.completions.create.call_args.kwargs["model"] == "o3"


@pytest.mark.asyncio
async def test_complete_stream_uses_mini_model_when_configured() -> None:
    client = _make_streaming_client("ok")
    llm = OpenAILLM(client, model="gpt-4o", mini_model="gpt-4o-mini")

    _ = [part async for part in llm.complete_stream([{"role": "user", "content": "hi"}], model="mini")]

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_complete_stream_falls_back_to_default_when_mini_not_configured() -> None:
    client = _make_streaming_client("ok")
    llm = OpenAILLM(client, model="gpt-4o")

    _ = [part async for part in llm.complete_stream([{"role": "user", "content": "hi"}], model="mini")]

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# LLM completion timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_records_into_active_tracker() -> None:
    from cogbase.llms.timing import track_llm_time

    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="test-model")

    with track_llm_time() as timing:
        await llm.complete([{"role": "user", "content": "hi"}])

    assert timing.calls == 1
    assert timing.seconds >= 0.0


@pytest.mark.asyncio
async def test_complete_stream_records_into_active_tracker() -> None:
    from cogbase.llms.timing import track_llm_time

    client = _make_streaming_client("a", "b")
    llm = OpenAILLM(client, model="test-model")

    with track_llm_time() as timing:
        _ = [p async for p in llm.complete_stream([{"role": "user", "content": "hi"}])]

    assert timing.calls == 1
    assert timing.seconds >= 0.0


@pytest.mark.asyncio
async def test_complete_without_tracker_does_not_error() -> None:
    client = _make_non_stream_client("ok")
    llm = OpenAILLM(client, model="test-model")

    result = await llm.complete([{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
