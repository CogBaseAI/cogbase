"""Tests for OpenAILLM."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.llms import LLMBase, OpenAILLM
from cogbase.llms.base import ChatMessage


def _make_non_stream_client(content) -> MagicMock:
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    response = SimpleNamespace(choices=[choice])
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _make_streaming_client(*deltas: str | None) -> MagicMock:
    async def _stream():
        for delta_text in deltas:
            delta = SimpleNamespace(content=delta_text)
            choice = SimpleNamespace(delta=delta)
            yield SimpleNamespace(choices=[choice])

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    return client


@pytest.mark.asyncio
async def test_complete_returns_response_text() -> None:
    client = _make_non_stream_client("hello world")
    llm = OpenAILLM(client, model="test-model")
    messages: list[ChatMessage] = [{"role": "user", "content": "hi"}]

    out = await llm.complete(messages, max_tokens=128, temperature=0.1)

    assert out == "hello world"
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

    out = await llm.complete([{"role": "user", "content": "hi"}])

    assert out == "alpha beta"


@pytest.mark.asyncio
async def test_complete_stream_yields_delta_content() -> None:
    client = _make_streaming_client("hel", None, "lo", "", "!")
    llm = OpenAILLM(client, model="test-model")

    out = [part async for part in llm.complete_stream([{"role": "user", "content": "hi"}])]

    assert out == ["hel", "lo", "!"]
    call = client.chat.completions.create.call_args.kwargs
    assert call["stream"] is True


@pytest.mark.asyncio
async def test_complete_passes_reasoning_effort() -> None:
    client = _make_non_stream_client("hello")
    llm = OpenAILLM(client, model="gpt-5")

    await llm.complete(
        [{"role": "user", "content": "think"}],
        reasoning_effort="high",
    )

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

    await llm.complete(
        [{"role": "user", "content": "think"}],
        reasoning_effort="minimal",
    )

    call = client.chat.completions.create.call_args.kwargs
    assert call["reasoning_effort"] == "minimal"


def test_is_subclass() -> None:
    assert issubclass(OpenAILLM, LLMBase)
