"""Tests for cogbase.engine.router."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.engine.router import LLMRouter, QueryPattern, RouteResult, _parse_llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_client(pattern: str, semantic_query: str | None = None) -> MagicMock:
    """Return a mock OpenAI-compatible async client."""
    payload: dict = {"pattern": pattern, "reasoning": "test"}
    if semantic_query is not None:
        payload["semantic_query"] = semantic_query

    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# LLMRouter — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["A", "B", "C", "D"])
async def test_llm_router_all_patterns(pattern: str) -> None:
    router = LLMRouter(_make_openai_client(pattern), model="test-model")
    result = await router.route("any query")
    assert result.pattern == QueryPattern(pattern)


@pytest.mark.asyncio
async def test_llm_router_returns_route_result() -> None:
    router = LLMRouter(_make_openai_client("B"), model="test-model")
    result = await router.route("what is the notice period")
    assert isinstance(result, RouteResult)


@pytest.mark.asyncio
async def test_llm_router_uses_semantic_query_from_response() -> None:
    router = LLMRouter(_make_openai_client("B", semantic_query="notice period"), model="test-model")
    result = await router.route("  what is the notice period?  ")
    assert result.semantic_query == "notice period"


@pytest.mark.asyncio
async def test_llm_router_falls_back_to_stripped_query_when_missing() -> None:
    router = LLMRouter(_make_openai_client("B"), model="test-model")
    result = await router.route("  what is the notice period?  ")
    assert result.semantic_query == "what is the notice period?"


@pytest.mark.asyncio
async def test_llm_router_passes_model_to_client() -> None:
    client = _make_openai_client("B")
    router = LLMRouter(client, model="llama3")
    await router.route("any query")
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama3"


@pytest.mark.asyncio
async def test_llm_router_sends_system_and_user_messages() -> None:
    client = _make_openai_client("B")
    router = LLMRouter(client, model="test-model")
    await router.route("my query")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert messages[1]["content"] == "my query"


# ---------------------------------------------------------------------------
# LLMRouter — errors propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_router_propagates_api_error() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    router = LLMRouter(client, model="test-model")
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await router.route("any query")


@pytest.mark.asyncio
async def test_llm_router_propagates_parse_error_after_retries() -> None:
    message = MagicMock()
    message.content = "not valid json"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    router = LLMRouter(client, model="test-model", max_retries=2)
    with pytest.raises(Exception):
        await router.route("any query")

    # 1 initial attempt + 2 retries = 3 total calls
    assert client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_llm_router_retries_on_bad_json_then_succeeds() -> None:
    """Router succeeds on the second attempt when the first returns bad JSON."""
    bad_message = MagicMock()
    bad_message.content = "not valid json"
    bad_choice = MagicMock()
    bad_choice.message = bad_message
    bad_response = MagicMock()
    bad_response.choices = [bad_choice]

    good_message = MagicMock()
    good_message.content = '{"pattern": "B", "semantic_query": "notice period"}'
    good_choice = MagicMock()
    good_choice.message = good_message
    good_response = MagicMock()
    good_response.choices = [good_choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[bad_response, good_response]
    )

    router = LLMRouter(client, model="test-model", max_retries=2)
    result = await router.route("what is the notice period")

    assert result.pattern == QueryPattern.B
    assert client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_llm_router_no_retry_when_max_retries_zero() -> None:
    message = MagicMock()
    message.content = "not valid json"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    router = LLMRouter(client, model="test-model", max_retries=0)
    with pytest.raises(Exception):
        await router.route("any query")

    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# _parse_llm_response — JSON parsing
# ---------------------------------------------------------------------------


def test_parse_bare_json() -> None:
    raw = '{"pattern": "C", "semantic_query": "compare contracts", "reasoning": "x"}'
    result = _parse_llm_response(raw, "original")
    assert result.pattern == QueryPattern.C
    assert result.semantic_query == "compare contracts"


def test_parse_json_with_code_fence() -> None:
    raw = '```json\n{"pattern": "D", "semantic_query": "draft letter"}\n```'
    result = _parse_llm_response(raw, "original")
    assert result.pattern == QueryPattern.D


def test_parse_json_with_plain_fence() -> None:
    raw = '```\n{"pattern": "A", "semantic_query": "list facts"}\n```'
    result = _parse_llm_response(raw, "original")
    assert result.pattern == QueryPattern.A


def test_parse_lowercase_pattern() -> None:
    raw = '{"pattern": "b", "semantic_query": "what is the penalty"}'
    result = _parse_llm_response(raw, "original")
    assert result.pattern == QueryPattern.B


def test_parse_missing_semantic_query_uses_original() -> None:
    raw = '{"pattern": "B", "reasoning": "semantic"}'
    result = _parse_llm_response(raw, "original query")
    assert result.semantic_query == "original query"


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(Exception):
        _parse_llm_response("not json", "original")


def test_parse_invalid_pattern_raises() -> None:
    with pytest.raises(Exception):
        _parse_llm_response('{"pattern": "Z"}', "original")
