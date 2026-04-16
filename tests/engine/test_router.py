"""Tests for cogbase.engine.router."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.engine.router import (
    CollectionTarget,
    LLMRouter,
    QueryPattern,
    RouteResult,
    _build_system_prompt,
    _parse_llm_response,
)
from cogbase.stores.filters import Op
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_client(
    pattern: str,
    semantic_query: str | None = None,
    structured_targets: list | None = None,
) -> MagicMock:
    """Return a mock OpenAI-compatible async client."""
    payload: dict = {"pattern": pattern, "reasoning": "test"}
    if semantic_query is not None:
        payload["semantic_query"] = semantic_query
    if structured_targets is not None:
        payload["structured_targets"] = structured_targets

    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


_SAMPLE_SCHEMA = [
    CollectionSchema(
        name="contracts", primary_fields=["doc_id"],
        fields={
            "doc_id":         FieldSchema(type=FieldType.STRING),
            "party_a":        FieldSchema(type=FieldType.STRING),
            "effective_date": FieldSchema(type=FieldType.STRING),
        },
    ),
    CollectionSchema(
        name="facts", primary_fields=["fact_id"],
        fields={
            "fact_id":    FieldSchema(type=FieldType.STRING),
            "type":       FieldSchema(type=FieldType.STRING),
            "confidence": FieldSchema(type=FieldType.FLOAT),
            "metadata":   FieldSchema(
                type=FieldType.JSON,
                json_schema='{"status": "string", "priority": "int"}',
            ),
        },
    ),
]


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


@pytest.mark.asyncio
async def test_llm_router_no_schema_gives_empty_targets() -> None:
    router = LLMRouter(_make_openai_client("A"), model="test-model")
    result = await router.route("list all contracts")
    assert result.structured_targets == []


@pytest.mark.asyncio
async def test_llm_router_schema_injected_into_system_prompt() -> None:
    client = _make_openai_client("B")
    router = LLMRouter(client, model="test-model", schema=_SAMPLE_SCHEMA)
    await router.route("any query")
    system_content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "contracts" in system_content
    assert "facts" in system_content
    assert "party_a" in system_content
    # field types and their valid operators should be present
    assert "float" in system_content           # confidence field type
    assert "json" in system_content            # metadata field type
    assert "dot notation" in system_content   # json fields filterable via field.subkey
    assert 'subkeys={"status": "string", "priority": "int"}' in system_content


@pytest.mark.asyncio
async def test_llm_router_parses_single_structured_target() -> None:
    targets_payload = [{"collection": "facts", "filters": []}]
    client = _make_openai_client("A", structured_targets=targets_payload)
    router = LLMRouter(client, model="test-model", schema=_SAMPLE_SCHEMA)
    result = await router.route("list all facts")
    assert len(result.structured_targets) == 1
    assert result.structured_targets[0].collection == "facts"
    assert result.structured_targets[0].filters == []


@pytest.mark.asyncio
async def test_llm_router_parses_multiple_structured_targets() -> None:
    targets_payload = [
        {"collection": "contracts", "filters": []},
        {"collection": "facts", "filters": []},
    ]
    client = _make_openai_client("C", structured_targets=targets_payload)
    router = LLMRouter(client, model="test-model", schema=_SAMPLE_SCHEMA)
    result = await router.route("compare contracts and facts")
    assert len(result.structured_targets) == 2
    assert result.structured_targets[0].collection == "contracts"
    assert result.structured_targets[1].collection == "facts"


@pytest.mark.asyncio
async def test_llm_router_parses_filters_in_target() -> None:
    targets_payload = [
        {
            "collection": "facts",
            "filters": [{"field": "type", "op": "=", "value": "date"}],
        }
    ]
    client = _make_openai_client("A", structured_targets=targets_payload)
    router = LLMRouter(client, model="test-model", schema=_SAMPLE_SCHEMA)
    result = await router.route("find date facts")
    target = result.structured_targets[0]
    assert len(target.filters) == 1
    assert target.filters[0].field == "type"
    assert target.filters[0].op == Op.EQ
    assert target.filters[0].value == "date"


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
    assert result.structured_targets == []


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


def test_parse_structured_targets_with_filters() -> None:
    raw = json.dumps({
        "pattern": "A",
        "semantic_query": "find date facts",
        "structured_targets": [
            {
                "collection": "facts",
                "filters": [{"field": "type", "op": "=", "value": "date"}],
            }
        ],
    })
    result = _parse_llm_response(raw, "original")
    assert len(result.structured_targets) == 1
    t = result.structured_targets[0]
    assert t.collection == "facts"
    assert t.filters[0].field == "type"
    assert t.filters[0].op == Op.EQ
    assert t.filters[0].value == "date"


def test_parse_multiple_structured_targets() -> None:
    raw = json.dumps({
        "pattern": "C",
        "semantic_query": "compare",
        "structured_targets": [
            {"collection": "contracts", "filters": []},
            {"collection": "facts", "filters": [{"field": "confidence", "op": ">=", "value": 0.8}]},
        ],
    })
    result = _parse_llm_response(raw, "original")
    assert len(result.structured_targets) == 2
    assert result.structured_targets[0].collection == "contracts"
    assert result.structured_targets[1].filters[0].op == Op.GTE
    assert result.structured_targets[1].filters[0].value == 0.8


def test_parse_null_structured_targets_treated_as_empty() -> None:
    raw = '{"pattern": "B", "semantic_query": "search", "structured_targets": null}'
    result = _parse_llm_response(raw, "original")
    assert result.structured_targets == []


def test_parse_is_null_filter_no_value() -> None:
    raw = json.dumps({
        "pattern": "A",
        "semantic_query": "unresolved",
        "structured_targets": [
            {"collection": "facts", "filters": [{"field": "resolution", "op": "is_null"}]},
        ],
    })
    result = _parse_llm_response(raw, "original")
    f = result.structured_targets[0].filters[0]
    assert f.op == Op.IS_NULL
    assert f.value is None


def test_parse_in_filter_with_array_value() -> None:
    raw = json.dumps({
        "pattern": "A",
        "semantic_query": "date or numeric facts",
        "structured_targets": [
            {
                "collection": "facts",
                "filters": [{"field": "type", "op": "in", "value": ["date", "numeric"]}],
            }
        ],
    })
    result = _parse_llm_response(raw, "original")
    f = result.structured_targets[0].filters[0]
    assert f.op == Op.IN
    assert f.value == ["date", "numeric"]


# ---------------------------------------------------------------------------
# available_patterns — system prompt restriction
# ---------------------------------------------------------------------------


def test_available_patterns_restricts_prompt_to_ad() -> None:
    """When available_patterns=[A, D], patterns B and C must not appear in the prompt."""
    prompt = _build_system_prompt(None, [QueryPattern.A, QueryPattern.D])
    assert "B —" not in prompt
    assert "C —" not in prompt
    assert "A —" in prompt
    assert "D —" in prompt


def test_available_patterns_pattern_ids_reflect_restriction() -> None:
    """The pattern placeholder in the return-JSON hint is narrowed to available patterns."""
    prompt = _build_system_prompt(None, [QueryPattern.A, QueryPattern.D])
    assert "<A|D>" in prompt


def test_available_patterns_none_includes_all_four() -> None:
    """Passing None (default) keeps all four patterns in the prompt."""
    prompt = _build_system_prompt(None, None)
    for label in ("A —", "B —", "C —", "D —"):
        assert label in prompt
    assert "<A|B|C|D>" in prompt


@pytest.mark.asyncio
async def test_llm_router_available_patterns_injected_into_system_prompt() -> None:
    """LLMRouter passes available_patterns to the system prompt builder."""
    client = _make_openai_client("A")
    router = LLMRouter(
        client,
        model="test-model",
        available_patterns=[QueryPattern.A, QueryPattern.D],
    )
    await router.route("list all contracts")
    system_content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "B —" not in system_content
    assert "C —" not in system_content
    assert "A —" in system_content
    assert "D —" in system_content
