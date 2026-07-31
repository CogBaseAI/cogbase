"""Tests for api/routers/app_generate.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from api.routers.app_generate import (
    _chat_turn_events,
    chat,
)
from api.models import GenerateChatRequest
from cogbase.core.app_generator import (
    GENERATOR_TOOLS,
    PROPOSE_APP_CONFIG_TOOL_NAME,
)


async def _text_stream(text: str):
    yield text


def _make_llm(*responses: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[{"content": r, "tool_calls": None} for r in responses]
    )
    llm.complete_stream = MagicMock(
        side_effect=[_text_stream(r) for r in responses]
    )
    return llm


def _tool_call_stream(name: str, arguments: dict, text: str = ""):
    """A stream that emits optional text, then a tool call in the final chunk."""
    async def stream(*args, **kwargs):
        if text:
            yield text
        yield {
            "content": text or None,
            "tool_calls": [
                {"id": "call-1", "name": name, "arguments": json.dumps(arguments)}
            ],
        }
    return stream


def _resources(llm) -> MagicMock:
    return MagicMock(llm=llm)


class TestChatTurn:
    async def test_chat_drains_shared_stream_and_returns_final_response(self):
        llm = _make_llm("A final response")
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat("acme", body, _resources(llm))

        assert response.content == "A final response"
        assert response.config_yaml is None
        assert llm.complete_stream.call_count == 1

    async def test_chat_turn_events_emit_result(self):
        llm = _make_llm("A final response")
        body = GenerateChatRequest(text="hello", history=[])

        events = []
        async for event in _chat_turn_events(
            body, _resources(llm), account_id="acme", log_prefix="test/chat"
        ):
            events.append(event)

        assert events[-1]["type"] == "result"
        assert events[-1]["result"]["content"] == "A final response"

    async def test_chat_is_account_scoped(self):
        """chat threads the request's account_id through to the turn logic.

        The account (from the X-Account-Id header) is the tenant boundary for a
        stateless generate turn — no namespace is involved since nothing is
        created until deploy.
        """
        llm = _make_llm("scoped response")
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat("tenant-42", body, _resources(llm))

        assert response.content == "scoped response"


class TestToolDispatch:
    def test_the_generator_has_exactly_one_tool(self):
        """The company profile is not the generator's concern in either direction.

        Collecting it belongs to the onboarding interview, editing it to
        ``PUT /profile``, and applying it to the query runner via the factory.
        """
        assert [t["name"] for t in GENERATOR_TOOLS] == [PROPOSE_APP_CONFIG_TOOL_NAME]

    async def test_unknown_tool_does_not_run_config_generation(self):
        """The loop used to run propose_app_config for any tool name."""
        llm = MagicMock()
        llm.complete_stream = MagicMock(
            side_effect=[
                _tool_call_stream("not_a_real_tool", {})(),
                _text_stream("recovered"),
            ]
        )
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat("acme", body, _resources(llm))

        assert response.config_yaml is None
        assert response.content == "recovered"
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "Unknown tool" in tool_result

    async def test_propose_app_config_is_dispatched(self, monkeypatch):
        async def fake_propose(llm, messages, *, needs_workflow):
            yield {
                "type": "result",
                "generation_context": "config generated",
                "config_yaml": "name: demo\n",
            }

        monkeypatch.setattr(
            "api.routers.app_generate.propose_app_config", fake_propose
        )
        llm = MagicMock()
        llm.complete_stream = MagicMock(
            side_effect=[
                _tool_call_stream(
                    PROPOSE_APP_CONFIG_TOOL_NAME, {"needs_workflow": False}
                )(),
                _text_stream("here is your app"),
            ]
        )
        body = GenerateChatRequest(text="build it", history=[])

        response = await chat("acme", body, _resources(llm))

        assert response.config_yaml == "name: demo\n"
        assert response.content == "here is your app"
