"""Tests for api/routers/app_generate.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from api.dependencies import RequestScope
from api.routers.app_generate import (
    _chat_turn_events,
    chat,
)
from api.models import GenerateChatRequest


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


def _scope(account_id: str = "acme", namespace_id: str = "default") -> RequestScope:
    return RequestScope(account_id=account_id, namespace_id=namespace_id)


class TestChatTurn:
    async def test_chat_drains_shared_stream_and_returns_final_response(self):
        llm = _make_llm("A final response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat(_scope(), body, system_resources)

        assert response.content == "A final response"
        assert response.config_yaml is None
        assert llm.complete_stream.call_count == 1

    async def test_chat_turn_events_emit_result(self):
        llm = _make_llm("A final response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        events = []
        async for event in _chat_turn_events(
            body, system_resources, account_id="acme", log_prefix="test/chat"
        ):
            events.append(event)

        assert events[-1]["type"] == "result"
        assert events[-1]["result"]["content"] == "A final response"

    async def test_chat_is_account_scoped(self):
        """chat threads the request's account_id through to the turn logic.

        The account (from the X-Account-Id header, surfaced as RequestScope) is
        the tenant boundary for a stateless generate turn — no namespace is
        involved since nothing is created until deploy.
        """
        llm = _make_llm("scoped response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat(_scope(account_id="tenant-42"), body, system_resources)

        assert response.content == "scoped response"
