"""Tests for the onboarding interview — POST /profile/interview/chat.

The interview has its own surface rather than living in the generator chat: a new
account is provisioned with a working namespace and app (``api/provisioning.py``),
so the user who most needs onboarding may never open the Build tab.

Mostly direct calls into the endpoint functions with a fake LLM (the pattern in
``test_app_generate.py``), plus a couple of HTTP tests for route wiring and SSE.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.app_cache import AppCache
from api.dependencies import (
    get_app_cache,
    get_skill_registry,
    get_system_resources,
    get_system_store,
)
from api.main import app as fastapi_app
from api.models import InterviewChatRequest
from api.routers.profile import interview_chat, interview_chat_stream
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.core.onboarding import (
    INTERVIEW_SKILL_NAME,
    SAVE_COMPANY_PROFILE_TOOL_NAME,
)
from cogbase.core.profile import MAX_PROFILE_BYTES, AccountProfileStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import ONBOARDING_SURFACE, Skill
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore

PROFILE_MD = "# Company Profile\n\n**Risk appetite:** conservative\n"

#: The interview is nothing without a script — questions and profile template both
#: live in a SKILL.md — so every turn here needs one registered.
SCRIPT_MD = "# Interview\n\nAsk who they are, then save the profile.\n"


def _interview_skill(markdown: str = SCRIPT_MD, *, skill_id: str = "builtin") -> Skill:
    return Skill(
        name=INTERVIEW_SKILL_NAME,
        description="d",
        raw_markdown=markdown,
        id=skill_id,
        # As skills_dir would: a builtin on the onboarding surface.
        builtin=True,
        surface=ONBOARDING_SURFACE,
    )


async def _text_stream(text: str):
    yield text


def _make_llm(*responses: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[{"content": r, "tool_calls": None} for r in responses]
    )
    llm.complete_stream = MagicMock(side_effect=[_text_stream(r) for r in responses])
    return llm


def _tool_call_stream(name: str, arguments: dict, text: str = ""):
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


def _saves_then_answers(markdown: str = PROFILE_MD) -> MagicMock:
    """First round calls save_company_profile; second round narrates the save."""
    llm = MagicMock()
    llm.complete_stream = MagicMock(
        side_effect=[
            _tool_call_stream(SAVE_COMPANY_PROFILE_TOOL_NAME, {"markdown": markdown})(),
            _text_stream("Saved. Here's what I heard — what did I get wrong?"),
        ]
    )
    return llm


def _resources(llm, document_store=None) -> MagicMock:
    return MagicMock(llm=llm, document_store=document_store)


@pytest_asyncio.fixture
async def deps():
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    registry = SkillRegistry()
    registry.register(_interview_skill(), account_id=None)  # as skills_dir would
    return {
        "document_store": InMemoryDocumentStore(),
        "system_store": system_store,
        "app_cache": AppCache(),
        "registry": registry,
    }


async def _run(llm, deps, *, text="hi", account_id="acme", history=None):
    return await interview_chat(
        account_id,
        InterviewChatRequest(text=text, history=history or []),
        _resources(llm, deps["document_store"]),
        deps["system_store"],
        deps["app_cache"],
        deps["registry"],
    )


def _system_prompt(llm: MagicMock) -> str:
    return llm.complete_stream.call_args_list[0][0][0][0]["content"]


async def _sse_frames(response) -> list[dict]:
    frames = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        frames.append(json.loads(chunk.removeprefix("data: ").strip()))
    return frames


class TestInterviewTurn:
    async def test_a_plain_turn_returns_the_reply(self, deps):
        response = await _run(_make_llm("Quick or full?"), deps)

        assert response.content == "Quick or full?"
        assert response.profile_saved is False
        assert response.markdown is None

    async def test_history_is_replayed_before_the_new_message(self, deps):
        from api.models import ChatMessage

        llm = _make_llm("go on")
        await _run(llm, deps, text="we're a SaaS company", history=[
            ChatMessage(role="assistant", content="Quick or full?"),
            ChatMessage(role="user", content="quick"),
        ])

        roles = [m["role"] for m in llm.complete_stream.call_args_list[0][0][0]]
        assert roles == ["system", "assistant", "user", "user"]

    async def test_no_llm_is_a_503(self, deps):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await _run(None, deps)
        assert exc.value.status_code == 503


class TestScriptSelection:
    async def test_the_registered_skill_is_the_script(self, deps):
        llm = _make_llm("hi")

        await _run(llm, deps)

        prompt = _system_prompt(llm)
        assert "Ask who they are" in prompt                # the skill's questions
        assert SAVE_COMPANY_PROFILE_TOOL_NAME in prompt    # the framework's plumbing

    async def test_a_replaced_skill_changes_the_interview(self, deps):
        """Editing the SKILL.md is the whole deployment story for a new script."""
        deps["registry"].register(
            _interview_skill("# Interview\n\nAsk about their submarines.\n"),
            account_id=None,
            replace=True,
        )
        llm = _make_llm("hi")

        await _run(llm, deps)

        assert "submarines" in _system_prompt(llm)

    async def test_no_script_is_a_503(self, deps):
        """The interview is its script; without one there is nothing to run."""
        from fastapi import HTTPException

        deps["registry"] = SkillRegistry()

        with pytest.raises(HTTPException) as exc:
            await _run(_make_llm("hi"), deps)

        assert exc.value.status_code == 503
        assert INTERVIEW_SKILL_NAME in exc.value.detail

    async def test_an_existing_profile_makes_the_turn_a_rerun(self, deps):
        await AccountProfileStore(deps["document_store"]).save("acme", PROFILE_MD)
        llm = _make_llm("what changed?")

        await _run(llm, deps)

        prompt = _system_prompt(llm)
        assert "**Risk appetite:** conservative" in prompt
        assert "re-run" in prompt

    async def test_another_accounts_profile_is_not_used(self, deps):
        await AccountProfileStore(deps["document_store"]).save("other", PROFILE_MD)
        llm = _make_llm("hi")

        await _run(llm, deps)

        assert "**Risk appetite:** conservative" not in _system_prompt(llm)

    async def test_a_broken_store_still_runs_the_interview(self, deps):
        """A read hiccup costs a re-ask, not the turn."""
        broken = MagicMock()
        broken.with_scope.side_effect = RuntimeError("store down")
        llm = _make_llm("hi")

        response = await interview_chat(
            "acme",
            InterviewChatRequest(text="hi"),
            _resources(llm, broken),
            deps["system_store"],
            deps["app_cache"],
            deps["registry"],
        )

        assert response.content == "hi"
        assert "BEGIN CURRENT PROFILE" not in _system_prompt(llm)


class TestSaving:
    async def test_profile_is_persisted(self, deps):
        await _run(_saves_then_answers(), deps)

        saved = await AccountProfileStore(deps["document_store"]).load("acme")
        assert saved == PROFILE_MD.strip()

    async def test_response_tells_the_ui_the_profile_was_saved(self, deps):
        """``profile_saved`` is what dismisses the onboarding card."""
        response = await _run(_saves_then_answers(), deps)

        assert response.profile_saved is True
        assert response.markdown == PROFILE_MD.strip()
        assert response.content == "Saved. Here's what I heard — what did I get wrong?"

    async def test_the_loop_continues_after_saving(self, deps):
        llm = _saves_then_answers()

        await _run(llm, deps)

        assert llm.complete_stream.call_count == 2
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "Company profile saved" in tool_result

    async def test_index_record_marks_the_interview_as_the_source(self, deps):
        await _run(_saves_then_answers(), deps)

        record = await deps["system_store"].get_profile_record("acme")
        assert record is not None
        assert record.source == "interview"

    async def test_live_apps_are_hot_patched(self, deps):
        """The provisioned app picks up the profile without a rebuild."""
        provisioned = MagicMock()
        deps["app_cache"].add("acme/legal-team/contract-analyst", provisioned)
        other_account = MagicMock()
        deps["app_cache"].add("other/legal-team/contract-analyst", other_account)

        await _run(_saves_then_answers(), deps)

        provisioned.set_account_profile.assert_called_once_with(PROFILE_MD.strip())
        other_account.set_account_profile.assert_not_called()

    async def test_oversized_profile_is_reported_not_raised(self, deps):
        llm = _saves_then_answers("x" * (MAX_PROFILE_BYTES + 1))

        response = await _run(llm, deps)

        assert response.content            # the turn still completes
        assert response.profile_saved is False
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "Profile not saved" in tool_result
        assert await AccountProfileStore(deps["document_store"]).load("acme") is None

    async def test_empty_markdown_saves_nothing(self, deps):
        response = await _run(_saves_then_answers("   "), deps)

        assert response.profile_saved is False
        assert await AccountProfileStore(deps["document_store"]).load("acme") is None

    async def test_without_a_document_store_the_model_is_told(self, deps):
        """The interview degrades where PUT /profile 503s — a live turn survives."""
        llm = _saves_then_answers()

        response = await interview_chat(
            "acme",
            InterviewChatRequest(text="hi"),
            _resources(llm, None),
            deps["system_store"],
            deps["app_cache"],
            deps["registry"],
        )

        assert response.profile_saved is False
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "not available on this deployment" in tool_result

    async def test_a_store_error_is_reported_not_raised(self, deps):
        broken = MagicMock()
        scoped = MagicMock()
        scoped.load = AsyncMock(side_effect=KeyError("missing"))
        scoped.save = AsyncMock(side_effect=RuntimeError("disk full"))
        broken.with_scope.return_value = scoped
        llm = _saves_then_answers()

        response = await interview_chat(
            "acme",
            InterviewChatRequest(text="hi"),
            _resources(llm, broken),
            deps["system_store"],
            deps["app_cache"],
            deps["registry"],
        )

        assert response.profile_saved is False
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "failed" in tool_result

    async def test_unknown_tool_is_reported_back(self, deps):
        llm = MagicMock()
        llm.complete_stream = MagicMock(side_effect=[
            _tool_call_stream("not_a_real_tool", {})(),
            _text_stream("recovered"),
        ])

        response = await _run(llm, deps)

        assert response.content == "recovered"
        assert response.profile_saved is False
        messages = llm.complete_stream.call_args_list[1][0][0]
        tool_result = [m for m in messages if m["role"] == "tool"][0]["content"]
        assert "Unknown tool" in tool_result


class TestStreaming:
    async def test_tokens_then_a_done_frame(self, deps):
        response = await interview_chat_stream(
            "acme",
            InterviewChatRequest(text="hi"),
            _resources(_saves_then_answers(), deps["document_store"]),
            deps["system_store"],
            deps["app_cache"],
            deps["registry"],
        )

        frames = await _sse_frames(response)

        assert frames[0]["token"]
        assert frames[-1]["done"] is True
        assert frames[-1]["profile_saved"] is True
        assert frames[-1]["markdown"] == PROFILE_MD.strip()

    async def test_a_failing_llm_streams_an_error_frame(self, deps):
        llm = MagicMock()
        llm.complete_stream = MagicMock(side_effect=RuntimeError("boom"))

        response = await interview_chat_stream(
            "acme",
            InterviewChatRequest(text="hi"),
            _resources(llm, deps["document_store"]),
            deps["system_store"],
            deps["app_cache"],
            deps["registry"],
        )
        frames = await _sse_frames(response)

        assert frames[-1]["error"] == "interview turn failed"


class TestRouting:
    """Over HTTP, to prove the routes are wired and account-scoped."""

    @pytest_asyncio.fixture
    async def client(self, deps):
        system_resources = SystemResources(
            structured_store=InMemoryStructuredStore(),
            document_store=deps["document_store"],
            llm=_saves_then_answers(),
        )
        fastapi_app.dependency_overrides[get_system_store] = lambda: deps["system_store"]
        fastapi_app.dependency_overrides[get_app_cache] = lambda: deps["app_cache"]
        fastapi_app.dependency_overrides[get_system_resources] = lambda: system_resources
        fastapi_app.dependency_overrides[get_skill_registry] = lambda: deps["registry"]

        transport = ASGITransport(app=fastapi_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        fastapi_app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_interview_saves_a_profile_the_get_route_then_serves(
        self, client, deps
    ):
        resp = await client.post(
            "/profile/interview/chat",
            json={"text": "we're a UK fintech", "history": []},
            headers={"X-Account-Id": "acme"},
        )

        assert resp.status_code == 200
        assert resp.json()["profile_saved"] is True

        got = await client.get("/profile", headers={"X-Account-Id": "acme"})
        assert got.json()["exists"] is True
        assert got.json()["source"] == "interview"

    @pytest.mark.asyncio
    async def test_the_profile_lands_in_the_calling_account_only(self, client):
        await client.post(
            "/profile/interview/chat",
            json={"text": "hi", "history": []},
            headers={"X-Account-Id": "acme"},
        )

        other = await client.get("/profile", headers={"X-Account-Id": "other"})
        assert other.json()["exists"] is False
