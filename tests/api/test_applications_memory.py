"""End-to-end test: /applications/{name}/query wired to a real ShortTermMemory.

Unlike test_applications.py (which mocks ``CogBaseApp.query_stream``), this test
builds a *real* CogBaseApp → QueryRunner → ShortTermMemory and only fakes the
LLM.  Two queries are issued over the HTTP layer with the same ``session_id``;
the assertion proves the second turn's prompt carries the first turn's context
as assembled by the real memory — i.e. the session round-trips through the API.
"""

from __future__ import annotations

import textwrap
import zipfile
import io
from unittest.mock import AsyncMock, MagicMock, patch

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
from api.main import app
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.core.app import CogBaseApp
from cogbase.core.query_runner import QueryRunner
from cogbase.memory import ShortTermMemory
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.structured.memory import InMemoryStructuredStore


_CONFIG_YAML = textwrap.dedent("""\
    name: memory-e2e-app
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-test
""").encode()


def _make_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", _CONFIG_YAML)
    return buf.getvalue()


def _streaming_llm(answers: list[str], captured: list[list]) -> MagicMock:
    """Fake LLM whose complete_stream yields the next queued answer as one token.

    Each call's ``messages`` list is appended to *captured* so the test can
    inspect exactly what the runner sent to the model on each turn.
    """
    llm = MagicMock()
    queue = list(answers)

    def _side_effect(messages, *a, **kw):
        captured.append(messages)
        answer = queue.pop(0)

        async def _gen():
            yield answer

        return _gen()

    llm.complete_stream = MagicMock(side_effect=_side_effect)
    return llm


def _real_app(name: str, mem: ShortTermMemory, llm: MagicMock) -> CogBaseApp:
    """A real CogBaseApp whose QueryRunner is wired to *mem* (no skills, no stores)."""
    runner = QueryRunner(
        app_name=name,
        llm=llm,
        document_store=MagicMock(),
        short_term=mem,
    )
    return CogBaseApp(
        name=name,
        pipelines=[],
        runner=runner,
        document_store=MagicMock(),
        structured_store=MagicMock(),
        workflow_runners={},
        llm=llm,
        task_store=MagicMock(),
    )


@pytest_asyncio.fixture
async def client():
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    app_cache = AppCache()
    system_resources = SystemResources(structured_store=InMemoryStructuredStore())

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_resources] = lambda: system_resources
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_endpoint_threads_session_through_real_short_term_memory(client):
    mem = ShortTermMemory()
    sid = await mem.start_session(app_name="memory-e2e-app")
    captured: list[list] = []
    llm = _streaming_llm(
        ["Paris is the capital of France.", "About 2 million people live there."],
        captured,
    )
    real_app = _real_app("memory-e2e-app", mem, llm)

    # Deploy the real app behind the API.
    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=real_app):
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", _make_bundle(), "application/zip")},
        )
    assert resp.status_code == 201

    # Turn 1 — no caller history; the session is the source of truth.
    r1 = await client.post(
        "/applications/memory-e2e-app/query",
        json={"text": "What is the capital of France?", "session_id": sid},
    )
    assert r1.status_code == 200
    assert "Paris" in r1.json()["answer"]
    assert r1.json()["session_id"] == sid

    # Turn 2 — same session, a follow-up that only resolves via prior context.
    r2 = await client.post(
        "/applications/memory-e2e-app/query",
        json={"text": "How many people live there?", "session_id": sid},
    )
    assert r2.status_code == 200
    assert "2 million" in r2.json()["answer"]

    # The second turn's prompt must contain turn 1, assembled by ShortTermMemory.
    second_turn_msgs = captured[1]
    contents = [str(m.get("content", "")) for m in second_turn_msgs]
    assert any("capital of France" in c for c in contents)
    assert any("Paris" in c for c in contents)
    assert any("How many people" in c for c in contents)

    # And the real session now holds both turns end-to-end.
    state = await mem.get(sid)
    transcript = " ".join(m.content for m in state.messages)
    assert "capital of France" in transcript
    assert "Paris" in transcript
    assert "How many people" in transcript
    assert "2 million" in transcript
