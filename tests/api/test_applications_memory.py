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
from cogbase.core.query_runner import MemoryTiers, QueryRunner, RetrievalResources
from cogbase.memory import EpisodicMemory, ShortTermMemory
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.log.local_fs import LocalFSLogStore
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


def _real_app(name: str, mem: ShortTermMemory, episodic: EpisodicMemory, llm: MagicMock) -> CogBaseApp:
    """A real CogBaseApp whose QueryRunner is wired to *mem* (no skills, no stores)."""
    runner = QueryRunner(
        app_id=name,
        llm=llm,
        resources=RetrievalResources(document_store=MagicMock()),
        memory=MemoryTiers(short_term=mem, episodic=episodic),
    )
    return CogBaseApp(
        name=name,
        pipelines=[],
        runner=runner,
        app_id=name,
        document_store=MagicMock(),
        structured_store=MagicMock(),
        workflow_runners={},
        llm=llm,
        task_store=MagicMock(),
        short_term=mem,
        episodic=episodic,
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
async def test_query_endpoint_threads_session_through_real_short_term_memory(client, tmp_path):
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    mem = ShortTermMemory(episodic=episodic)
    sid = await mem.start_session(app_id="memory-e2e-app")
    captured: list[list] = []
    llm = _streaming_llm(
        ["Paris is the capital of France.", "About 2 million people live there."],
        captured,
    )
    real_app = _real_app("memory-e2e-app", mem, episodic, llm)

    # Deploy the real app behind the API.
    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=real_app):
        resp = await client.post(
            "/namespaces/default/applications",
            files={"bundle": ("bundle.zip", _make_bundle(), "application/zip")},
        )
    assert resp.status_code == 201

    # Turn 1 — no caller history; the session is the source of truth.
    r1 = await client.post(
        "/namespaces/default/applications/memory-e2e-app/query",
        json={"text": "What is the capital of France?", "session_id": sid},
    )
    assert r1.status_code == 200
    assert "Paris" in r1.json()["answer"]
    assert r1.json()["session_id"] == sid

    # Turn 2 — same session, a follow-up that only resolves via prior context.
    r2 = await client.post(
        "/namespaces/default/applications/memory-e2e-app/query",
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


@pytest.mark.asyncio
async def test_session_history_list_and_transcript(client, tmp_path):
    """The session index lists a chat after its first turn; transcript reads the log."""
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    mem = ShortTermMemory(episodic=episodic)
    sid = await mem.start_session(app_id="memory-e2e-app")
    captured: list[list] = []
    llm = _streaming_llm(["Paris is the capital.", "About 2 million."], captured)
    real_app = _real_app("memory-e2e-app", mem, episodic, llm)

    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=real_app):
        resp = await client.post(
            "/namespaces/default/applications",
            files={"bundle": ("bundle.zip", _make_bundle(), "application/zip")},
        )
    assert resp.status_code == 201

    # No turns yet -> the session is not in the history list.
    r = await client.get("/namespaces/default/applications/memory-e2e-app/sessions")
    assert r.status_code == 200
    assert r.json()["sessions"] == []

    # Two turns on the same session.
    await client.post(
        "/namespaces/default/applications/memory-e2e-app/query",
        json={"text": "What is the capital of France?", "session_id": sid},
    )
    await client.post(
        "/namespaces/default/applications/memory-e2e-app/query",
        json={"text": "How many people live there?", "session_id": sid},
    )

    # The session now shows once, titled by the first message, count == turns.
    r = await client.get("/namespaces/default/applications/memory-e2e-app/sessions")
    sessions = r.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["title"] == "What is the capital of France?"
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["status"] == "open"

    # Transcript reads the durable log: user + assistant turns, in order.
    r = await client.get(f"/namespaces/default/applications/memory-e2e-app/sessions/{sid}")
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "What is the capital of France?"
    assert msgs[1]["content"].strip() == "Paris is the capital."
    # User turns carry no references; assistant turns re-hydrate the answer's
    # reference payload (empty here — the canned answer retrieved nothing).
    assert msgs[0]["references"] is None
    assert msgs[1]["references"] == {
        "structured_records": [],
        "chunks": [],
        "document_slices": [],
        "memories": [],
    }

    # Closing flips the index row to 'closed'.
    r = await client.post(f"/namespaces/default/applications/memory-e2e-app/sessions/{sid}/close")
    assert r.status_code == 200
    r = await client.get("/namespaces/default/applications/memory-e2e-app/sessions")
    assert r.json()["sessions"][0]["status"] == "closed"


@pytest.mark.asyncio
async def test_delete_session_removes_index_row_and_transcript(client, tmp_path):
    """Deleting a session drops it from the history list and erases its log."""
    episodic = EpisodicMemory(LocalFSLogStore(tmp_path))
    mem = ShortTermMemory(episodic=episodic)
    sid = await mem.start_session(app_id="memory-e2e-app")
    captured: list[list] = []
    llm = _streaming_llm(["Paris is the capital."], captured)
    real_app = _real_app("memory-e2e-app", mem, episodic, llm)

    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=real_app):
        resp = await client.post(
            "/namespaces/default/applications",
            files={"bundle": ("bundle.zip", _make_bundle(), "application/zip")},
        )
    assert resp.status_code == 201

    await client.post(
        "/namespaces/default/applications/memory-e2e-app/query",
        json={"text": "What is the capital of France?", "session_id": sid},
    )

    # The session is listed and its transcript is readable.
    r = await client.get("/namespaces/default/applications/memory-e2e-app/sessions")
    assert len(r.json()["sessions"]) == 1

    # Delete it.
    r = await client.delete(f"/namespaces/default/applications/memory-e2e-app/sessions/{sid}")
    assert r.status_code == 200
    assert r.json() == {"session_id": sid, "deleted": True}

    # Gone from the history list.
    r = await client.get("/namespaces/default/applications/memory-e2e-app/sessions")
    assert r.json()["sessions"] == []

    # The durable episodic log is gone: transcript replays empty.
    r = await client.get(f"/namespaces/default/applications/memory-e2e-app/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["messages"] == []
