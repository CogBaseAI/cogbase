"""Integration tests for the /namespaces REST endpoints.

Uses httpx.AsyncClient pointed at the FastAPI app with dependency overrides —
no real LLM calls or file I/O happens.
"""

from __future__ import annotations

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
from api.system_store import AppRecord, SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.structured.memory import InMemoryStructuredStore


@pytest_asyncio.fixture
async def app_overrides():
    """AsyncClient plus the underlying SystemStore for seeding test data."""
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
        yield {"client": ac, "system_store": system_store}

    app.dependency_overrides.clear()


def _seed_app(system_store: SystemStore, name: str, namespace_id: str, account_id: str = "default"):
    return system_store.save_app(AppRecord(
        app_id=name,
        account_id=account_id,
        namespace_id=namespace_id,
        name=name,
        config_yaml="name: x\nllm:\n  model: gpt-4o-mini\n",
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    ))


class TestCreateNamespace:
    @pytest.mark.asyncio
    async def test_create(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.post(
            "/namespaces",
            json={"namespace_id": "team-a", "display_name": "Team A", "description": "d"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["namespace_id"] == "team-a"
        assert body["display_name"] == "Team A"
        assert body["account_id"] == "default"
        assert body["created_at"]

    @pytest.mark.asyncio
    async def test_create_minimal(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.post("/namespaces", json={"namespace_id": "team-a"})
        assert resp.status_code == 201
        assert resp.json()["display_name"] is None

    @pytest.mark.asyncio
    async def test_duplicate_conflicts(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"})
        resp = await client.post("/namespaces", json={"namespace_id": "team-a"})
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_invalid_name_rejected(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.post("/namespaces", json={"namespace_id": "bad name!"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_scoped_by_account(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"}, headers={"X-Account-Id": "acct-1"})
        # Same handle in a different account is a distinct namespace.
        resp = await client.post("/namespaces", json={"namespace_id": "team-a"}, headers={"X-Account-Id": "acct-2"})
        assert resp.status_code == 201


class TestListNamespaces:
    @pytest.mark.asyncio
    async def test_list_empty(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.get("/namespaces")
        assert resp.status_code == 200
        assert resp.json() == {"namespaces": [], "total": 0}

    @pytest.mark.asyncio
    async def test_list_returns_created(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"})
        await client.post("/namespaces", json={"namespace_id": "team-b"})
        resp = await client.get("/namespaces")
        assert resp.json()["total"] == 2
        assert {n["namespace_id"] for n in resp.json()["namespaces"]} == {"team-a", "team-b"}

    @pytest.mark.asyncio
    async def test_list_scoped_by_account(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"}, headers={"X-Account-Id": "acct-1"})
        await client.post("/namespaces", json={"namespace_id": "team-b"}, headers={"X-Account-Id": "acct-2"})
        resp = await client.get("/namespaces", headers={"X-Account-Id": "acct-1"})
        assert {n["namespace_id"] for n in resp.json()["namespaces"]} == {"team-a"}


class TestGetNamespace:
    @pytest.mark.asyncio
    async def test_get(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a", "display_name": "Team A"})
        resp = await client.get("/namespaces/team-a")
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Team A"

    @pytest.mark.asyncio
    async def test_get_missing_404(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.get("/namespaces/ghost")
        assert resp.status_code == 404


class TestUpdateNamespace:
    @pytest.mark.asyncio
    async def test_update_fields(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a", "display_name": "old"})
        resp = await client.patch("/namespaces/team-a", json={"display_name": "new", "description": "d"})
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "new"
        assert resp.json()["description"] == "d"

    @pytest.mark.asyncio
    async def test_partial_update_leaves_other_field(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a", "display_name": "keep"})
        resp = await client.patch("/namespaces/team-a", json={"description": "only-desc"})
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "keep"
        assert resp.json()["description"] == "only-desc"

    @pytest.mark.asyncio
    async def test_empty_update_422(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"})
        resp = await client.patch("/namespaces/team-a", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_update_missing_404(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.patch("/namespaces/ghost", json={"display_name": "x"})
        assert resp.status_code == 404


class TestDeleteNamespace:
    @pytest.mark.asyncio
    async def test_delete_empty(self, app_overrides):
        client = app_overrides["client"]
        await client.post("/namespaces", json={"namespace_id": "team-a"})
        resp = await client.delete("/namespaces/team-a")
        assert resp.status_code == 204
        assert (await client.get("/namespaces/team-a")).status_code == 404

    @pytest.mark.asyncio
    async def test_delete_missing_404(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.delete("/namespaces/ghost")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_default_refused(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.delete("/namespaces/default")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_delete_nonempty_refused(self, app_overrides):
        client = app_overrides["client"]
        system_store = app_overrides["system_store"]
        await client.post("/namespaces", json={"namespace_id": "team-a"})
        await _seed_app(system_store, "my-app", "team-a")
        resp = await client.delete("/namespaces/team-a")
        assert resp.status_code == 409
        # still there
        assert (await client.get("/namespaces/team-a")).status_code == 200


class TestNamespaceAutoRegistration:
    @pytest.mark.asyncio
    async def test_app_create_registers_namespace(self, app_overrides):
        """Creating an app in a namespace should surface it in GET /namespaces."""
        import io
        import textwrap
        import zipfile

        client = app_overrides["client"]
        config_yaml = textwrap.dedent("""\
            name: auto-app
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-test
        """).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.yaml", config_yaml)

        resp = await client.post(
            "/namespaces/team-auto/applications",
            files={"bundle": ("a.zip", buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 201
        listed = await client.get("/namespaces")
        assert "team-auto" in {n["namespace_id"] for n in listed.json()["namespaces"]}
