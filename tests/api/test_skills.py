"""Tests for the skills APIs:

  GET  /skills
  GET  /applications/{name}/skills
  POST /applications/{name}/skills
  DELETE /applications/{name}/skills/{skill_name}

Also covers creating/updating applications with skills declared in config.yaml.
"""

from __future__ import annotations

import io
import textwrap
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import (
    get_app_cache,
    get_skill_registry,
    get_system_config,
    get_system_store,
    get_system_structured_store,
)
from api.main import app
from api.app_cache import AppCache
from api.system_config import SystemConfig
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import Skill
from cogbase.stores.structured.memory import InMemoryStructuredStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, description: str = "A test skill") -> Skill:
    return Skill(
        name=name,
        description=description,
        raw_markdown=f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n",
        metadata={},
    )


def _make_registry(*skill_names: str) -> SkillRegistry:
    registry = SkillRegistry()
    for name in skill_names:
        registry.register(_make_skill(name))
    return registry


def _make_bundle(config_yaml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", config_yaml)
    return buf.getvalue()


_BASE_CONFIG = textwrap.dedent("""\
    name: my-app
    llm:
      provider: openai
      model: gpt-4o-mini
""")

_BASE_BUNDLE = _make_bundle(_BASE_CONFIG)


def _mock_app_instance() -> MagicMock:
    inst = MagicMock()
    inst.setup = AsyncMock()
    return inst


def _make_system_store() -> SystemStore:
    return SystemStore(store=InMemoryStructuredStore())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def registry():
    """A registry pre-populated with two known skills."""
    return _make_registry("skill-alpha", "skill-beta")


@pytest_asyncio.fixture
async def client(registry):
    """AsyncClient with all external dependencies overridden."""
    system_store = _make_system_store()
    await system_store.setup()

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: AppCache()
    app.dependency_overrides[get_system_config] = lambda: SystemConfig.model_validate(
        {"system_db": {"type": "memory"}}
    )
    app.dependency_overrides[get_system_structured_store] = lambda: InMemoryStructuredStore()
    app.dependency_overrides[get_skill_registry] = lambda: registry

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def _create_app(client, config_yaml: str = _BASE_CONFIG) -> None:
    bundle = _make_bundle(config_yaml)
    with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# GET /skills
# ---------------------------------------------------------------------------

class TestListSkills:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.get("/skills")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_all_registered_skills(self, client):
        resp = await client.get("/skills")
        body = resp.json()
        assert body["total"] == 2
        names = {s["name"] for s in body["skills"]}
        assert names == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_skill_response_includes_description(self, client):
        resp = await client.get("/skills")
        by_name = {s["name"]: s for s in resp.json()["skills"]}
        assert by_name["skill-alpha"]["description"] == "A test skill"

    @pytest.mark.asyncio
    async def test_empty_when_no_skills_registered(self, client, registry):
        empty_registry = SkillRegistry()
        app.dependency_overrides[get_skill_registry] = lambda: empty_registry
        resp = await client.get("/skills")
        body = resp.json()
        assert body["total"] == 0
        assert body["skills"] == []


# ---------------------------------------------------------------------------
# GET /applications/{name}/skills
# ---------------------------------------------------------------------------

class TestListApplicationSkills:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_skills(self, client):
        await _create_app(client)
        resp = await client.get("/applications/my-app/skills")
        assert resp.status_code == 200
        assert resp.json() == {"app_name": "my-app", "skills": []}

    @pytest.mark.asyncio
    async def test_returns_skills_declared_in_config(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n"
        await _create_app(client, config)
        resp = await client.get("/applications/my-app/skills")
        assert resp.status_code == 200
        assert resp.json()["skills"] == ["skill-alpha"]

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.get("/applications/ghost/skills")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /applications/{name}/skills
# ---------------------------------------------------------------------------

class TestAddApplicationSkill:
    @pytest.mark.asyncio
    async def test_returns_201_with_updated_skills(self, client):
        await _create_app(client)
        resp = await client.post(
            "/applications/my-app/skills",
            json={"skill_name": "skill-alpha"},
        )
        assert resp.status_code == 201
        assert resp.json() == {"app_name": "my-app", "skills": ["skill-alpha"]}

    @pytest.mark.asyncio
    async def test_skill_persisted_in_config_yaml(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})

        resp = await client.get("/applications/my-app/skills")
        assert "skill-alpha" in resp.json()["skills"]

    @pytest.mark.asyncio
    async def test_adding_multiple_skills(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.post("/applications/my-app/skills", json={"skill_name": "skill-beta"})
        assert resp.status_code == 201
        assert set(resp.json()["skills"]) == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_idempotent_when_skill_already_assigned(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        assert resp.status_code == 201
        assert resp.json()["skills"].count("skill-alpha") == 1

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.post(
            "/applications/ghost/skills",
            json={"skill_name": "skill-alpha"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_skill_not_in_registry(self, client):
        await _create_app(client)
        resp = await client.post(
            "/applications/my-app/skills",
            json={"skill_name": "nonexistent-skill"},
        )
        assert resp.status_code == 404
        assert "nonexistent-skill" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /applications/{name}/skills/{skill_name}
# ---------------------------------------------------------------------------

class TestRemoveApplicationSkill:
    @pytest.mark.asyncio
    async def test_returns_204_on_success(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.delete("/applications/my-app/skills/skill-alpha")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_skill_removed_from_config_yaml(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        await client.delete("/applications/my-app/skills/skill-alpha")

        resp = await client.get("/applications/my-app/skills")
        assert "skill-alpha" not in resp.json()["skills"]

    @pytest.mark.asyncio
    async def test_removes_only_specified_skill(self, client):
        await _create_app(client)
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        await client.post("/applications/my-app/skills", json={"skill_name": "skill-beta"})
        await client.delete("/applications/my-app/skills/skill-alpha")

        resp = await client.get("/applications/my-app/skills")
        skills = resp.json()["skills"]
        assert "skill-alpha" not in skills
        assert "skill-beta" in skills

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.delete("/applications/ghost/skills/skill-alpha")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_when_skill_not_assigned(self, client):
        await _create_app(client)
        resp = await client.delete("/applications/my-app/skills/skill-alpha")
        assert resp.status_code == 404
        assert "skill-alpha" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Skills in config.yaml at create / update time
# ---------------------------------------------------------------------------

class TestSkillsInConfig:
    @pytest.mark.asyncio
    async def test_create_with_skills_stores_them(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n  - skill-beta\n"
        await _create_app(client, config)

        resp = await client.get("/applications/my-app/skills")
        assert set(resp.json()["skills"]) == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_create_with_unknown_skill_returns_422(self, client):
        config = _BASE_CONFIG + "skills:\n  - unknown-skill\n"
        bundle = _make_bundle(config)
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", bundle, "application/zip")},
            )
        assert resp.status_code == 422
        assert "unknown-skill" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_update_replaces_skills(self, client):
        config_v1 = _BASE_CONFIG + "skills:\n  - skill-alpha\n"
        await _create_app(client, config_v1)

        config_v2 = _BASE_CONFIG + "skills:\n  - skill-beta\n"
        bundle_v2 = _make_bundle(config_v2)
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/my-app",
                files={"bundle": ("bundle.zip", bundle_v2, "application/zip")},
            )
        assert resp.status_code == 200

        resp = await client.get("/applications/my-app/skills")
        skills = resp.json()["skills"]
        assert "skill-beta" in skills
        assert "skill-alpha" not in skills

    @pytest.mark.asyncio
    async def test_update_with_unknown_skill_returns_422(self, client):
        await _create_app(client)
        config_v2 = _BASE_CONFIG + "skills:\n  - ghost-skill\n"
        bundle_v2 = _make_bundle(config_v2)
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/my-app",
                files={"bundle": ("bundle.zip", bundle_v2, "application/zip")},
            )
        assert resp.status_code == 422
        assert "ghost-skill" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_skills_visible_in_app_config_response(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n"
        await _create_app(client, config)

        resp = await client.get("/applications/my-app")
        assert resp.status_code == 200
        assert "skill-alpha" in resp.json()["config"]["skills"]
