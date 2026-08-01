"""Tests for the skills APIs:

  GET  /skills
  GET  /applications/{name}/skills
  POST /applications/{name}/skills
  DELETE /applications/{name}/skills/{skill_ref}

Also covers creating/updating applications with skills declared in config.yaml.

Skills are referenced by name in the API (or by raw id when unassigning a dangling
ref). In these tests the skill id and display name are the same string (e.g.
"skill-alpha") for readability.
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
    get_system_resources,
    get_system_store,
)
from api.system_resources import SystemResources
from api.main import app
from api.app_cache import AppCache
from api.system_store import NamespaceRecord, SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import ONBOARDING_SURFACE, Skill
from cogbase.stores.structured.memory import InMemoryStructuredStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, description: str = "A test skill", skill_id: str | None = None) -> Skill:
    return Skill(
        name=name,
        description=description,
        raw_markdown=f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n",
        id=skill_id or name,
        metadata={},
    )


def _make_platform_skill(name: str = "onboarding-interview") -> Skill:
    """A builtin serving a non-application surface — the onboarding interview is the
    live example. Only ``skills_dir`` can produce one; uploads are always
    application skills."""
    skill = _make_skill(name, description="An interview script")
    skill.builtin = True
    skill.surface = ONBOARDING_SURFACE
    return skill


def _make_registry(*skill_ids: str) -> SkillRegistry:
    registry = SkillRegistry()
    for skill_id in skill_ids:
        registry.register(_make_skill(skill_id))
    return registry


def _make_bundle(config_yaml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", config_yaml)
    return buf.getvalue()


def _skill_names(resp_json: dict) -> set[str]:
    return {s["name"] for s in resp_json["skills"]}


_BASE_CONFIG = textwrap.dedent("""\
    name: my-app
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-test
""")

_BASE_BUNDLE = _make_bundle(_BASE_CONFIG)


def _mock_app_instance() -> MagicMock:
    return MagicMock()


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
    await system_store.save_namespace(
        NamespaceRecord(
            account_id="default",
            namespace_id="default",
            name="default",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: AppCache()
    app.dependency_overrides[get_system_resources] = lambda: SystemResources(structured_store=InMemoryStructuredStore())
    app.dependency_overrides[get_skill_registry] = lambda: registry

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def _create_app(client, config_yaml: str = _BASE_CONFIG) -> None:
    bundle = _make_bundle(config_yaml)
    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
        resp = await client.post(
            "/namespaces/default/applications",
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
    async def test_skill_response_includes_id_and_description(self, client):
        resp = await client.get("/skills")
        by_name = {s["name"]: s for s in resp.json()["skills"]}
        assert by_name["skill-alpha"]["description"] == "A test skill"
        assert by_name["skill-alpha"]["id"] == "skill-alpha"

    @pytest.mark.asyncio
    async def test_empty_when_no_skills_registered(self, client, registry):
        empty_registry = SkillRegistry()
        app.dependency_overrides[get_skill_registry] = lambda: empty_registry
        resp = await client.get("/skills")
        body = resp.json()
        assert body["total"] == 0
        assert body["skills"] == []

    @pytest.mark.asyncio
    async def test_platform_surface_skills_are_hidden_by_default(self, client, registry):
        """The Skills tab lists what an app can be assigned. The onboarding
        interview is a builtin no app can use, so listing it would only invite
        the mistake the assignment guard then refuses."""
        registry.register(_make_platform_skill(), account_id=None)

        names = _skill_names((await client.get("/skills")).json())

        assert names == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_surface_all_includes_them(self, client, registry):
        registry.register(_make_platform_skill(), account_id=None)

        body = (await client.get("/skills?surface=all")).json()

        assert "onboarding-interview" in _skill_names(body)

    @pytest.mark.asyncio
    async def test_a_specific_surface_can_be_requested(self, client, registry):
        registry.register(_make_platform_skill(), account_id=None)

        body = (await client.get(f"/skills?surface={ONBOARDING_SURFACE}")).json()

        assert _skill_names(body) == {"onboarding-interview"}

    @pytest.mark.asyncio
    async def test_surface_is_reported_on_each_skill(self, client):
        by_name = {s["name"]: s for s in (await client.get("/skills")).json()["skills"]}
        assert by_name["skill-alpha"]["surface"] == "application"


# ---------------------------------------------------------------------------
# GET /applications/{name}/skills
# ---------------------------------------------------------------------------

class TestListApplicationSkills:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_skills(self, client):
        await _create_app(client)
        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert resp.status_code == 200
        assert resp.json() == {"app_name": "my-app", "skills": []}

    @pytest.mark.asyncio
    async def test_returns_skills_declared_in_config(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n"
        await _create_app(client, config)
        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert resp.status_code == 200
        assert resp.json()["skills"] == [{"id": "skill-alpha", "name": "skill-alpha", "missing": False}]

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.get("/namespaces/default/applications/ghost/skills")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_dangling_skill_id_surfaced_as_missing_not_500(self, client, registry):
        # Creation validates skill ids up-front, but a reference can go dangling
        # out of band afterwards — a skill dropped from skills_dir, or a node
        # whose registry has not finished syncing. The listing must degrade
        # gracefully (surface the ghost as a broken ref the UI can clean up)
        # rather than raise a 500 or silently drop it.
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n  - skill-beta\n"
        await _create_app(client, config)
        registry.unregister("skill-beta")

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert resp.status_code == 200
        assert resp.json()["skills"] == [
            {"id": "skill-alpha", "name": "skill-alpha", "missing": False},
            {"id": "skill-beta", "name": "skill-beta", "missing": True},
        ]


# ---------------------------------------------------------------------------
# POST /applications/{name}/skills
# ---------------------------------------------------------------------------

class TestAddApplicationSkill:
    @pytest.mark.asyncio
    async def test_returns_201_with_updated_skills(self, client):
        await _create_app(client)
        resp = await client.post(
            "/namespaces/default/applications/my-app/skills",
            json={"skill_name": "skill-alpha"},
        )
        assert resp.status_code == 201
        assert resp.json() == {
            "app_name": "my-app",
            "skills": [{"id": "skill-alpha", "name": "skill-alpha", "missing": False}],
        }

    @pytest.mark.asyncio
    async def test_skill_persisted_in_config_yaml(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert "skill-alpha" in _skill_names(resp.json())

    @pytest.mark.asyncio
    async def test_a_platform_surface_skill_cannot_be_assigned(self, client, registry):
        """The onboarding interview is 200 lines of interview script; assigning it
        would hand that to the query runner's skill selector as a candidate tool."""
        registry.register(_make_platform_skill(), account_id=None)
        await _create_app(client)

        resp = await client.post(
            "/namespaces/default/applications/my-app/skills",
            json={"skill_name": "onboarding-interview"},
        )

        assert resp.status_code == 422
        assert ONBOARDING_SURFACE in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_a_platform_surface_skill_is_refused_in_config_yaml(self, client, registry):
        """The other door into config.skills — bundle upload — is guarded too."""
        registry.register(_make_platform_skill(), account_id=None)

        bundle = _make_bundle(_BASE_CONFIG + "skills:\n  - onboarding-interview\n")
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.post(
                "/namespaces/default/applications",
                files={"bundle": ("app.zip", bundle, "application/zip")},
            )

        assert resp.status_code == 422
        assert "onboarding-interview" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_adding_multiple_skills(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-beta"})
        assert resp.status_code == 201
        assert _skill_names(resp.json()) == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_idempotent_when_skill_already_assigned(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        assert resp.status_code == 201
        names = [s["name"] for s in resp.json()["skills"]]
        assert names.count("skill-alpha") == 1

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.post(
            "/namespaces/default/applications/ghost/skills",
            json={"skill_name": "skill-alpha"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_skill_not_in_registry(self, client):
        await _create_app(client)
        resp = await client.post(
            "/namespaces/default/applications/my-app/skills",
            json={"skill_name": "nonexistent-skill"},
        )
        assert resp.status_code == 404
        assert "nonexistent-skill" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /applications/{name}/skills/{skill_id}
# ---------------------------------------------------------------------------

class TestRemoveApplicationSkill:
    @pytest.mark.asyncio
    async def test_returns_204_on_success(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        resp = await client.delete("/namespaces/default/applications/my-app/skills/skill-alpha")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_skill_removed_from_config_yaml(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        await client.delete("/namespaces/default/applications/my-app/skills/skill-alpha")

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert "skill-alpha" not in _skill_names(resp.json())

    @pytest.mark.asyncio
    async def test_removes_only_specified_skill(self, client):
        await _create_app(client)
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-alpha"})
        await client.post("/namespaces/default/applications/my-app/skills", json={"skill_name": "skill-beta"})
        await client.delete("/namespaces/default/applications/my-app/skills/skill-alpha")

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        names = _skill_names(resp.json())
        assert "skill-alpha" not in names
        assert "skill-beta" in names

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_app(self, client):
        resp = await client.delete("/namespaces/default/applications/ghost/skills/skill-alpha")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_when_skill_not_assigned(self, client):
        await _create_app(client)
        resp = await client.delete("/namespaces/default/applications/my-app/skills/skill-alpha")
        assert resp.status_code == 404
        assert "skill-alpha" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_ref_can_be_unassigned_by_id(self, client, registry):
        # A dangling ref can't be resolved by name (the skill is gone from the
        # registry), so the UI unassigns it by raw skill id instead.
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n  - skill-beta\n"
        await _create_app(client, config)
        registry.unregister("skill-beta")

        resp = await client.delete("/namespaces/default/applications/my-app/skills/skill-beta")
        assert resp.status_code == 204

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert resp.json()["skills"] == [{"id": "skill-alpha", "name": "skill-alpha", "missing": False}]


# ---------------------------------------------------------------------------
# Skills in config.yaml at create / update time
# ---------------------------------------------------------------------------

class TestSkillsInConfig:
    @pytest.mark.asyncio
    async def test_create_with_skills_stores_them(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n  - skill-beta\n"
        await _create_app(client, config)

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        assert _skill_names(resp.json()) == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_create_with_unknown_skill_returns_422(self, client):
        config = _BASE_CONFIG + "skills:\n  - unknown-skill\n"
        bundle = _make_bundle(config)
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.post(
                "/namespaces/default/applications",
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
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.patch(
                "/namespaces/default/applications/my-app",
                files={"bundle": ("bundle.zip", bundle_v2, "application/zip")},
            )
        assert resp.status_code == 200

        resp = await client.get("/namespaces/default/applications/my-app/skills")
        names = _skill_names(resp.json())
        assert "skill-beta" in names
        assert "skill-alpha" not in names

    @pytest.mark.asyncio
    async def test_update_with_unknown_skill_returns_422(self, client):
        await _create_app(client)
        config_v2 = _BASE_CONFIG + "skills:\n  - ghost-skill\n"
        bundle_v2 = _make_bundle(config_v2)
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.patch(
                "/namespaces/default/applications/my-app",
                files={"bundle": ("bundle.zip", bundle_v2, "application/zip")},
            )
        assert resp.status_code == 422
        assert "ghost-skill" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_skills_visible_in_app_config_response(self, client):
        config = _BASE_CONFIG + "skills:\n  - skill-alpha\n"
        await _create_app(client, config)

        resp = await client.get("/namespaces/default/applications/my-app")
        assert resp.status_code == 200
        assert "skill-alpha" in resp.json()["config"]["skills"]
