"""Tests for the skill upload/CRUD APIs:

  POST   /skills          — upload a ZIP bundle (assigns a UUID)
  PUT    /skills/{id}      — replace an existing skill's bundle
  GET    /skills/{id}      — fetch one skill
  DELETE /skills/{id}      — remove from store, cache, and registry
"""

from __future__ import annotations

import io
import zipfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import (
    get_skill_bundle_store,
    get_skill_registry,
    get_system_store,
)
from api.main import app
from api.system_store import SystemStore
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import Skill
from cogbase.skills.store import SkillBundleStore
from cogbase.stores.document.local_fs import LocalFSDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


VALID_MD = "---\nname: greeter\ndescription: Says hi.\n---\n# Greeter\nRun `python hello.py`.\n"


@pytest_asyncio.fixture
async def ctx(tmp_path):
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    registry = SkillRegistry()
    bundle_store = SkillBundleStore(LocalFSDocumentStore(tmp_path / "docs"), cache_dir=tmp_path / "cache")

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_skill_registry] = lambda: registry
    app.dependency_overrides[get_skill_bundle_store] = lambda: bundle_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, system_store, registry, bundle_store

    app.dependency_overrides.clear()


async def _upload(client, files: dict[str, str]):
    return await client.post(
        "/skills",
        files={"bundle": ("skill.zip", _zip(files), "application/zip")},
    )


class TestUploadSkill:
    @pytest.mark.asyncio
    async def test_upload_assigns_id_and_persists(self, ctx):
        client, system_store, registry, bundle_store = ctx
        resp = await _upload(client, {"SKILL.md": VALID_MD, "hello.py": "print('hi')\n"})
        assert resp.status_code == 201
        body = resp.json()
        skill_id = body["id"]
        assert body["name"] == "greeter"

        # Registered in memory, recorded in the system store, materialized locally.
        assert registry.get(skill_id).name == "greeter"
        assert await system_store.get_skill(skill_id) is not None
        assert (bundle_store.skill_dir(skill_id) / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_upload_rejects_bundle_without_skill_md(self, ctx):
        client, system_store, registry, _ = ctx
        resp = await _upload(client, {"readme.txt": "nope"})
        assert resp.status_code == 422
        assert registry.all_skills() == []
        assert await system_store.list_skills() == []

    @pytest.mark.asyncio
    async def test_upload_rejects_invalid_frontmatter(self, ctx):
        client, *_ = ctx
        resp = await _upload(client, {"SKILL.md": "# no front matter\n"})
        assert resp.status_code == 422


class TestReplaceGetDeleteSkill:
    @pytest.mark.asyncio
    async def test_put_keeps_id_and_updates_metadata(self, ctx):
        client, system_store, registry, _ = ctx
        skill_id = (await _upload(client, {"SKILL.md": VALID_MD})).json()["id"]

        updated_md = "---\nname: greeter\ndescription: Says hello v2.\n---\n# Greeter v2\n"
        resp = await client.put(
            f"/skills/{skill_id}",
            files={"bundle": ("skill.zip", _zip({"SKILL.md": updated_md}), "application/zip")},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == skill_id
        assert registry.get(skill_id).description == "Says hello v2."
        assert len(await system_store.list_skills()) == 1  # replaced, not duplicated

    @pytest.mark.asyncio
    async def test_put_unknown_id_returns_404(self, ctx):
        client, *_ = ctx
        resp = await client.put(
            "/skills/ghost",
            files={"bundle": ("skill.zip", _zip({"SKILL.md": VALID_MD}), "application/zip")},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_skill(self, ctx):
        client, *_ = ctx
        skill_id = (await _upload(client, {"SKILL.md": VALID_MD})).json()["id"]
        resp = await client.get(f"/skills/{skill_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "greeter"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_404(self, ctx):
        client, *_ = ctx
        assert (await client.get("/skills/ghost")).status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_everywhere(self, ctx):
        client, system_store, registry, bundle_store = ctx
        skill_id = (await _upload(client, {"SKILL.md": VALID_MD})).json()["id"]

        resp = await client.delete(f"/skills/{skill_id}")
        assert resp.status_code == 204
        with pytest.raises(KeyError):
            registry.get(skill_id)
        assert await system_store.get_skill(skill_id) is None
        assert not bundle_store.skill_dir(skill_id).exists()

    @pytest.mark.asyncio
    async def test_delete_unknown_returns_404(self, ctx):
        client, *_ = ctx
        assert (await client.delete("/skills/ghost")).status_code == 404


class TestBuiltinSkillsAreReadOnly:
    """Built-in (skills_dir) skills are registered but live in no system store;
    PUT/DELETE must reject them with a clear 403 rather than a misleading 404."""

    def _register_builtin(self, registry, skill_id="builtin-skill"):
        registry.register(
            Skill(
                name=skill_id,
                description="A built-in skill.",
                raw_markdown=f"---\nname: {skill_id}\ndescription: d\n---\n# {skill_id}\n",
                id=skill_id,
                builtin=True,
            )
        )

    @pytest.mark.asyncio
    async def test_list_exposes_builtin_flag(self, ctx):
        client, _, registry, _bs = ctx
        self._register_builtin(registry)
        await _upload(client, {"SKILL.md": VALID_MD})

        by_name = {s["name"]: s for s in (await client.get("/skills")).json()["skills"]}
        assert by_name["builtin-skill"]["builtin"] is True
        assert by_name["greeter"]["builtin"] is False

    @pytest.mark.asyncio
    async def test_put_builtin_returns_403(self, ctx):
        client, _, registry, _bs = ctx
        self._register_builtin(registry)
        resp = await client.put(
            "/skills/builtin-skill",
            files={"bundle": ("skill.zip", _zip({"SKILL.md": VALID_MD}), "application/zip")},
        )
        assert resp.status_code == 403
        assert "built-in" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_builtin_returns_403_and_keeps_skill(self, ctx):
        client, _, registry, _bs = ctx
        self._register_builtin(registry)
        resp = await client.delete("/skills/builtin-skill")
        assert resp.status_code == 403
        assert registry.get("builtin-skill").builtin is True
