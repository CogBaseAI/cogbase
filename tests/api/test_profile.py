"""Integration tests for the account-scoped /profile REST endpoints.

Uses httpx.AsyncClient over the FastAPI app with dependency overrides — the
document store and system store are in-memory, so no real I/O happens.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.app_cache import AppCache, cache_key
from api.dependencies import (
    get_app_cache,
    get_skill_registry,
    get_system_resources,
    get_system_store,
)
from api.main import app
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.core.profile import MAX_PROFILE_BYTES
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore

PROFILE_MD = "# Company Profile\n\n**Jurisdictions:** Delaware, England & Wales\n"


class FakeApp:
    """Stands in for a cached ``CogBaseApp``; records profile hot-patches."""

    def __init__(self) -> None:
        self.profile: str | None = None
        self.patches: list[str | None] = []

    def set_account_profile(self, markdown: str | None) -> None:
        self.profile = markdown
        self.patches.append(markdown)


@pytest_asyncio.fixture
async def app_overrides():
    """AsyncClient plus the app cache and document store, for seeding/assertions."""
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    app_cache = AppCache()
    document_store = InMemoryDocumentStore()
    system_resources = SystemResources(
        structured_store=InMemoryStructuredStore(),
        document_store=document_store,
    )

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_resources] = lambda: system_resources
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield {
            "client": ac,
            "system_store": system_store,
            "app_cache": app_cache,
            "document_store": document_store,
        }

    app.dependency_overrides.clear()


class TestGetProfile:
    @pytest.mark.asyncio
    async def test_absent_profile_is_200_not_404(self, app_overrides):
        """Cold start is a state the UI branches on, not an error."""
        resp = await app_overrides["client"].get("/profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["exists"] is False
        assert body["markdown"] is None
        assert body["updated_at"] is None

    @pytest.mark.asyncio
    async def test_round_trip(self, app_overrides):
        client = app_overrides["client"]
        await client.put("/profile", json={"markdown": PROFILE_MD})
        resp = await client.get("/profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["markdown"] == PROFILE_MD
        assert body["exists"] is True
        assert body["source"] == "manual"
        assert body["updated_at"]

    @pytest.mark.asyncio
    async def test_updated_by_null_without_a_token(self, app_overrides):
        """In dev mode tenancy is header-only, so there is no principal to name."""
        client = app_overrides["client"]
        await client.put("/profile", json={"markdown": PROFILE_MD})
        assert (await client.get("/profile")).json()["updated_by"] is None


class TestPutProfile:
    @pytest.mark.asyncio
    async def test_put_returns_the_saved_profile(self, app_overrides):
        resp = await app_overrides["client"].put(
            "/profile", json={"markdown": PROFILE_MD}
        )
        assert resp.status_code == 200
        assert resp.json()["markdown"] == PROFILE_MD
        assert resp.json()["exists"] is True

    @pytest.mark.asyncio
    async def test_put_replaces_previous_version(self, app_overrides):
        client = app_overrides["client"]
        await client.put("/profile", json={"markdown": PROFILE_MD})
        await client.put("/profile", json={"markdown": "# v2\n"})
        assert (await client.get("/profile")).json()["markdown"] == "# v2\n"

    @pytest.mark.asyncio
    async def test_oversized_body_rejected(self, app_overrides):
        """Past the cap the profile crowds out the documents an answer cites."""
        oversized = "x" * (MAX_PROFILE_BYTES + 1)
        resp = await app_overrides["client"].put(
            "/profile", json={"markdown": oversized}
        )
        assert resp.status_code == 413
        # Nothing was persisted, so the account still reads as profile-less.
        assert (await app_overrides["client"].get("/profile")).json()["exists"] is False

    @pytest.mark.asyncio
    async def test_writes_the_index_record(self, app_overrides):
        client, system_store = app_overrides["client"], app_overrides["system_store"]
        await client.put("/profile", json={"markdown": PROFILE_MD})
        record = await system_store.get_profile_record("default")
        assert record is not None
        assert record.source == "manual"
        assert record.updated_at


class TestDeleteProfile:
    @pytest.mark.asyncio
    async def test_delete_removes_body_and_record(self, app_overrides):
        client, system_store = app_overrides["client"], app_overrides["system_store"]
        await client.put("/profile", json={"markdown": PROFILE_MD})

        resp = await client.delete("/profile")
        assert resp.status_code == 204
        assert (await client.get("/profile")).json()["exists"] is False
        assert await system_store.get_profile_record("default") is None

    @pytest.mark.asyncio
    async def test_delete_without_a_profile_is_idempotent(self, app_overrides):
        assert (await app_overrides["client"].delete("/profile")).status_code == 204


class TestAccountIsolation:
    @pytest.mark.asyncio
    async def test_profiles_do_not_leak_across_accounts(self, app_overrides):
        client = app_overrides["client"]
        await client.put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )

        other = await client.get("/profile", headers={"X-Account-Id": "acct-2"})
        assert other.json()["exists"] is False

        mine = await client.get("/profile", headers={"X-Account-Id": "acct-1"})
        assert mine.json()["markdown"] == PROFILE_MD

    @pytest.mark.asyncio
    async def test_delete_only_affects_the_calling_account(self, app_overrides):
        client = app_overrides["client"]
        for acct in ("acct-1", "acct-2"):
            await client.put(
                "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": acct}
            )

        await client.delete("/profile", headers={"X-Account-Id": "acct-1"})
        survivor = await client.get("/profile", headers={"X-Account-Id": "acct-2"})
        assert survivor.json()["exists"] is True


class TestHotPatch:
    """A write reaches already-built app instances instead of evicting them."""

    def _seed_cache(self, app_cache: AppCache) -> dict[str, FakeApp]:
        apps = {
            "acct-1/ns-a/alpha": FakeApp(),
            "acct-1/ns-b/beta": FakeApp(),   # same account, different namespace
            "acct-2/ns-a/gamma": FakeApp(),  # different tenant
        }
        for key, instance in apps.items():
            app_cache.add(key, instance)
        return apps

    @pytest.mark.asyncio
    async def test_put_patches_every_app_in_the_account(self, app_overrides):
        apps = self._seed_cache(app_overrides["app_cache"])
        await app_overrides["client"].put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )
        assert apps["acct-1/ns-a/alpha"].profile == PROFILE_MD
        assert apps["acct-1/ns-b/beta"].profile == PROFILE_MD

    @pytest.mark.asyncio
    async def test_put_leaves_other_accounts_untouched(self, app_overrides):
        apps = self._seed_cache(app_overrides["app_cache"])
        await app_overrides["client"].put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )
        assert apps["acct-2/ns-a/gamma"].patches == []

    @pytest.mark.asyncio
    async def test_instances_survive_the_edit(self, app_overrides):
        """Hot-patch, not evict: the warm instances stay in the cache."""
        app_cache = app_overrides["app_cache"]
        apps = self._seed_cache(app_cache)
        await app_overrides["client"].put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )
        assert app_cache.get("acct-1/ns-a/alpha") is apps["acct-1/ns-a/alpha"]

    @pytest.mark.asyncio
    async def test_delete_clears_the_live_block(self, app_overrides):
        apps = self._seed_cache(app_overrides["app_cache"])
        client = app_overrides["client"]
        await client.put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )
        await client.delete("/profile", headers={"X-Account-Id": "acct-1"})
        assert apps["acct-1/ns-a/alpha"].profile is None
        assert apps["acct-1/ns-a/alpha"].patches == [PROFILE_MD, None]

    @pytest.mark.asyncio
    async def test_rejected_write_does_not_patch(self, app_overrides):
        apps = self._seed_cache(app_overrides["app_cache"])
        await app_overrides["client"].put(
            "/profile",
            json={"markdown": "x" * (MAX_PROFILE_BYTES + 1)},
            headers={"X-Account-Id": "acct-1"},
        )
        assert apps["acct-1/ns-a/alpha"].patches == []

    @pytest.mark.asyncio
    async def test_cache_key_prefix_match_is_exact(self, app_overrides):
        """``acct-1`` must not sweep up ``acct-10``'s instances."""
        app_cache = app_overrides["app_cache"]
        mine, lookalike = FakeApp(), FakeApp()
        app_cache.add(cache_key("acct-1", "ns", "a"), mine)
        app_cache.add(cache_key("acct-10", "ns", "a"), lookalike)

        await app_overrides["client"].put(
            "/profile", json={"markdown": PROFILE_MD}, headers={"X-Account-Id": "acct-1"}
        )
        assert mine.profile == PROFILE_MD
        assert lookalike.patches == []


class TestNoDocumentStore:
    @pytest.mark.asyncio
    async def test_profile_routes_are_503(self, app_overrides):
        """A deployment with no system document store cannot hold profiles."""
        app.dependency_overrides[get_system_resources] = lambda: SystemResources(
            structured_store=InMemoryStructuredStore(), document_store=None
        )
        client = app_overrides["client"]
        assert (await client.get("/profile")).status_code == 503
        assert (
            await client.put("/profile", json={"markdown": PROFILE_MD})
        ).status_code == 503
