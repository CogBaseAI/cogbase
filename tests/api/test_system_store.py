"""Unit tests for api/system_store.py — SystemStore and AppRecord."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cogbase.stores.structured.memory import InMemoryStructuredStore
from api.system_store import AppRecord, SystemStore


def _make_record(app_id: str = "app-1", name: str = "my-app", status: str = "active") -> AppRecord:
    return AppRecord(
        app_id=app_id,
        name=name,
        config_yaml="name: my-app\nllm:\n  model: gpt-4o-mini\n",
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest_asyncio.fixture
async def store() -> SystemStore:
    backend = InMemoryStructuredStore()
    ss = SystemStore(store=backend)
    await ss.setup()
    return ss


class TestSystemStoreSetup:
    @pytest.mark.asyncio
    async def test_setup_idempotent(self):
        backend = InMemoryStructuredStore()
        ss = SystemStore(store=backend)
        await ss.setup()
        await ss.setup()  # second call must not raise
        assert await ss.list_apps() == []


class TestSystemStoreSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_get_app(self, store):
        record = _make_record()
        await store.save_app(record)
        fetched = await store.get_app("app-1")
        assert fetched is not None
        assert fetched.app_id == "app-1"
        assert fetched.name == "my-app"
        assert fetched.status == "active"

    @pytest.mark.asyncio
    async def test_get_app_returns_none_for_unknown(self, store):
        result = await store.get_app("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_upserts_record(self, store):
        record = _make_record()
        await store.save_app(record)
        updated = record.model_copy(update={"status": "error", "error": "something failed"})
        await store.save_app(updated)
        fetched = await store.get_app("app-1")
        assert fetched.status == "error"
        assert fetched.error == "something failed"


class TestSystemStoreGetByName:
    @pytest.mark.asyncio
    async def test_get_app_by_name_found(self, store):
        await store.save_app(_make_record(app_id="app-1", name="alpha"))
        await store.save_app(_make_record(app_id="app-2", name="beta"))
        found = await store.get_app_by_name("beta")
        assert found is not None
        assert found.app_id == "app-2"

    @pytest.mark.asyncio
    async def test_get_app_by_name_not_found(self, store):
        assert await store.get_app_by_name("ghost") is None

    @pytest.mark.asyncio
    async def test_get_app_by_name_returns_first_match(self, store):
        await store.save_app(_make_record(app_id="app-1", name="alpha"))
        found = await store.get_app_by_name("alpha")
        assert found.app_id == "app-1"


class TestSystemStoreListApps:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        assert await store.list_apps() == []

    @pytest.mark.asyncio
    async def test_list_returns_all(self, store):
        await store.save_app(_make_record(app_id="app-1", name="a"))
        await store.save_app(_make_record(app_id="app-2", name="b"))
        apps = await store.list_apps()
        assert len(apps) == 2
        ids = {r.app_id for r in apps}
        assert ids == {"app-1", "app-2"}


class TestSystemStoreDeleteApp:
    @pytest.mark.asyncio
    async def test_delete_removes_record(self, store):
        await store.save_app(_make_record(app_id="app-1"))
        await store.delete_app("app-1")
        assert await store.get_app("app-1") is None

    @pytest.mark.asyncio
    async def test_delete_only_removes_target(self, store):
        await store.save_app(_make_record(app_id="app-1", name="a"))
        await store.save_app(_make_record(app_id="app-2", name="b"))
        await store.delete_app("app-1")
        assert await store.get_app("app-1") is None
        assert await store.get_app("app-2") is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, store):
        # Must not raise
        await store.delete_app("ghost")
