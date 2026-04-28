"""Unit tests for api/system_store.py — SystemStore and AppRecord."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cogbase.stores.structured.memory import InMemoryStructuredStore
from api.system_store import AppRecord, SystemStore


def _make_record(name: str = "my-app", status: str = "active") -> AppRecord:
    return AppRecord(
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
        fetched = await store.get_app("my-app")
        assert fetched is not None
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
        fetched = await store.get_app("my-app")
        assert fetched.status == "error"
        assert fetched.error == "something failed"


class TestSystemStoreListApps:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        assert await store.list_apps() == []

    @pytest.mark.asyncio
    async def test_list_returns_all(self, store):
        await store.save_app(_make_record(name="a"))
        await store.save_app(_make_record(name="b"))
        apps = await store.list_apps()
        assert len(apps) == 2
        names = {r.name for r in apps}
        assert names == {"a", "b"}


class TestSystemStoreDeleteApp:
    @pytest.mark.asyncio
    async def test_delete_removes_record(self, store):
        await store.save_app(_make_record(name="my-app"))
        await store.delete_app("my-app")
        assert await store.get_app("my-app") is None

    @pytest.mark.asyncio
    async def test_delete_only_removes_target(self, store):
        await store.save_app(_make_record(name="a"))
        await store.save_app(_make_record(name="b"))
        await store.delete_app("a")
        assert await store.get_app("a") is None
        assert await store.get_app("b") is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, store):
        # Must not raise
        await store.delete_app("ghost")
