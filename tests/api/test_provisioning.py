"""Unit tests for api/provisioning.py — the default starter workspace.

Signup integration (that a fresh account lands with the workspace) is covered in
tests/api/test_auth.py; here we drive ``provision_default_workspace`` directly
with fully-wired resources to prove the happy path: an *active* contract-analyst
app with no documents ingested.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.app_cache import AppCache, cache_key
from api.provisioning import DEFAULT_NAMESPACE_NAME, provision_default_workspace
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.stores.document.memory import InMemoryDocumentStore
from cogbase.stores.structured.memory import InMemoryStructuredStore


def _mock_scoped_store() -> MagicMock:
    store = MagicMock()
    store.with_scope.return_value = store
    store.create_collection = AsyncMock()
    store.register_schema = MagicMock()
    return store


def _full_resources() -> SystemResources:
    return SystemResources(
        structured_store=_mock_scoped_store(),
        vector_store=_mock_scoped_store(),
        document_store=InMemoryDocumentStore(),
        llm=MagicMock(),
        embedder=MagicMock(),
        skill_registry=None,
    )


@pytest.mark.asyncio
async def test_provisions_active_app_with_no_documents():
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    app_cache = AppCache()
    account_id = "acct_test"

    await provision_default_workspace(
        account_id,
        system_store=system_store,
        app_cache=app_cache,
        system_resources=_full_resources(),
    )

    ns = await system_store.get_namespace(account_id, DEFAULT_NAMESPACE_NAME)
    assert ns is not None and ns.name == DEFAULT_NAMESPACE_NAME

    app = await system_store.get_app(account_id, DEFAULT_NAMESPACE_NAME, "contract-analyst")
    assert app is not None and app.status == "active"

    # The live instance is cached and no demo documents were ingested.
    assert app_cache.get(cache_key(account_id, DEFAULT_NAMESPACE_NAME, "contract-analyst")) is not None
    assert await system_store.list_docs(app.app_id) == []


@pytest.mark.asyncio
async def test_provisioning_is_idempotent():
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    account_id = "acct_test"

    for _ in range(2):
        await provision_default_workspace(
            account_id,
            system_store=system_store,
            app_cache=AppCache(),
            system_resources=_full_resources(),
        )

    apps = await system_store.list_apps(account_id, DEFAULT_NAMESPACE_NAME)
    assert [a.name for a in apps] == ["contract-analyst"]
