"""Unit tests for api/routers/system.py — GET and PATCH /system/config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.app_cache import AppCache
from api.dependencies import get_app_cache, get_system_resources, get_system_store, get_skill_registry
from api.main import app
from api.system_resources import SystemResources
from api.system_store import SystemStore
from cogbase.config.models import EmbeddingConfig, LLMConfig
from cogbase.skills.registry import SkillRegistry
from cogbase.stores.structured.memory import InMemoryStructuredStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_llm_config(**kwargs) -> LLMConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-testtesttest",
        base_url="https://api.openai.com/v1",
    )
    return LLMConfig(**{**defaults, **kwargs})


def _make_embedding_config(**kwargs) -> EmbeddingConfig:
    defaults = dict(
        provider="openai",
        model="text-embedding-3-small",
        api_key="sk-testtesttest",
        dimensions=1536,
        base_url="https://api.openai.com/v1",
    )
    return EmbeddingConfig(**{**defaults, **kwargs})


@pytest_asyncio.fixture
async def client_with_resources(request):
    """Yield (AsyncClient, SystemResources, AppCache, SystemStore)."""
    resources: SystemResources = getattr(request, "param", None) or SystemResources()
    system_store = SystemStore(store=InMemoryStructuredStore())
    await system_store.setup()
    app_cache = AppCache()

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_resources] = lambda: resources
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, resources, app_cache, system_store

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get(client: AsyncClient) -> dict:
    resp = await client.get("/system/config")
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# _mask_key
# ---------------------------------------------------------------------------

class TestMaskKey:
    """Unit tests for the private _mask_key helper — imported directly."""

    def test_none_returns_none(self):
        from api.routers.system import _mask_key
        assert _mask_key(None) is None

    def test_empty_string_passthrough(self):
        from api.routers.system import _mask_key
        assert _mask_key("EMPTY") == "EMPTY"

    def test_short_key_passthrough(self):
        from api.routers.system import _mask_key
        assert _mask_key("ab") == "ab"

    def test_exactly_four_chars_passthrough(self):
        from api.routers.system import _mask_key
        assert _mask_key("abcd") == "abcd"

    def test_long_key_masked(self):
        from api.routers.system import _mask_key
        assert _mask_key("sk-1234567890abcd") == "***abcd"

    def test_five_char_key_masked(self):
        from api.routers.system import _mask_key
        assert _mask_key("12345") == "***2345"


# ---------------------------------------------------------------------------
# GET /system/config
# ---------------------------------------------------------------------------

class TestGetSystemConfig:
    @pytest.mark.asyncio
    async def test_returns_nulls_when_no_config(self, client_with_resources):
        client, _, _, _ = client_with_resources
        data = await _get(client)
        assert data["llm"] is None
        assert data["embedding"] is None

    @pytest.mark.asyncio
    async def test_returns_llm_config(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.llm_config = _make_llm_config(model="gpt-4o", api_key="sk-abcdefgh")
        data = await _get(client)
        assert data["llm"]["provider"] == "openai"
        assert data["llm"]["model"] == "gpt-4o"
        assert data["llm"]["api_key"] == "***efgh"
        assert data["embedding"] is None

    @pytest.mark.asyncio
    async def test_returns_embedding_config(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.embedding_config = _make_embedding_config(model="text-embedding-3-large", api_key="sk-xxxx1234")
        data = await _get(client)
        assert data["llm"] is None
        assert data["embedding"]["provider"] == "openai"
        assert data["embedding"]["model"] == "text-embedding-3-large"
        assert data["embedding"]["api_key"] == "***1234"

    @pytest.mark.asyncio
    async def test_returns_both_configs(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.llm_config = _make_llm_config()
        resources.embedding_config = _make_embedding_config()
        data = await _get(client)
        assert data["llm"] is not None
        assert data["embedding"] is not None

    @pytest.mark.asyncio
    async def test_api_key_masked_in_response(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.llm_config = _make_llm_config(api_key="sk-supersecretkey")
        data = await _get(client)
        assert "supersecretkey" not in data["llm"]["api_key"]
        assert data["llm"]["api_key"].startswith("***")

    @pytest.mark.asyncio
    async def test_empty_api_key_passthrough(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.llm_config = _make_llm_config(api_key="EMPTY")
        data = await _get(client)
        assert data["llm"]["api_key"] == "EMPTY"

    @pytest.mark.asyncio
    async def test_mini_model_returned(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        resources.llm_config = _make_llm_config(mini_model="gpt-4o-mini")
        data = await _get(client)
        assert data["llm"]["mini_model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# PATCH /system/config
# ---------------------------------------------------------------------------

class TestPatchSystemConfig:
    @pytest.mark.asyncio
    async def test_requires_at_least_one_field(self, client_with_resources):
        client, _, _, _ = client_with_resources
        resp = await client.patch("/system/config", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_update_llm_config(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        mock_llm = MagicMock()
        with patch("api.routers.system.build_llm", return_value=mock_llm):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm"]["model"] == "gpt-4o"
        assert data["llm"]["provider"] == "openai"
        assert resources.llm is mock_llm
        assert resources.llm_config.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_update_embedding_config(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        mock_embedder = MagicMock()
        with patch("api.routers.system.build_embedding", return_value=mock_embedder):
            resp = await client.patch("/system/config", json={
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                    "dimensions": 3072,
                }
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding"]["model"] == "text-embedding-3-large"
        assert data["embedding"]["dimensions"] == 3072
        assert resources.embedder is mock_embedder
        assert resources.embedding_config.dimensions == 3072

    @pytest.mark.asyncio
    async def test_update_both_llm_and_embedding(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        with patch("api.routers.system.build_llm", return_value=mock_llm), \
             patch("api.routers.system.build_embedding", return_value=mock_embedder):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                },
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                    "dimensions": 1536,
                },
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm"]["model"] == "gpt-4o"
        assert data["embedding"]["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_clears_app_cache_on_success(self, client_with_resources):
        client, _, app_cache, _ = client_with_resources
        app_cache.add("some-app", MagicMock())
        mock_llm = MagicMock()
        with patch("api.routers.system.build_llm", return_value=mock_llm):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 200
        assert app_cache.get("some-app") is None

    @pytest.mark.asyncio
    async def test_bad_llm_config_returns_422(self, client_with_resources):
        client, _, _, _ = client_with_resources
        with patch("api.routers.system.build_llm", side_effect=ValueError("bad provider")):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 422
        assert "bad provider" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_bad_embedding_config_returns_422(self, client_with_resources):
        client, _, _, _ = client_with_resources
        with patch("api.routers.system.build_embedding", side_effect=ValueError("bad model")):
            resp = await client.patch("/system/config", json={
                "embedding": {
                    "provider": "openai",
                    "model": "unknown-model",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                    "dimensions": 512,
                }
            })
        assert resp.status_code == 422
        assert "bad model" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_response_masks_api_key(self, client_with_resources):
        client, _, _, _ = client_with_resources
        mock_llm = MagicMock()
        with patch("api.routers.system.build_llm", return_value=mock_llm):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-supersecretkey",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 200
        api_key = resp.json()["llm"]["api_key"]
        assert "supersecretkey" not in api_key
        assert api_key.startswith("***")

    @pytest.mark.asyncio
    async def test_omitting_llm_leaves_existing_llm_unchanged(self, client_with_resources):
        client, resources, _, _ = client_with_resources
        original_llm = MagicMock()
        resources.llm = original_llm
        resources.llm_config = _make_llm_config(model="gpt-3.5-turbo")

        mock_embedder = MagicMock()
        with patch("api.routers.system.build_embedding", return_value=mock_embedder):
            resp = await client.patch("/system/config", json={
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                    "dimensions": 1536,
                }
            })
        assert resp.status_code == 200
        assert resources.llm is original_llm
        assert resources.llm_config.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_llm_override_persisted_to_system_store(self, client_with_resources):
        client, _, _, system_store = client_with_resources
        mock_llm = MagicMock()
        with patch("api.routers.system.build_llm", return_value=mock_llm):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 200
        overrides = await system_store.load_system_config_overrides()
        assert "llm" in overrides
        from cogbase.config.models import LLMConfig
        restored = LLMConfig.model_validate_json(overrides["llm"])
        assert restored.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_embedding_override_persisted_to_system_store(self, client_with_resources):
        client, _, _, system_store = client_with_resources
        mock_embedder = MagicMock()
        with patch("api.routers.system.build_embedding", return_value=mock_embedder):
            resp = await client.patch("/system/config", json={
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                    "dimensions": 3072,
                }
            })
        assert resp.status_code == 200
        overrides = await system_store.load_system_config_overrides()
        assert "embedding" in overrides
        from cogbase.config.models import EmbeddingConfig
        restored = EmbeddingConfig.model_validate_json(overrides["embedding"])
        assert restored.model == "text-embedding-3-large"
        assert restored.dimensions == 3072

    @pytest.mark.asyncio
    async def test_failed_build_does_not_persist_override(self, client_with_resources):
        client, _, _, system_store = client_with_resources
        with patch("api.routers.system.build_llm", side_effect=ValueError("bad")):
            resp = await client.patch("/system/config", json={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-abcdefgh",
                    "base_url": "https://api.openai.com/v1",
                }
            })
        assert resp.status_code == 422
        overrides = await system_store.load_system_config_overrides()
        assert "llm" not in overrides
