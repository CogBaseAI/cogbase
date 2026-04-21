"""Integration tests for the /applications REST endpoints.

All tests use httpx.AsyncClient pointed at the FastAPI app with dependency
overrides injected — no real LLM calls or file I/O happens.
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import (
    get_registry,
    get_system_config,
    get_system_store,
    get_system_structured_store,
)
from api.main import app
from api.registry import AppRegistry
from api.system_config import SystemConfig
from api.system_store import SystemStore
from cogbase.stores.structured.memory import InMemoryStructuredStore


# ---------------------------------------------------------------------------
# Fixtures — lightweight dependency overrides
# ---------------------------------------------------------------------------

def _make_system_store() -> SystemStore:
    backend = InMemoryStructuredStore()
    return SystemStore(store=backend)


def _make_registry() -> AppRegistry:
    return AppRegistry()


def _system_config() -> SystemConfig:
    return SystemConfig.model_validate({"system_db": {"type": "memory"}})


_VALID_YAML = textwrap.dedent("""\
    name: my-contract-analyzer
    llm:
      provider: openai
      model: gpt-4o-mini
    pack:
      name: legal.contract_analyst
""").encode()


def _mock_app_instance() -> MagicMock:
    """Minimal mock that satisfies the build_app / app lifecycle contract."""
    inst = MagicMock()
    inst.setup = AsyncMock()
    return inst


@pytest_asyncio.fixture
async def client():
    """AsyncClient with all external dependencies swapped out."""
    system_store = _make_system_store()
    await system_store.setup()
    registry = _make_registry()
    system_structured_store = InMemoryStructuredStore()

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_system_config] = lambda: _system_config()
    app.dependency_overrides[get_system_structured_store] = lambda: system_structured_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /applications
# ---------------------------------------------------------------------------

class TestCreateApplication:
    @pytest.mark.asyncio
    async def test_create_returns_201(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-contract-analyzer"
        assert data["status"] == "active"
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_create_stores_config_in_response(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert resp.status_code == 201
        config = resp.json()["config"]
        assert config["name"] == "my-contract-analyzer"

    @pytest.mark.asyncio
    async def test_create_conflict_returns_409(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
            resp = await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_invalid_yaml_returns_422(self, client):
        bad_yaml = b"not: valid: yaml: app: config\n"
        resp = await client.post(
            "/applications",
            files={"config_file": ("config.yaml", bad_yaml, "application/yaml")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_records_error_status_when_setup_fails(self, client):
        failing_app = MagicMock()
        failing_app.setup = AsyncMock(side_effect=RuntimeError("setup boom"))
        with patch("api.routers.applications.build_app", return_value=failing_app):
            resp = await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "error"
        assert "boom" in data["error"]

    @pytest.mark.asyncio
    async def test_create_non_mapping_yaml_returns_422(self, client):
        resp = await client.post(
            "/applications",
            files={"config_file": ("config.yaml", b"- item1\n- item2\n", "application/yaml")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /applications
# ---------------------------------------------------------------------------

class TestListApplications:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/applications")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applications"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_created_apps(self, client):
        yaml_a = b"name: app-a\nllm:\n  model: gpt-4o-mini\npack:\n  name: legal.contract_analyst\n"
        yaml_b = b"name: app-b\nllm:\n  model: gpt-4o-mini\npack:\n  name: legal.contract_analyst\n"
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post("/applications", files={"config_file": ("a.yaml", yaml_a, "application/yaml")})
            await client.post("/applications", files={"config_file": ("b.yaml", yaml_b, "application/yaml")})
        resp = await client.get("/applications")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        names = {a["name"] for a in body["applications"]}
        assert names == {"app-a", "app-b"}


# ---------------------------------------------------------------------------
# GET /applications/{app_name}
# ---------------------------------------------------------------------------

class TestGetApplication:
    @pytest.mark.asyncio
    async def test_get_existing(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        resp = await client.get("/applications/my-contract-analyzer")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my-contract-analyzer"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, client):
        resp = await client.get("/applications/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# PATCH /applications/{app_name}
# ---------------------------------------------------------------------------

class TestUpdateApplication:
    @pytest.mark.asyncio
    async def test_update_success(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )

        updated_yaml = _VALID_YAML.replace(b"gpt-4o-mini", b"gpt-4o")
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/my-contract-analyzer",
                files={"config_file": ("config.yaml", updated_yaml, "application/yaml")},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_404(self, client):
        resp = await client.patch(
            "/applications/ghost",
            files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_name_conflict_returns_409(self, client):
        yaml_a = b"name: app-a\nllm:\n  model: gpt-4o-mini\npack:\n  name: legal.contract_analyst\n"
        yaml_b = b"name: app-b\nllm:\n  model: gpt-4o-mini\npack:\n  name: legal.contract_analyst\n"
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post("/applications", files={"config_file": ("a.yaml", yaml_a, "application/yaml")})
            await client.post("/applications", files={"config_file": ("b.yaml", yaml_b, "application/yaml")})

        # Try to rename app-a to app-b (already taken)
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/app-a",
                files={"config_file": ("config.yaml", yaml_b, "application/yaml")},
            )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_records_error_when_setup_fails(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )

        failing_app = MagicMock()
        failing_app.setup = AsyncMock(side_effect=RuntimeError("update boom"))
        with patch("api.routers.applications.build_app", return_value=failing_app):
            resp = await client.patch(
                "/applications/my-contract-analyzer",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# DELETE /applications/{app_name}
# ---------------------------------------------------------------------------

class TestDeleteApplication:
    @pytest.mark.asyncio
    async def test_delete_returns_204(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        resp = await client.delete("/applications/my-contract-analyzer")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        await client.delete("/applications/my-contract-analyzer")
        resp = await client.get("/applications")
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        resp = await client.delete("/applications/ghost")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_from_registry(self, client):
        registry = _make_registry()
        app.dependency_overrides[get_registry] = lambda: registry

        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"config_file": ("config.yaml", _VALID_YAML, "application/yaml")},
            )
        assert registry.get("my-contract-analyzer") is not None

        await client.delete("/applications/my-contract-analyzer")
        assert registry.get("my-contract-analyzer") is None
