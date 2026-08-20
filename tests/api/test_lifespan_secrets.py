"""D2 mitigation: jwt_secret and the provider api_keys may arrive as env vars
instead of the system YAML, are read into process state at startup, and are
deleted from ``os.environ`` before any request can be served — so a
tenant-triggered subprocess (``run_python``/``shell``, via
``cogbase.core.query_runner._tool_env``) has nothing to inherit even if the
allowlist there ever widened by mistake.

Drives ``api.main.lifespan`` for real against a fresh ``FastAPI()`` app and a
temp system config file, rather than mocking around it — the guarantee this
covers only holds if the actual startup sequence enforces it.
"""

from __future__ import annotations

import os
import textwrap

import pytest
from fastapi import FastAPI

from api.auth import get_jwt_secret
from api.main import lifespan

_ENV_VARS = ("COGBASE_JWT_SECRET", "COGBASE_LLM_API_KEY", "COGBASE_EMBEDDING_API_KEY")


@pytest.fixture
def system_config_file(tmp_path):
    config_file = tmp_path / "system.yaml"
    config_file.write_text(textwrap.dedent("""\
        system_db:
          type: memory
        llm:
          provider: openai-compatible
          model: test-model
          base_url: http://localhost:0/v1
          api_key: yaml-placeholder-not-a-real-key
        embedding:
          provider: openai-compatible
          model: test-embedding-model
          base_url: http://localhost:0/v1
          api_key: yaml-placeholder-not-a-real-key
    """))
    return config_file


@pytest.mark.asyncio
async def test_env_vars_override_yaml_and_are_deleted_before_serving(
    system_config_file, monkeypatch
):
    monkeypatch.setenv("COGBASE_CONFIG", str(system_config_file))
    monkeypatch.setenv("COGBASE_JWT_SECRET", "env-jwt-secret")
    monkeypatch.setenv("COGBASE_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("COGBASE_EMBEDDING_API_KEY", "env-embedding-key")

    app = FastAPI()
    async with lifespan(app):
        # Consumed into in-memory state...
        assert get_jwt_secret() == "env-jwt-secret"
        assert app.state.system_resources.llm_config.api_key == "env-llm-key"
        assert app.state.system_resources.embedding_config.api_key == "env-embedding-key"

        # ...and gone from the process environment before any request is served.
        for var in _ENV_VARS:
            assert var not in os.environ


@pytest.mark.asyncio
async def test_yaml_value_used_when_env_var_absent(system_config_file, monkeypatch):
    monkeypatch.setenv("COGBASE_CONFIG", str(system_config_file))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    app = FastAPI()
    async with lifespan(app):
        assert app.state.system_resources.llm_config.api_key == "yaml-placeholder-not-a-real-key"
        assert app.state.system_resources.embedding_config.api_key == "yaml-placeholder-not-a-real-key"


class TestPostgresDsnOverride:
    """``_apply_secret_env_overrides`` is pure w.r.t. ``os.environ`` (an
    explicit mapping is passed in), so the Postgres-DSN half of D2's
    secrets-never-on-disk rule is testable without a real database — unlike
    the jwt/api-key half above, which drives the full ``lifespan`` because
    ``type: memory`` needs no real connection. ``type: postgres`` does.
    """

    def _cfg(self, **overrides):
        # A placeholder url satisfies the postgres/pgvector validators (they
        # require *some* url) so the test can assert the env override
        # replaces it, the same way the YAML template's placeholder does.
        from api.system_config import SystemConfig
        return SystemConfig.model_validate({
            "system_db": {"type": "postgres", "url": "postgresql://placeholder/db"},
            "structured_store": {"type": "postgres", "url": "postgresql://placeholder/db"},
            "vector_store": {"type": "pgvector", "url": "postgresql://placeholder/db"},
            **overrides,
        })

    def test_postgres_dsn_overrides_system_db_and_structured_store(self):
        from api.main import _apply_secret_env_overrides
        cfg = self._cfg()
        _apply_secret_env_overrides(cfg, {"COGBASE_POSTGRES_DSN": "postgresql://u:p@host/db"})
        assert cfg.system_db.url == "postgresql://u:p@host/db"
        assert cfg.structured_store.url == "postgresql://u:p@host/db"

    def test_pgvector_dsn_falls_back_to_postgres_dsn(self):
        from api.main import _apply_secret_env_overrides
        cfg = self._cfg()
        _apply_secret_env_overrides(cfg, {"COGBASE_POSTGRES_DSN": "postgresql://u:p@host/db"})
        assert cfg.vector_store.url == "postgresql://u:p@host/db"

    def test_pgvector_dsn_env_var_takes_priority_over_postgres_dsn(self):
        from api.main import _apply_secret_env_overrides
        cfg = self._cfg()
        _apply_secret_env_overrides(cfg, {
            "COGBASE_POSTGRES_DSN": "postgresql://u:p@host/db",
            "COGBASE_PGVECTOR_DSN": "postgresql://u:p@vector-host/db",
        })
        assert cfg.vector_store.url == "postgresql://u:p@vector-host/db"
        assert cfg.structured_store.url == "postgresql://u:p@host/db"

    def test_sqlite_and_faiss_types_are_untouched(self):
        from api.main import _apply_secret_env_overrides
        cfg = self._cfg(
            system_db={"type": "sqlite", "path": "./x.db"},
            structured_store=None,
            vector_store=None,
        )
        _apply_secret_env_overrides(cfg, {"COGBASE_POSTGRES_DSN": "postgresql://u:p@host/db"})
        assert cfg.system_db.url is None

    @pytest.mark.asyncio
    async def test_lifespan_deletes_dsn_env_vars_before_serving(self, tmp_path, monkeypatch):
        config_file = tmp_path / "system.yaml"
        config_file.write_text("system_db:\n  type: memory\n")
        monkeypatch.setenv("COGBASE_CONFIG", str(config_file))
        monkeypatch.setenv("COGBASE_POSTGRES_DSN", "postgresql://u:p@host/db")
        monkeypatch.setenv("COGBASE_PGVECTOR_DSN", "postgresql://u:p@vector-host/db")

        app = FastAPI()
        async with lifespan(app):
            assert "COGBASE_POSTGRES_DSN" not in os.environ
            assert "COGBASE_PGVECTOR_DSN" not in os.environ
