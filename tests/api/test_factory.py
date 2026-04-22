"""Unit tests for api/factory.py — build_structured_store and build_app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.config import AppConfig, StructuredStoreConfig, VectorStoreConfig
from api.factory import build_structured_store, build_app
from cogbase.stores.structured.memory import InMemoryStructuredStore
from cogbase.stores.structured.sqlite import SQLiteStructuredStore


# ---------------------------------------------------------------------------
# build_structured_store
# ---------------------------------------------------------------------------

class TestBuildStructuredStore:
    def test_memory_type(self):
        cfg = StructuredStoreConfig(type="memory")
        store = build_structured_store(cfg)
        assert isinstance(store, InMemoryStructuredStore)

    def test_sqlite_type(self, tmp_path):
        cfg = StructuredStoreConfig(type="sqlite", path=str(tmp_path / "test.db"))
        store = build_structured_store(cfg)
        assert isinstance(store, SQLiteStructuredStore)
        store.close()

    def test_sqlite_in_memory(self):
        cfg = StructuredStoreConfig(type="sqlite", path=":memory:")
        store = build_structured_store(cfg)
        assert isinstance(store, SQLiteStructuredStore)
        store.close()


# ---------------------------------------------------------------------------
# build_app — structured store resolution
# ---------------------------------------------------------------------------

_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'

_MINIMAL_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
extraction_schema: '{_SCHEMA}'
"""

_FULL_CONFIG_YAML = f"""\
name: test_app
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
chunker:
  type: fixed
  chunk_size: 512
  overlap: 64
extraction_schema: '{_SCHEMA}'
"""


def _mock_openai_client():
    """Return a minimal mock that satisfies LegalContractApp.__init__."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


class TestBuildAppStructuredStoreResolution:
    @patch("api.factory._build_llm_client")
    def test_raises_when_no_stores(self, mock_llm):
        mock_llm.return_value = _mock_openai_client()
        cfg = AppConfig.from_yaml(_MINIMAL_CONFIG_YAML)
        with pytest.raises(ValueError, match="structured store is required"):
            build_app(cfg)

    @patch("api.factory._build_llm_client")
    def test_uses_system_store_when_no_app_store(self, mock_llm):
        mock_llm.return_value = _mock_openai_client()
        cfg = AppConfig.from_yaml(_MINIMAL_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = build_app(cfg, system_structured_store=system_store)
        structured_store = app._ingest_pipeline._structured_collection.store
        assert structured_store is system_store

    @patch("api.factory._build_llm_client")
    def test_app_config_store_overrides_system_store(self, mock_llm, tmp_path):
        mock_llm.return_value = _mock_openai_client()
        cfg_yaml = _MINIMAL_CONFIG_YAML + (
            "structured_store:\n  type: sqlite\n  path: \":memory:\"\n"
        )
        cfg = AppConfig.from_yaml(cfg_yaml)
        system_store = InMemoryStructuredStore()
        app = build_app(cfg, system_structured_store=system_store)
        structured_store = app._ingest_pipeline._structured_collection.store
        assert isinstance(structured_store, SQLiteStructuredStore)


class TestBuildAppVectorStoreResolution:
    @patch("api.factory._build_llm_client")
    def test_no_vector_store_when_no_embedding(self, mock_llm):
        mock_llm.return_value = _mock_openai_client()
        cfg = AppConfig.from_yaml(_MINIMAL_CONFIG_YAML)
        system_store = InMemoryStructuredStore()
        app = build_app(cfg, system_structured_store=system_store)
        assert app._ingest_pipeline._vector_collection is None

    @patch("api.factory._build_llm_client")
    def test_system_vector_store_cfg_used_when_embedding_present(self, mock_llm):
        mock_llm.return_value = _mock_openai_client()
        cfg = AppConfig.from_yaml(_FULL_CONFIG_YAML)
        sys_vs_cfg = VectorStoreConfig(type="faiss", dim=1536)
        system_store = InMemoryStructuredStore()

        with patch("api.factory._build_embedder") as mock_emb:
            mock_emb.return_value = MagicMock()
            app = build_app(
                cfg,
                system_structured_store=system_store,
                system_vector_store_cfg=sys_vs_cfg,
            )

        assert app._ingest_pipeline._vector_collection is not None

    @patch("api.factory._build_llm_client")
    def test_no_vector_store_when_no_embedding_even_with_system_cfg(self, mock_llm):
        """System vector_store config is ignored when app has no embedding."""
        mock_llm.return_value = _mock_openai_client()
        cfg = AppConfig.from_yaml(_MINIMAL_CONFIG_YAML)
        sys_vs_cfg = VectorStoreConfig(type="faiss", dim=1536)
        system_store = InMemoryStructuredStore()
        app = build_app(
            cfg,
            system_structured_store=system_store,
            system_vector_store_cfg=sys_vs_cfg,
        )
        assert app._ingest_pipeline._vector_collection is None
