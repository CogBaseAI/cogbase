"""Unit tests for api/config.py."""

from __future__ import annotations

import os
import textwrap

import pytest

from api.config import (
    AppConfig,
    ChunkerConfig,
    EmbeddingConfig,
    LLMConfig,
    PackConfig,
    StructuredStoreConfig,
    VectorStoreConfig,
)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_resolved_api_key_explicit(self):
        cfg = LLMConfig(model="gpt-4o-mini", api_key="sk-explicit")
        assert cfg.resolved_api_key() == "sk-explicit"

    def test_resolved_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cfg = LLMConfig(model="gpt-4o-mini")
        assert cfg.resolved_api_key() == "sk-from-env"

    def test_resolved_api_key_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cfg = LLMConfig(model="gpt-4o-mini", api_key="sk-explicit")
        assert cfg.resolved_api_key() == "sk-explicit"

    def test_resolved_api_key_none_when_absent(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(model="gpt-4o-mini")
        assert cfg.resolved_api_key() is None

    def test_default_provider_is_openai(self):
        cfg = LLMConfig(model="gpt-4o-mini")
        assert cfg.provider == "openai"


# ---------------------------------------------------------------------------
# StructuredStoreConfig
# ---------------------------------------------------------------------------

class TestStructuredStoreConfig:
    def test_memory_type_valid(self):
        cfg = StructuredStoreConfig(type="memory")
        assert cfg.type == "memory"

    def test_sqlite_requires_path(self):
        with pytest.raises(Exception, match="path is required"):
            StructuredStoreConfig(type="sqlite")

    def test_sqlite_with_path_valid(self):
        cfg = StructuredStoreConfig(type="sqlite", path="./test.db")
        assert cfg.path == "./test.db"

    def test_postgres_requires_url(self):
        with pytest.raises(Exception, match="url is required"):
            StructuredStoreConfig(type="postgres")

    def test_postgres_with_url_valid(self):
        cfg = StructuredStoreConfig(type="postgres", url="postgresql://localhost/db")
        assert cfg.url == "postgresql://localhost/db"

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            StructuredStoreConfig(type="badtype")


# ---------------------------------------------------------------------------
# VectorStoreConfig
# ---------------------------------------------------------------------------

class TestVectorStoreConfig:
    def test_faiss_default(self):
        cfg = VectorStoreConfig(type="faiss")
        assert cfg.dim == 1536

    def test_faiss_custom_dim(self):
        cfg = VectorStoreConfig(type="faiss", dim=512)
        assert cfg.dim == 512

    def test_pgvector_requires_url(self):
        with pytest.raises(Exception, match="url is required"):
            VectorStoreConfig(type="pgvector")

    def test_pgvector_with_url_valid(self):
        cfg = VectorStoreConfig(type="pgvector", url="postgresql://localhost/db", dim=768)
        assert cfg.url == "postgresql://localhost/db"


# ---------------------------------------------------------------------------
# ChunkerConfig
# ---------------------------------------------------------------------------

class TestChunkerConfig:
    def test_defaults(self):
        cfg = ChunkerConfig()
        assert cfg.type == "fixed"
        assert cfg.chunk_size == 512
        assert cfg.overlap == 64

    def test_custom_values(self):
        cfg = ChunkerConfig(type="langchain", chunk_size=256, overlap=32)
        assert cfg.type == "langchain"
        assert cfg.chunk_size == 256
        assert cfg.overlap == 32


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimensions is None

    def test_custom_dimensions(self):
        cfg = EmbeddingConfig(dimensions=512)
        assert cfg.dimensions == 512


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent("""\
    name: test-app
    llm:
      provider: openai
      model: gpt-4o-mini
""")

_FULL_YAML = textwrap.dedent("""\
    name: full-app
    llm:
      provider: openai
      model: gpt-4o-mini
    embedding:
      provider: openai
      model: text-embedding-3-small
    chunker:
      type: fixed
      chunk_size: 256
      overlap: 32
    pack:
      name: legal.contract_analyst
""")


class TestAppConfig:
    def test_from_yaml_minimal(self):
        cfg = AppConfig.from_yaml(_MINIMAL_YAML)
        assert cfg.name == "test-app"
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.structured_store is None
        assert cfg.vector_store is None
        assert cfg.embedding is None
        assert cfg.chunker is None

    def test_from_yaml_full(self):
        cfg = AppConfig.from_yaml(_FULL_YAML)
        assert cfg.name == "full-app"
        assert cfg.embedding is not None
        assert cfg.chunker is not None
        assert cfg.chunker.chunk_size == 256
        assert cfg.pack.name == "legal.contract_analyst"

    def test_from_yaml_with_explicit_store(self):
        yaml_text = textwrap.dedent("""\
            name: my-app
            llm:
              model: gpt-4o-mini
            structured_store:
              type: sqlite
              path: ./data/my.db
            vector_store:
              type: faiss
              dim: 768
            embedding:
              provider: openai
              model: text-embedding-3-small
            chunker:
              type: fixed
              chunk_size: 512
              overlap: 64
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.structured_store.type == "sqlite"
        assert cfg.structured_store.path == "./data/my.db"
        assert cfg.vector_store.dim == 768

    def test_embedding_without_chunker_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
        """)
        with pytest.raises(Exception, match="embedding and chunker"):
            AppConfig.from_yaml(yaml_text)

    def test_chunker_without_embedding_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            chunker:
              type: fixed
              chunk_size: 512
              overlap: 64
        """)
        with pytest.raises(Exception, match="embedding and chunker"):
            AppConfig.from_yaml(yaml_text)

    def test_vector_store_without_embedding_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            vector_store:
              type: faiss
              dim: 1536
        """)
        with pytest.raises(Exception, match="vector_store requires embedding"):
            AppConfig.from_yaml(yaml_text)

    def test_from_yaml_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            AppConfig.from_yaml("- item1\n- item2\n")

    def test_pack_config_optional(self):
        cfg = AppConfig.from_yaml(_MINIMAL_YAML)
        assert cfg.pack is None
