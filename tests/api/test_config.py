"""Unit tests for cogbase/config/config.py."""

from __future__ import annotations

import os
import textwrap

import pytest

from cogbase.config.config import (
    AppConfig,
    ChunkerConfig,
    DocumentCollectionConfig,
    EmbeddingConfig,
    LLMConfig,
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
        assert cfg.type == "faiss"

    def test_pgvector_requires_url(self):
        with pytest.raises(Exception, match="url is required"):
            VectorStoreConfig(type="pgvector")

    def test_pgvector_with_url_valid(self):
        cfg = VectorStoreConfig(type="pgvector", url="postgresql://localhost/db")
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
    chunk_collections:
      - name: doc_chunks
        chunker:
          type: fixed
          chunk_size: 256
          overlap: 32
""")


class TestAppConfig:
    def test_from_yaml_minimal(self):
        cfg = AppConfig.from_yaml(_MINIMAL_YAML)
        assert cfg.name == "test-app"
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.structured_store is None
        assert cfg.vector_store is None
        assert cfg.embedding is None
        assert cfg.chunk_collections == []

    def test_from_yaml_full(self):
        cfg = AppConfig.from_yaml(_FULL_YAML)
        assert cfg.name == "full-app"
        assert cfg.embedding is not None
        assert len(cfg.chunk_collections) == 1
        assert cfg.chunk_collections[0].chunker.chunk_size == 256

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
            embedding:
              provider: openai
              model: text-embedding-3-small
            chunk_collections:
              - name: doc_chunks
                chunker:
                  type: fixed
                  chunk_size: 512
                  overlap: 64
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.structured_store.type == "sqlite"
        assert cfg.structured_store.path == "./data/my.db"
        assert cfg.vector_store.type == "faiss"

    def test_vector_collections_without_embedding_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            chunk_collections:
              - name: doc_chunks
                chunker:
                  type: fixed
                  chunk_size: 512
                  overlap: 64
        """)
        with pytest.raises(Exception, match="embedding is required when chunk_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_embedding_alone_is_valid(self):
        yaml_text = textwrap.dedent("""\
            name: ok-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.embedding is not None
        assert cfg.chunk_collections == []

    def test_from_yaml_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            AppConfig.from_yaml("- item1\n- item2\n")

    def test_document_collections_empty_by_default(self):
        cfg = AppConfig.from_yaml(_MINIMAL_YAML)
        assert cfg.document_collections == []

    def test_pipeline_step_literal_includes_document_embed(self):
        yaml_text = textwrap.dedent("""\
            name: ok-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            document_collections:
              - name: doc_summary
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: doc_summary
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.pipeline.steps[0].tool == "document-embed-upsert"
        assert cfg.pipeline.steps[0].collection == "doc_summary"

    def test_document_collections_without_embedding_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            document_collections:
              - name: doc_summary
        """)
        with pytest.raises(Exception, match="embedding is required when document_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_step_references_unknown_document_collection_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            document_collections:
              - name: doc_summary
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown document collection"):
            AppConfig.from_yaml(yaml_text)

    def test_full_three_step_config_parses(self):
        _SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: contracts
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            chunk_collections:
              - name: document_chunks
                chunker:
                  type: fixed
                  chunk_size: 512
                  overlap: 64
            structured_collections:
              - name: contract_extraction
                schema: '{_SCHEMA}'
                extractor:
                  type: llm
            document_collections:
              - name: document_summary
                prompt: "Summarize in one sentence."
                max_tokens: 128
            pipeline:
              parallel: false
              steps:
                - tool: chunk-embed-upsert
                  collection: document_chunks
                - tool: extract-structured
                  collection: contract_extraction
                - tool: document-embed-upsert
                  collection: document_summary
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.chunk_collections) == 1
        assert len(cfg.structured_collections) == 1
        assert len(cfg.document_collections) == 1
        assert cfg.document_collections[0].name == "document_summary"
        assert cfg.document_collections[0].prompt == "Summarize in one sentence."
        assert cfg.document_collections[0].max_tokens == 128
        assert len(cfg.pipeline.steps) == 3
        tools = [s.tool for s in cfg.pipeline.steps]
        assert tools == ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]


# ---------------------------------------------------------------------------
# DocumentCollectionConfig
# ---------------------------------------------------------------------------

class TestDocumentCollectionConfig:
    def test_defaults(self):
        cfg = DocumentCollectionConfig(name="s")
        assert cfg.name == "s"
        assert cfg.prompt is None
        assert cfg.max_tokens == 1024
        assert cfg.metadata_fields == []

    def test_custom_values(self):
        cfg = DocumentCollectionConfig(
            name="doc_summary",
            prompt="One sentence please.",
            max_tokens=64,
        )
        assert cfg.prompt == "One sentence please."
        assert cfg.max_tokens == 64

    def test_metadata_fields(self):
        cfg = DocumentCollectionConfig(
            name="meetings",
            metadata_fields=["customer_id", "deal_stage"],
        )
        assert cfg.metadata_fields == ["customer_id", "deal_stage"]
