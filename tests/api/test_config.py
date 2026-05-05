"""Unit tests for cogbase/config/config.py."""

from __future__ import annotations

import textwrap

import pytest

from cogbase.config.config import (
    AppConfig,
    ChunkerConfig,
    VectorCollectionConfig,
    EmbeddingConfig,
    ExtractorConfig,
    LLMConfig,
    PipelineStepConfig,
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

    def test_faiss_path_valid(self):
        cfg = VectorStoreConfig(type="faiss", path="/tmp/faiss-store")
        assert cfg.path == "/tmp/faiss-store"

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
        assert cfg.type == "langchain"
        assert cfg.chunk_size == 1024
        assert cfg.overlap == 128

    def test_custom_values(self):
        cfg = ChunkerConfig(type="fixed", chunk_size=256, overlap=32)
        assert cfg.type == "fixed"
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
    vector_collections:
      - name: doc_chunks
    pipeline:
      steps:
        - tool: chunk-embed-upsert
          collection: doc_chunks
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
        assert cfg.vector_collections == []

    def test_from_yaml_full(self):
        cfg = AppConfig.from_yaml(_FULL_YAML)
        assert cfg.name == "full-app"
        assert cfg.embedding is not None
        assert len(cfg.vector_collections) == 1
        assert cfg.pipeline.steps[0].chunker.chunk_size == 256

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
            vector_collections:
              - name: doc_chunks
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
            vector_collections:
              - name: doc_chunks
        """)
        with pytest.raises(Exception, match="embedding is required when vector_collections"):
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
        assert cfg.vector_collections == []

    def test_from_yaml_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            AppConfig.from_yaml("- item1\n- item2\n")

    def test_pipeline_step_literal_includes_document_embed(self):
        yaml_text = textwrap.dedent("""\
            name: ok-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: doc_summary
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: doc_summary
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.pipeline.steps[0].tool == "document-embed-upsert"
        assert cfg.pipeline.steps[0].collection == "doc_summary"

    def test_vector_collections_without_embedding_raises_for_doc_embed(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            vector_collections:
              - name: doc_summary
        """)
        with pytest.raises(Exception, match="embedding is required when vector_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_step_references_unknown_vector_collection_raises(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: doc_summary
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown vector collection"):
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
            vector_collections:
              - name: document_chunks
              - name: document_summary
            structured_collections:
              - name: contract_extraction
                schema: '{_SCHEMA}'
            pipeline:
              parallel: false
              steps:
                - tool: chunk-embed-upsert
                  collection: document_chunks
                - tool: extract-structured
                  collection: contract_extraction
                  extractor:
                    type: llm
                - tool: document-embed-upsert
                  collection: document_summary
                  prompt: "Summarize in one sentence."
                  max_tokens: 128
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.vector_collections) == 2
        assert len(cfg.structured_collections) == 1
        assert len(cfg.pipeline.steps) == 3
        tools = [s.tool for s in cfg.pipeline.steps]
        assert tools == ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
        doc_step = cfg.pipeline.steps[2]
        assert doc_step.prompt == "Summarize in one sentence."
        assert doc_step.max_tokens == 128


# ---------------------------------------------------------------------------
# ExtractorConfig
# ---------------------------------------------------------------------------

class TestExtractorConfig:
    def test_defaults(self):
        cfg = ExtractorConfig()
        assert cfg.type == "llm"
        assert cfg.prompt is None
        assert cfg.extract_as_list is False
        assert cfg.list_field == "items"
        assert cfg.item_id_field == "item_id"

    def test_custom_item_id_field(self):
        cfg = ExtractorConfig(item_id_field="clause_id")
        assert cfg.item_id_field == "clause_id"

    def test_extract_as_list_true(self):
        cfg = ExtractorConfig(extract_as_list=True, list_field="clauses", item_id_field="clause_id")
        assert cfg.extract_as_list is True
        assert cfg.list_field == "clauses"
        assert cfg.item_id_field == "clause_id"

    def test_yaml_list_extractor_parses(self):
        _SCHEMA = '{"type":"object","properties":{"text":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: clauses-app
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: contract_clauses
                schema: '{_SCHEMA}'
            pipeline:
              steps:
                - tool: extract-structured
                  collection: contract_clauses
                  extractor:
                    type: llm
                    extract_as_list: true
                    list_field: clauses
                    item_id_field: clause_id
                    prompt: contract_clauses_prompt.txt
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        ext = cfg.pipeline.steps[0].extractor
        assert ext.extract_as_list is True
        assert ext.list_field == "clauses"
        assert ext.item_id_field == "clause_id"
        assert ext.prompt == "contract_clauses_prompt.txt"

    def test_yaml_extractor_defaults_when_omitted(self):
        _SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: simple-app
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: records
                schema: '{_SCHEMA}'
            pipeline:
              steps:
                - tool: extract-structured
                  collection: records
                  extractor:
                    type: llm
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        ext = cfg.pipeline.steps[0].extractor
        assert ext.extract_as_list is False
        assert ext.list_field == "items"
        assert ext.item_id_field == "item_id"


# ---------------------------------------------------------------------------
# DocumentCollectionConfig
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WhenCondition / metadata-based step routing
# ---------------------------------------------------------------------------

class TestWhenCondition:
    def test_step_without_when_is_none(self):
        yaml_text = textwrap.dedent("""\
            name: app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: chunks
            pipeline:
              steps:
                - tool: chunk-embed-upsert
                  collection: chunks
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.pipeline.steps[0].when is None

    def test_step_with_when_metadata_parses(self):
        yaml_text = textwrap.dedent("""\
            name: app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: rule_chunks
            pipeline:
              steps:
                - tool: chunk-embed-upsert
                  collection: rule_chunks
                  when:
                    metadata:
                      doc_type: rules
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        step = cfg.pipeline.steps[0]
        assert step.when is not None
        assert step.when.metadata == {"doc_type": "rules"}

    def test_routed_ingestion_config_parses(self):
        """Full contract-compliance routing config from the README."""
        yaml_text = textwrap.dedent("""\
            name: contract-compliance
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: rule_chunks
              - name: contract_chunks
            pipeline:
              steps:
                - tool: chunk-embed-upsert
                  collection: rule_chunks
                  when:
                    metadata:
                      doc_type: rules
                - tool: chunk-embed-upsert
                  collection: contract_chunks
                  when:
                    metadata:
                      doc_type: contract
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.pipeline.steps) == 2

        rule_step = cfg.pipeline.steps[0]
        assert rule_step.collection == "rule_chunks"
        assert rule_step.when.metadata == {"doc_type": "rules"}

        contract_step = cfg.pipeline.steps[1]
        assert contract_step.collection == "contract_chunks"
        assert contract_step.when.metadata == {"doc_type": "contract"}

    def test_when_metadata_empty_by_default(self):
        yaml_text = textwrap.dedent("""\
            name: app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: chunks
            pipeline:
              steps:
                - tool: chunk-embed-upsert
                  collection: chunks
                  when: {}
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.pipeline.steps[0].when.metadata == {}


# ---------------------------------------------------------------------------
# VectorCollectionConfig
# ---------------------------------------------------------------------------

class TestVectorCollectionConfig:
    def test_defaults(self):
        cfg = VectorCollectionConfig(name="s")
        assert cfg.name == "s"
        assert cfg.dimensions == 1536

    def test_custom_description(self):
        cfg = VectorCollectionConfig(name="chunks", description="Passage chunks for search.")
        assert cfg.description == "Passage chunks for search."

    def test_step_prompt_and_max_tokens_on_step_config(self):
        cfg = PipelineStepConfig(tool="document-embed-upsert", collection="doc_summary", prompt="One sentence.", max_tokens=64)
        assert cfg.prompt == "One sentence."
        assert cfg.max_tokens == 64

    def test_vector_collection_metadata_fields(self):
        cfg = VectorCollectionConfig(name="meetings", metadata_fields=["customer_id", "deal_stage"])
        assert cfg.metadata_fields == ["customer_id", "deal_stage"]
