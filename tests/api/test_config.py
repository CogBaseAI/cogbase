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
        description: Full-text document chunks for detailed retrieval.
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
                description: Full-text document chunks for detailed retrieval.
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
                description: Full-text document chunks for detailed retrieval.
        """)
        with pytest.raises(Exception, match="embedding is required when vector_collections"):
            AppConfig.from_yaml(yaml_text)

    def test_vector_collection_description_is_required(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: doc_chunks
        """)
        with pytest.raises(Exception, match="description"):
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
                description: One summary vector per document for topic-level search.
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: doc_summary
                  doc_prompt: "Summarize in one sentence."
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.pipeline.steps[0].tool == "document-embed-upsert"
        assert cfg.pipeline.steps[0].collection == "doc_summary"
        assert cfg.pipeline.steps[0].doc_prompt == "Summarize in one sentence."

    def test_vector_collections_without_embedding_raises_for_doc_embed(self):
        yaml_text = textwrap.dedent("""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            vector_collections:
              - name: doc_summary
                description: One summary vector per document for topic-level search.
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
                description: One summary vector per document for topic-level search.
            pipeline:
              steps:
                - tool: document-embed-upsert
                  collection: nonexistent
        """)
        with pytest.raises(Exception, match="unknown vector collection"):
            AppConfig.from_yaml(yaml_text)

    def test_full_three_step_config_parses(self):
        _EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
        _RECORD_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"},"doc_id":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: contracts
            llm:
              model: gpt-4o-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_collections:
              - name: document_chunks
                description: Full-text document chunks for detailed retrieval.
              - name: document_summary
                description: One summary vector per document for topic-level search.
            structured_collections:
              - name: contract_extraction
                description: Extracted contract facts and entities for exact lookup.
                schema: '{_RECORD_SCHEMA}'
                primary_fields: [doc_id]
            pipeline:
              parallel: false
              steps:
                - tool: chunk-embed-upsert
                  collection: document_chunks
                - tool: extract-structured
                  collection: contract_extraction
                  extractor:
                    type: llm
                    extraction_schema: '{_EXTRACTION_SCHEMA}'
                - tool: document-embed-upsert
                  collection: document_summary
                  doc_prompt: "Summarize in one sentence."
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.vector_collections) == 2
        assert len(cfg.structured_collections) == 1
        assert len(cfg.pipeline.steps) == 3
        tools = [s.tool for s in cfg.pipeline.steps]
        assert tools == ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
        doc_step = cfg.pipeline.steps[2]
        assert doc_step.doc_prompt == "Summarize in one sentence."

    def test_structured_collection_description_is_required(self):
        _SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: bad-app
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: records
                schema: '{_SCHEMA}'
        """)
        with pytest.raises(Exception, match="description"):
            AppConfig.from_yaml(yaml_text)


# ---------------------------------------------------------------------------
# ExtractorConfig
# ---------------------------------------------------------------------------

class TestExtractorConfig:
    _EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'

    def test_required_fields(self):
        cfg = ExtractorConfig(extraction_schema=self._EXTRACTION_SCHEMA)
        assert cfg.type == "llm"
        assert cfg.extraction_schema == self._EXTRACTION_SCHEMA
        assert cfg.prompt is None
        assert cfg.record_mode == "one"
        assert cfg.response_field == "items"
        assert cfg.id_field is None
        assert cfg.id_template is None

    def test_custom_id_field(self):
        cfg = ExtractorConfig(extraction_schema=self._EXTRACTION_SCHEMA, id_field="clause_id")
        assert cfg.id_field == "clause_id"

    def test_record_mode_many(self):
        cfg = ExtractorConfig(
            extraction_schema=self._EXTRACTION_SCHEMA,
            record_mode="many",
            response_field="clauses",
            id_field="clause_id",
            id_template="{doc_id}__{index:04d}",
        )
        assert cfg.record_mode == "many"
        assert cfg.response_field == "clauses"
        assert cfg.id_field == "clause_id"
        assert cfg.id_template == "{doc_id}__{index:04d}"

    def test_yaml_list_extractor_parses(self):
        _EXTRACTION_SCHEMA = '{"type":"object","properties":{"text":{"type":"string"}}}'
        _RECORD_SCHEMA = '{"type":"object","properties":{"text":{"type":"string"},"clause_id":{"type":"string"},"doc_id":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: clauses-app
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: contract_clauses
                description: Extracted contract clauses with clause type and verbatim text.
                schema: '{_RECORD_SCHEMA}'
                primary_fields: [clause_id]
            pipeline:
              steps:
                - tool: extract-structured
                  collection: contract_clauses
                  extractor:
                    type: llm
                    extraction_schema: '{_EXTRACTION_SCHEMA}'
                    record_mode: many
                    response_field: clauses
                    id_field: clause_id
                    id_template: "{{doc_id}}__{{index:04d}}"
                    prompt: contract_clauses_prompt.txt
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        ext = cfg.pipeline.steps[0].extractor
        assert ext.extraction_schema == _EXTRACTION_SCHEMA
        assert ext.record_mode == "many"
        assert ext.response_field == "clauses"
        assert ext.id_field == "clause_id"
        assert ext.id_template == "{doc_id}__{index:04d}"
        assert ext.prompt == "contract_clauses_prompt.txt"

    def test_yaml_extractor_parses(self):
        _EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'
        _RECORD_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"},"doc_id":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: simple-app
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: records
                description: Generic structured records for exact lookup.
                schema: '{_RECORD_SCHEMA}'
                primary_fields: [doc_id]
            pipeline:
              steps:
                - tool: extract-structured
                  collection: records
                  extractor:
                    type: llm
                    extraction_schema: '{_EXTRACTION_SCHEMA}'
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        ext = cfg.pipeline.steps[0].extractor
        assert ext.extraction_schema == _EXTRACTION_SCHEMA
        assert ext.record_mode == "one"
        assert ext.response_field == "items"
        assert ext.id_field is None


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
                description: Full-text document chunks for detailed retrieval.
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
                description: Policy and rules chunks for compliance checks.
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
                description: Policy and rules chunks for compliance checks.
              - name: contract_chunks
                description: Contract text chunks for clause-level retrieval.
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
                description: Full-text document chunks for detailed retrieval.
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
        cfg = VectorCollectionConfig(name="s", description="test collection")
        assert cfg.name == "s"
        assert cfg.dimensions == 1536

    def test_custom_description(self):
        cfg = VectorCollectionConfig(name="chunks", description="Passage chunks for search.")
        assert cfg.description == "Passage chunks for search."

    def test_empty_description_raises(self):
        with pytest.raises(Exception, match="must be set"):
            VectorCollectionConfig(name="chunks", description=" ")

    def test_step_prompt_on_step_config(self):
        cfg = PipelineStepConfig(tool="document-embed-upsert", collection="doc_summary", doc_prompt="One sentence.")
        assert cfg.doc_prompt == "One sentence."

    def test_vector_collection_metadata_fields(self):
        cfg = VectorCollectionConfig(
            name="meetings",
            description="Meeting notes and extracted records for search.",
            metadata_fields=["customer_id", "deal_stage"],
        )
        assert cfg.metadata_fields == ["customer_id", "deal_stage"]
