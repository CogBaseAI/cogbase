"""Unit tests for cogbase/config/config.py."""

from __future__ import annotations

import textwrap

import pytest

from cogbase.config.config import (
    AppConfig,
    ChunkerConfig,
    ChunkEmbedUpsertStepConfig,
    DocumentEmbedUpsertStepConfig,
    VectorCollectionConfig,
    EmbeddingConfig,
    ExtractorConfig,
    ExtractStructuredStepConfig,
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
    pipelines:
      - name: main
        routing_description: Full-text documents for chunked vector indexing.
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
        step = cfg.pipelines[0].steps[0]
        assert isinstance(step, ChunkEmbedUpsertStepConfig)
        assert step.chunker.chunk_size == 256

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
            pipelines:
              - name: main
                routing_description: Documents to embed as per-document summaries.
                steps:
                  - tool: document-embed-upsert
                    collection: doc_summary
                    doc_prompt: "Summarize in one sentence."
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        step = cfg.pipelines[0].steps[0]
        assert isinstance(step, DocumentEmbedUpsertStepConfig)
        assert step.tool == "document-embed-upsert"
        assert step.collection == "doc_summary"
        assert step.doc_prompt == "Summarize in one sentence."

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
            pipelines:
              - name: main
                routing_description: Documents to embed.
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
            pipelines:
              - name: main
                routing_description: Contract documents for full three-step ingestion.
                steps:
                  - tool: chunk-embed-upsert
                    collection: document_chunks
                  - tool: extract-structured
                    collection: contract_extraction
                    extractor:
                      type: llm
                      extraction_schema: '{_EXTRACTION_SCHEMA}'
                      prompt: ""
                  - tool: document-embed-upsert
                    collection: document_summary
                    doc_prompt: "Summarize in one sentence."
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.vector_collections) == 2
        assert len(cfg.structured_collections) == 1
        assert len(cfg.pipelines[0].steps) == 3
        tools = [s.tool for s in cfg.pipelines[0].steps]
        assert tools == ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
        doc_step = cfg.pipelines[0].steps[2]
        assert isinstance(cfg.pipelines[0].steps[0], ChunkEmbedUpsertStepConfig)
        assert isinstance(cfg.pipelines[0].steps[1], ExtractStructuredStepConfig)
        assert isinstance(cfg.pipelines[0].steps[2], DocumentEmbedUpsertStepConfig)
        assert doc_step.doc_prompt == "Summarize in one sentence."

    def test_config_format_prompt_returns_string(self):
        result = AppConfig.config_format_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_config_format_prompt_contains_all_pipeline_tools(self):
        result = AppConfig.config_format_prompt()
        tools = ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
        for tool in tools:
            assert tool in result, f"tool {tool!r} missing from config_format_prompt output"

    def test_config_format_prompt_contains_all_workflow_tools(self):
        # The workflow step union nests an Annotated discriminated union inside another
        # union — the renderer must recurse into nested unions so every leaf step type
        # (and its field descriptions) is documented for the LLM.
        result = AppConfig.config_format_prompt()
        tools = ["structured-query", "vector-search", "llm-structured", "structured-save"]
        for tool in tools:
            assert tool in result, f"workflow tool {tool!r} missing from config_format_prompt output"
        assert "primary_fields" in result, "structured-save primary_fields description missing"

    def test_config_format_prompt_contains_top_level_sections(self):
        result = AppConfig.config_format_prompt()
        for section in ("name:", "vector_collections:", "structured_collections:", "pipelines:"):
            assert section in result, f"{section!r} missing from config_format_prompt output"

    def test_config_format_prompt_tool_order_matches_literal(self):
        result = AppConfig.config_format_prompt()
        tools = ["chunk-embed-upsert", "extract-structured", "document-embed-upsert"]
        positions = [result.index(t) for t in tools]
        assert positions == sorted(positions), "tools appear out of Literal order in prompt"

    def test_config_format_prompt_uses_field_descriptions(self):
        app_prompt = AppConfig.config_format_prompt()
        assert "Application name" in app_prompt
        assert "Configured ingestion pipelines." in app_prompt
        assert "LLM configuration." not in app_prompt

        llm_prompt = LLMConfig.config_format_prompt()
        assert "LLM provider." in llm_prompt
        assert "Explicit API key." in llm_prompt

    def test_config_format_prompt_respects_prompt_skip_flag(self):
        from pydantic import BaseModel, Field
        from cogbase.config.prompt import ConfigPromptMixin

        class _SkipExample(ConfigPromptMixin, BaseModel):
            visible: str = Field(description="Visible field.")
            hidden: str = Field(
                default="secret",
                description="Hidden field.",
                json_schema_extra={"prompt_skip": True},
            )

        prompt = _SkipExample.config_format_prompt()
        assert "Visible field." in prompt
        assert "Hidden field." not in prompt

        collection_prompt = VectorCollectionConfig.config_format_prompt()
        assert "Collection description, shown to the LLM as context for a query." in collection_prompt
        assert "Metadata keys copied onto each stored vector." not in collection_prompt

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
# StructuredSaveStepConfig — primary_fields propagation
# ---------------------------------------------------------------------------


_SAVE_PROPAGATION_YAML = """\
name: save-prop
llm:
  model: gpt-4o-mini
structured_collections:
  - name: findings
    description: Workflow output collection populated by structured-save.
    schema: '{schema}'
workflows:
  - name: produce-findings
    params_from_collection:
      collection: findings
      filters:
        doc_id: "{{{{ doc.doc_id }}}}"
      params:
        doc_id: "{{{{ record.doc_id }}}}"
    steps:
      - id: judge
        tool: llm-structured
        prompt: Judge it.
        output_schema: '{schema}'
      - id: save
        tool: structured-save
        collection: findings
        primary_fields: [doc_id, finding_id]
        records:
          - "{{{{ steps.judge.output }}}}"
"""


class TestStructuredSavePrimaryFieldsPropagation:
    _SCHEMA = '{"type":"object","properties":{"doc_id":{"type":"string"},"finding_id":{"type":"string"}}}'

    def test_save_step_primary_fields_propagate_to_target_collection(self):
        yaml_text = _SAVE_PROPAGATION_YAML.format(schema=self._SCHEMA)
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["doc_id", "finding_id"]

    def test_save_step_primary_fields_overrides_collection_value(self):
        # Collection has stale primary_fields; save step's value wins.
        yaml_text = _SAVE_PROPAGATION_YAML.format(schema=self._SCHEMA).replace(
            "schema: '" + self._SCHEMA + "'",
            "schema: '" + self._SCHEMA + "'\n    primary_fields: [stale]",
            1,
        )
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["doc_id", "finding_id"]

    def test_empty_save_step_primary_fields_leaves_collection_value(self):
        # When the save step omits primary_fields, the collection's value is preserved.
        yaml_text = _SAVE_PROPAGATION_YAML.format(schema=self._SCHEMA)
        yaml_text = yaml_text.replace(
            "        primary_fields: [doc_id, finding_id]\n", ""
        ).replace(
            "    schema: '" + self._SCHEMA + "'",
            "    schema: '" + self._SCHEMA + "'\n    primary_fields: [doc_id]",
        )
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["doc_id"]

    def test_nested_foreach_save_step_propagates(self):
        # The save step lives inside a foreach block — propagation must recurse.
        yaml_text = textwrap.dedent(f"""\
            name: nested-save
            llm:
              model: gpt-4o-mini
            structured_collections:
              - name: items
                description: Source items.
                schema: '{self._SCHEMA}'
                primary_fields: [doc_id]
              - name: findings
                description: Workflow output collection.
                schema: '{self._SCHEMA}'
            workflows:
              - name: review-each
                params_from_collection:
                  collection: items
                  filters:
                    doc_id: "{{{{ doc.doc_id }}}}"
                  params:
                    doc_id: "{{{{ record.doc_id }}}}"
                steps:
                  - id: load
                    tool: structured-query
                    collection: items
                    filters:
                      doc_id: "{{{{ input.doc_id }}}}"
                  - id: each
                    foreach: "{{{{ steps.load.records }}}}"
                    steps:
                      - id: save
                        tool: structured-save
                        collection: findings
                        primary_fields: [finding_id]
                        records:
                          - "{{{{ item }}}}"
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["finding_id"]


# ---------------------------------------------------------------------------
# ExtractorConfig
# ---------------------------------------------------------------------------

class TestExtractorConfig:
    _EXTRACTION_SCHEMA = '{"type":"object","properties":{"value":{"type":"string"}}}'

    def test_required_fields(self):
        cfg = ExtractorConfig(extraction_schema=self._EXTRACTION_SCHEMA, prompt="")
        assert cfg.type == "llm"
        assert cfg.extraction_schema == self._EXTRACTION_SCHEMA
        assert cfg.prompt == ""
        assert cfg.record_mode == "one"
        assert cfg.response_field == "items"
        assert cfg.id_field is None
        assert cfg.id_template is None

    def test_custom_id_field(self):
        cfg = ExtractorConfig(extraction_schema=self._EXTRACTION_SCHEMA, prompt="", id_field="clause_id")
        assert cfg.id_field == "clause_id"

    def test_record_mode_many(self):
        cfg = ExtractorConfig(
            extraction_schema=self._EXTRACTION_SCHEMA,
            prompt="",
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
            pipelines:
              - name: main
                routing_description: Contract documents to extract individual clauses from.
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
        ext = cfg.pipelines[0].steps[0].extractor
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
            pipelines:
              - name: main
                routing_description: Generic documents to extract structured records from.
                steps:
                  - tool: extract-structured
                    collection: records
                    extractor:
                      type: llm
                      extraction_schema: '{_EXTRACTION_SCHEMA}'
                      prompt: ""
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        ext = cfg.pipelines[0].steps[0].extractor
        assert ext.extraction_schema == _EXTRACTION_SCHEMA
        assert ext.record_mode == "one"
        assert ext.response_field == "items"
        assert ext.id_field is None


# ---------------------------------------------------------------------------
# DocumentCollectionConfig
# ---------------------------------------------------------------------------


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
        cfg = DocumentEmbedUpsertStepConfig(collection="doc_summary", doc_prompt="One sentence.")
        assert cfg.doc_prompt == "One sentence."

    def test_chunk_step_prompt_skips_fixed_tool(self):
        prompt = ChunkEmbedUpsertStepConfig.config_format_prompt()
        assert "tool: chunk-embed-upsert  # Pipeline tool to run." in prompt
        assert "default: chunk-embed-upsert" not in prompt
        assert "chunker:" not in prompt

    def test_vector_collection_metadata_fields(self):
        cfg = VectorCollectionConfig(
            name="meetings",
            description="Meeting notes and extracted records for search.",
            metadata_fields=["customer_id", "deal_stage"],
        )
        assert cfg.metadata_fields == ["customer_id", "deal_stage"]
