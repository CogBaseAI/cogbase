"""Unit tests for cogbase/config/config.py."""

from __future__ import annotations

import json
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
    def test_default_provider_is_openai(self):
        cfg = LLMConfig(model="gpt-4o-mini", api_key="sk-test")
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
        cfg = EmbeddingConfig(api_key="EMPTY")
        assert cfg.provider == "openai"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimensions == 1536

    def test_custom_dimensions(self):
        cfg = EmbeddingConfig(api_key="EMPTY", dimensions=512)
        assert cfg.dimensions == 512


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent("""\
    name: test-app
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-test
""")

_FULL_YAML = textwrap.dedent("""\
    name: full-app
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-test
    embedding:
      provider: openai
      model: text-embedding-3-small
      api_key: sk-test
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
              api_key: sk-test
            structured_store:
              type: sqlite
              path: ./data/my.db
            vector_store:
              type: faiss
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
              api_key: sk-test
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
              api_key: sk-test
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
              api_key: sk-test
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
              api_key: sk-test
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
              api_key: sk-test
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: sk-test
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
        assert "api_key:" in llm_prompt

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
              api_key: sk-test
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
  api_key: sk-test
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
              api_key: sk-test
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
# StructuredSaveStepConfig — primary_fields must be in upstream
# LLMStructuredStepConfig.output_schema when records sources the LLM output.
# ---------------------------------------------------------------------------


class TestStructuredSavePrimaryFieldsAgainstUpstreamSchema:
    _COLLECTION_SCHEMA = (
        '{"type":"object","properties":'
        '{"doc_id":{"type":"string"},"clause_id":{"type":"string"},'
        '"status":{"type":"string"}}}'
    )
    _OUTPUT_SCHEMA_FULL = (
        '{"type":"object","properties":'
        '{"doc_id":{"type":"string"},"clause_id":{"type":"string"},'
        '"status":{"type":"string"}}}'
    )
    _OUTPUT_SCHEMA_MISSING_CLAUSE_ID = (
        '{"type":"object","properties":'
        '{"doc_id":{"type":"string"},"status":{"type":"string"}}}'
    )
    _OUTPUT_SCHEMA_ONLY_STATUS = (
        '{"type":"object","properties":{"status":{"type":"string"}}}'
    )

    @staticmethod
    def _workflow_yaml(output_schema: str, primary_fields: str = "[doc_id, clause_id]") -> str:
        # Mirrors the contract_compliance demo: a judge llm-structured step
        # feeds its output into a save_finding structured-save step.
        return textwrap.dedent(f"""\
            name: save-validation
            llm:
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: findings
                description: Workflow output collection.
                schema: '{TestStructuredSavePrimaryFieldsAgainstUpstreamSchema._COLLECTION_SCHEMA}'
            workflows:
              - name: judge-and-save
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
                    input:
                      clause: "{{{{ item }}}}"
                    output_schema: '{output_schema}'
                  - id: save_finding
                    tool: structured-save
                    collection: findings
                    primary_fields: {primary_fields}
                    records:
                      - "{{{{ steps.judge.output }}}}"
        """)

    def test_primary_fields_subset_of_output_schema_validates(self):
        yaml_text = self._workflow_yaml(self._OUTPUT_SCHEMA_FULL)
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["doc_id", "clause_id"]

    def test_primary_field_missing_from_output_schema_raises(self):
        yaml_text = self._workflow_yaml(self._OUTPUT_SCHEMA_MISSING_CLAUSE_ID)
        with pytest.raises(Exception, match=r"primary_fields \['clause_id'\] missing"):
            AppConfig.from_yaml(yaml_text)

    def test_error_names_workflow_save_step_and_upstream_step(self):
        yaml_text = self._workflow_yaml(self._OUTPUT_SCHEMA_MISSING_CLAUSE_ID)
        with pytest.raises(Exception) as exc_info:
            AppConfig.from_yaml(yaml_text)
        msg = str(exc_info.value)
        assert "'judge-and-save'" in msg
        assert "'save_finding'" in msg
        assert "'judge'" in msg

    def test_multiple_missing_primary_fields_all_listed(self):
        yaml_text = self._workflow_yaml(self._OUTPUT_SCHEMA_ONLY_STATUS)
        with pytest.raises(Exception) as exc_info:
            AppConfig.from_yaml(yaml_text)
        msg = str(exc_info.value)
        assert "'doc_id'" in msg
        assert "'clause_id'" in msg

    def test_records_not_referencing_llm_structured_step_skipped(self):
        # records sources from the foreach item rather than an llm-structured step output.
        # The constraint does not apply — primary_fields can name fields the validator
        # cannot see, and validation must not block this case.
        yaml_text = textwrap.dedent(f"""\
            name: passthrough-save
            llm:
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: items
                description: Source items.
                schema: '{self._COLLECTION_SCHEMA}'
                primary_fields: [doc_id]
              - name: findings
                description: Workflow output collection.
                schema: '{self._COLLECTION_SCHEMA}'
            workflows:
              - name: passthrough
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
                        primary_fields: [doc_id, clause_id]
                        records:
                          - "{{{{ item }}}}"
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        assert sc.primary_fields == ["doc_id", "clause_id"]

    def test_nested_foreach_validates_against_upstream_schema(self):
        # Mirrors the contract_compliance demo: judge + save_finding both inside foreach.
        yaml_text = textwrap.dedent(f"""\
            name: nested-validation
            llm:
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: clauses
                description: Source clauses.
                schema: '{self._COLLECTION_SCHEMA}'
                primary_fields: [doc_id, clause_id]
              - name: findings
                description: Workflow output collection.
                schema: '{self._COLLECTION_SCHEMA}'
            workflows:
              - name: review-each-clause
                params_from_collection:
                  collection: clauses
                  filters:
                    doc_id: "{{{{ doc.doc_id }}}}"
                  params:
                    doc_id: "{{{{ record.doc_id }}}}"
                steps:
                  - id: load
                    tool: structured-query
                    collection: clauses
                    filters:
                      doc_id: "{{{{ input.doc_id }}}}"
                  - id: each
                    foreach: "{{{{ steps.load.records }}}}"
                    steps:
                      - id: judge
                        tool: llm-structured
                        prompt: Judge the clause.
                        input:
                          clause: "{{{{ item }}}}"
                        output_schema: '{self._OUTPUT_SCHEMA_MISSING_CLAUSE_ID}'
                      - id: save_finding
                        tool: structured-save
                        collection: findings
                        primary_fields: [doc_id, clause_id]
                        records:
                          - "{{{{ steps.judge.output }}}}"
        """)
        with pytest.raises(Exception, match=r"primary_fields \['clause_id'\] missing"):
            AppConfig.from_yaml(yaml_text)

    def test_invalid_output_schema_json_raises_clear_error(self):
        # Bad JSON in upstream output_schema → clear validation error, not a silent skip.
        yaml_text = self._workflow_yaml("not-valid-json")
        with pytest.raises(Exception, match="output_schema is not valid JSON"):
            AppConfig.from_yaml(yaml_text)

    def test_empty_save_primary_fields_skips_check(self):
        # No primary_fields → nothing to validate, no error even if output_schema is empty.
        yaml_text = self._workflow_yaml(
            '{"type":"object","properties":{}}',
            primary_fields="[]",
        )
        cfg = AppConfig.from_yaml(yaml_text)
        sc = next(s for s in cfg.structured_collections if s.name == "findings")
        # Empty save primary_fields means the propagation pass leaves the collection
        # value untouched — here it was unset on the collection, so it stays empty.
        assert sc.primary_fields == []


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

    def test_one_with_id_field_raises(self):
        with pytest.raises(Exception, match="id_field must not be set for record_mode=one"):
            ExtractorConfig(extraction_schema=self._EXTRACTION_SCHEMA, prompt="", id_field="clause_id")

    def test_many_missing_response_field_raises(self):
        with pytest.raises(Exception, match="record_mode=many requires"):
            ExtractorConfig(
                extraction_schema=self._EXTRACTION_SCHEMA,
                prompt="",
                record_mode="many",
                response_field=None,
                id_field="clause_id",
                id_template="{doc_id}__{index:04d}",
            )

    def test_many_missing_id_field_raises(self):
        with pytest.raises(Exception, match="record_mode=many requires"):
            ExtractorConfig(
                extraction_schema=self._EXTRACTION_SCHEMA,
                prompt="",
                record_mode="many",
                response_field="items",
                id_template="{doc_id}__{index:04d}",
            )

    def test_many_missing_id_template_raises(self):
        with pytest.raises(Exception, match="record_mode=many requires"):
            ExtractorConfig(
                extraction_schema=self._EXTRACTION_SCHEMA,
                prompt="",
                record_mode="many",
                response_field="items",
                id_field="clause_id",
            )

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

    def test_many_mode_strips_id_field_from_extraction_schema(self):
        schema_with_id = json.dumps({
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "description": "per-record id"},
                "text": {"type": "string"},
            },
            "required": ["clause_id", "text"],
        })
        cfg = ExtractorConfig(
            extraction_schema=schema_with_id,
            prompt="",
            record_mode="many",
            response_field="clauses",
            id_field="clause_id",
            id_template="{doc_id}__{index:04d}",
        )
        parsed = json.loads(cfg.extraction_schema)
        assert "clause_id" not in parsed["properties"]
        assert "clause_id" not in parsed.get("required", [])
        assert "text" in parsed["properties"]

    def test_many_mode_strips_id_field_not_in_required(self):
        schema_with_id = json.dumps({
            "type": "object",
            "properties": {
                "clause_id": {"type": "string"},
                "text": {"type": "string"},
            },
        })
        cfg = ExtractorConfig(
            extraction_schema=schema_with_id,
            prompt="",
            record_mode="many",
            response_field="clauses",
            id_field="clause_id",
            id_template="{doc_id}__{index:04d}",
        )
        parsed = json.loads(cfg.extraction_schema)
        assert "clause_id" not in parsed["properties"]
        assert "clause_id" not in parsed.get("required", [])

    def test_yaml_list_extractor_parses(self):
        _EXTRACTION_SCHEMA = '{"type":"object","properties":{"text":{"type":"string"}}}'
        _RECORD_SCHEMA = '{"type":"object","properties":{"text":{"type":"string"},"clause_id":{"type":"string"},"doc_id":{"type":"string"}}}'
        yaml_text = textwrap.dedent(f"""\
            name: clauses-app
            llm:
              model: gpt-4o-mini
              api_key: sk-test
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
              api_key: sk-test
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


# ---------------------------------------------------------------------------
# AppConfig._validate — workflow collection reference checks
# ---------------------------------------------------------------------------

_SCHEMA = '{"type":"object","properties":{"doc_id":{"type":"string"}}}'


def _ref_yaml(*, steps_yaml: str, pfc_collection: str = "records") -> str:
    """Full AppConfig YAML with one workflow. Callers supply the workflow steps block."""
    return textwrap.dedent(f"""\
        name: ref-test
        llm:
          model: gpt-4o-mini
          api_key: sk-test
        vector_collections:
          - name: chunks
            description: Passage chunks.
        structured_collections:
          - name: records
            description: Extracted records.
            schema: '{_SCHEMA}'
            primary_fields: [doc_id]
        workflows:
          - name: wf
            params_from_collection:
              collection: {pfc_collection}
              filters:
                doc_id: "{{{{ doc.doc_id }}}}"
              params:
                doc_id: "{{{{ record.doc_id }}}}"
            steps:
    """) + steps_yaml


class TestWorkflowCollectionReferenceValidation:
    # Steps YAML must be pre-indented to align with `steps:` (6-space list items).
    _LOAD_RECORDS = "      - id: load\n        tool: structured-query\n        collection: records\n"

    def test_valid_structured_query_step_passes(self):
        cfg = AppConfig.from_yaml(_ref_yaml(steps_yaml=self._LOAD_RECORDS))
        assert cfg.workflows[0].name == "wf"

    def test_unknown_structured_query_collection_raises(self):
        steps = "      - id: load\n        tool: structured-query\n        collection: does_not_exist\n"
        with pytest.raises(Exception, match=r"unknown structured collection.*'does_not_exist'"):
            AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))

    def test_valid_vector_search_step_passes(self):
        steps = '      - id: s\n        tool: vector-search\n        collection: chunks\n        query: "q"\n'
        cfg = AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))
        assert cfg.workflows[0].name == "wf"

    def test_unknown_vector_search_collection_raises(self):
        steps = '      - id: s\n        tool: vector-search\n        collection: missing_vectors\n        query: "q"\n'
        with pytest.raises(Exception, match=r"unknown vector collection.*'missing_vectors'"):
            AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))

    def test_valid_structured_save_step_passes(self):
        steps = (
            f"      - id: judge\n        tool: llm-structured\n        prompt: p\n"
            f"        output_schema: '{_SCHEMA}'\n"
            f"      - id: save\n        tool: structured-save\n        collection: records\n"
            f"        primary_fields: [doc_id]\n"
            f"        records:\n          - \"{{{{ steps.judge.output }}}}\"\n"
        )
        cfg = AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))
        assert cfg.workflows[0].name == "wf"

    def test_unknown_structured_save_collection_raises(self):
        steps = (
            f"      - id: judge\n        tool: llm-structured\n        prompt: p\n"
            f"        output_schema: '{_SCHEMA}'\n"
            f"      - id: save\n        tool: structured-save\n        collection: ghost_collection\n"
            f"        primary_fields: [doc_id]\n"
            f"        records:\n          - \"{{{{ steps.judge.output }}}}\"\n"
        )
        with pytest.raises(Exception, match=r"unknown structured collection.*'ghost_collection'"):
            AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))

    def test_unknown_params_from_collection_raises(self):
        with pytest.raises(Exception, match=r"params_from_collection.*unknown.*'no_such_collection'"):
            AppConfig.from_yaml(_ref_yaml(
                pfc_collection="no_such_collection",
                steps_yaml=self._LOAD_RECORDS,
            ))

    def test_nested_foreach_collection_reference_validated(self):
        steps = (
            "      - id: load\n        tool: structured-query\n        collection: records\n"
            "      - id: each\n        foreach: \"{{ steps.load.records }}\"\n        steps:\n"
            "          - id: search\n            tool: vector-search\n"
            "            collection: no_such_vec\n            query: \"{{ item }}\"\n"
        )
        with pytest.raises(Exception, match=r"unknown vector collection.*'no_such_vec'"):
            AppConfig.from_yaml(_ref_yaml(steps_yaml=steps))


# ---------------------------------------------------------------------------
# AppConfig._validate — multi-extractor conflict detection
# ---------------------------------------------------------------------------


_SCHEMA_A = '{"type":"object","properties":{"a":{"type":"string"}}}'
_SCHEMA_B = '{"type":"object","properties":{"b":{"type":"string"}}}'


def _multi_extractor_yaml(*, schema2: str, prompt2: str) -> str:
    """Two pipelines each with an extract-structured step targeting the same collection."""
    return textwrap.dedent(f"""\
        name: conflict-test
        llm:
          model: gpt-4o-mini
          api_key: sk-test
        structured_collections:
          - name: shared
            description: Shared structured collection.
            schema: '{_SCHEMA_A}'
            primary_fields: [a]
        pipelines:
          - name: pipeline-a
            routing_description: Doc type A.
            steps:
              - tool: extract-structured
                collection: shared
                extractor:
                  extraction_schema: '{_SCHEMA_A}'
                  prompt: "extract A"
          - name: pipeline-b
            routing_description: Doc type B.
            steps:
              - tool: extract-structured
                collection: shared
                extractor:
                  extraction_schema: '{schema2}'
                  prompt: "{prompt2}"
    """)


class TestMultiExtractorConflictValidation:
    def test_two_pipelines_identical_extractor_passes(self):
        # Same schema and prompt — no conflict.
        yaml_text = _multi_extractor_yaml(schema2=_SCHEMA_A, prompt2="extract A")
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.pipelines) == 2

    def test_different_schema_raises(self):
        yaml_text = _multi_extractor_yaml(schema2=_SCHEMA_B, prompt2="extract A")
        with pytest.raises(Exception, match=r"'shared'.*different schemas or prompts"):
            AppConfig.from_yaml(yaml_text)

    def test_different_prompt_raises(self):
        yaml_text = _multi_extractor_yaml(schema2=_SCHEMA_A, prompt2="extract B differently")
        with pytest.raises(Exception, match=r"'shared'.*different schemas or prompts"):
            AppConfig.from_yaml(yaml_text)

    def test_error_names_both_conflicting_pipelines(self):
        yaml_text = _multi_extractor_yaml(schema2=_SCHEMA_B, prompt2="extract B")
        with pytest.raises(Exception) as exc_info:
            AppConfig.from_yaml(yaml_text)
        msg = str(exc_info.value)
        assert "'pipeline-a'" in msg
        assert "'pipeline-b'" in msg

    def test_single_pipeline_no_conflict(self):
        yaml_text = textwrap.dedent(f"""\
            name: single-pipe
            llm:
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: records
                description: Records.
                schema: '{_SCHEMA_A}'
                primary_fields: [a]
            pipelines:
              - name: only
                routing_description: All docs.
                steps:
                  - tool: extract-structured
                    collection: records
                    extractor:
                      extraction_schema: '{_SCHEMA_A}'
                      prompt: "extract"
        """)
        cfg = AppConfig.from_yaml(yaml_text)
        assert len(cfg.pipelines) == 1
