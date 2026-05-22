"""Unit tests for cogbase/core/app_generator.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import yaml

from cogbase.core.app_generator import (
    _build_collection_to_pipeline_map,
    _extract_record_schemas,
    _inject_pipeline_record_schemas,
    _inject_workflow_output_schemas,
    _make_record_schema,
    _parse_and_validate_schemas,
    _run_propose_extraction_schemas,
    _run_propose_pipeline_config,
    _run_propose_workflow_config,
    _run_propose_workflow_schemas,
    _validate_extraction_schema,
    _validate_workflow_cross_pipeline_doc_id_filters,
    _validate_workflow_output_schema,
)
from cogbase.config.config import AppConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MINIMAL_SCHEMA_YAML = """\
contracts:
  type: object
  properties:
    vendor_name:
      type: string
      description: Vendor name
"""

_MINIMAL_CONFIG_YAML = """\
name: test-app
vector_collections:
  - name: chunks
    description: Semantic search chunks.
pipelines:
  - name: main
    routing_description: Documents for chunked vector indexing.
    steps:
      - tool: chunk-embed-upsert
        collection: chunks
"""

_CONFIG_YAML_WITH_STRUCTURED = """\
name: test-app
vector_collections:
  - name: chunks
    description: Semantic search chunks.
structured_collections:
  - name: contracts
    description: Extracted contract facts.
    primary_fields: [doc_id]
pipelines:
  - name: main
    routing_description: Contract documents for chunked indexing and structured extraction.
    steps:
      - tool: chunk-embed-upsert
        collection: chunks
      - tool: extract-structured
        collection: contracts
        extractor:
          type: llm
          extraction_schema: '{"type":"object","properties":{"vendor":{"type":"string","description":"Vendor name"}}}'
          prompt: Extract contract data.
"""

_WORKFLOW_SCHEMA_YAML = """\
clause_compliance_findings:
  type: object
  properties:
    doc_id:
      type: string
      description: Source document id
    clause_id:
      type: string
      description: Reviewed clause id
    status:
      type: string
      description: Compliance status
"""

# Full record schema for "contracts" collection (extraction fields + injected doc_id).
_CONTRACTS_RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "document identifier"},
        "vendor": {"type": "string", "description": "Vendor name"},
    },
    "required": ["doc_id"],
}

# Keyed record schemas as they come out of _extract_record_schemas.
_RECORD_SCHEMAS = {"contracts": json.dumps(_CONTRACTS_RECORD_SCHEMA)}

# Pipeline config dict as produced by _run_propose_pipeline_config (schema already injected).
_PIPELINE_CONFIG_DICT = {
    "name": "test-app",
    "vector_collections": [{"name": "chunks", "description": "Semantic search chunks."}],
    "structured_collections": [
        {
            "name": "contracts",
            "description": "Extracted contract facts.",
            "schema": json.dumps(_CONTRACTS_RECORD_SCHEMA),
            "primary_fields": ["doc_id"],
        }
    ],
    "pipelines": [
        {
            "name": "main",
            "routing_description": "Contract documents for chunked indexing and structured extraction.",
            "steps": [
                {"tool": "chunk-embed-upsert", "collection": "chunks"},
                {
                    "tool": "extract-structured",
                    "collection": "contracts",
                    "extractor": {
                        "type": "llm",
                        "extraction_schema": '{"type":"object","properties":{"vendor":{"type":"string","description":"Vendor name"}}}',
                        "prompt": "Extract contract data.",
                    },
                },
            ],
        }
    ],
}

# Workflow additions YAML: only the new structured_collections + workflows sections.
_WORKFLOW_ADDITIONS_YAML = """\
structured_collections:
  - name: clause_compliance_findings
    description: Clause-level compliance findings.
    primary_fields: [clause_id]
workflows:
  - name: check-compliance
    trigger:
      type: manual
    params_from_collection:
      collection: contracts
      filters:
        doc_id: "{{ doc.doc_id }}"
      params:
        doc_id: "{{ record.doc_id }}"
    steps:
      - id: judge
        tool: llm-structured
        prompt: Judge compliance.
        input:
          doc_id: "{{ input.doc_id }}"
        output_schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"Source document id"},"clause_id":{"type":"string","description":"Reviewed clause id"},"status":{"type":"string","description":"Compliance status"}}}'
      - id: save
        tool: structured-save
        collection: clause_compliance_findings
        records:
          - "{{ steps.judge.output }}"
"""

# Workflow output schema for clause_compliance_findings.
_FINDING_WORKFLOW_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "Source document id"},
        "clause_id": {"type": "string", "description": "Reviewed clause id"},
        "status": {"type": "string", "description": "Compliance status"},
    },
})

_CONFIG_YAML_WITH_WORKFLOW = """\
name: test-app
vector_collections:
  - name: chunks
    description: Semantic search chunks.
structured_collections:
  - name: contracts
    description: Extracted contract facts.
    primary_fields: [doc_id]
  - name: clause_compliance_findings
    description: Clause-level compliance findings.
    primary_fields: [clause_id]
pipelines:
  - name: main
    routing_description: Contract documents for chunked indexing and structured extraction.
    steps:
      - tool: chunk-embed-upsert
        collection: chunks
      - tool: extract-structured
        collection: contracts
        extractor:
          type: llm
          extraction_schema: '{"type":"object","properties":{"vendor":{"type":"string","description":"Vendor name"}}}'
          prompt: Extract contract data.
workflows:
  - name: check-compliance
    trigger:
      type: manual
    params_from_collection:
      collection: contracts
      filters:
        doc_id: "{{ doc.doc_id }}"
      params:
        doc_id: "{{ record.doc_id }}"
    steps:
      - id: judge
        tool: llm-structured
        prompt: Judge compliance.
        input:
          doc_id: "{{ input.doc_id }}"
        output_schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"Source document id"},"clause_id":{"type":"string","description":"Reviewed clause id"},"status":{"type":"string","description":"Compliance status"}}}'
      - id: save
        tool: structured-save
        collection: clause_compliance_findings
        records:
          - "{{ steps.judge.output }}"
"""


async def _text_stream(text: str):
    """Async generator that yields a single text chunk, simulating complete_stream."""
    yield text


def _make_llm(*responses: str) -> MagicMock:
    """Return a mock LLMBase whose complete() and complete_stream() yield each string in order."""
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[{"content": r, "tool_calls": None} for r in responses]
    )
    llm.complete_stream = MagicMock(
        side_effect=[_text_stream(r) for r in responses]
    )
    return llm


_CONVERSATION = [
    {"role": "user", "content": "Build a contract analysis app"},
    {"role": "assistant", "content": "Got it, let me propose schemas."},
]


class TestMakeRecordSchema:
    def test_injects_doc_id_property(self):
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = _make_record_schema(schema)
        assert "doc_id" in result["properties"]
        assert result["properties"]["doc_id"] == {
            "type": "string",
            "description": "document identifier",
        }

    def test_doc_id_first_in_required(self):
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = _make_record_schema(schema)
        assert result["required"][0] == "doc_id"
        assert "name" in result["required"]

    def test_doc_id_not_duplicated_in_required_when_absent(self):
        schema = {"properties": {"amount": {"type": "number"}}, "required": ["amount"]}
        result = _make_record_schema(schema)
        assert result["required"].count("doc_id") == 1

    def test_no_properties_key_gets_one_added(self):
        schema = {}
        result = _make_record_schema(schema)
        assert "doc_id" in result["properties"]
        assert result["required"] == ["doc_id"]

    def test_realistic_schema_with_type_and_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "contract_type": {"type": "string"},
                "parties": {"type": "array", "items": {"type": "string"}},
                "effective_date": {"type": "string"},
            },
            "required": ["contract_type", "parties", "effective_date"],
            "additionalProperties": False,
        }
        result = _make_record_schema(schema)

        assert result["type"] == "object"
        assert result["additionalProperties"] is False

        assert result["properties"]["doc_id"] == {
            "type": "string",
            "description": "document identifier",
        }
        assert result["properties"]["parties"] == {"type": "array", "items": {"type": "string"}}

        assert result["required"][0] == "doc_id"
        assert set(result["required"]) == {"doc_id", "contract_type", "parties", "effective_date"}

    def test_does_not_mutate_input(self):
        schema = {"properties": {"title": {"type": "string"}}, "required": ["title"]}
        original_required = list(schema["required"])
        original_props = dict(schema["properties"])
        _make_record_schema(schema)
        assert schema["required"] == original_required
        assert schema["properties"] == original_props

    def test_injects_id_field_for_many_mode(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        result = _make_record_schema(schema, id_field="clause_id")
        assert result["properties"]["doc_id"] == {
            "type": "string",
            "description": "document identifier",
        }
        assert result["properties"]["clause_id"] == {
            "type": "string",
            "description": "record identifier",
        }
        assert result["required"][0] == "doc_id"
        assert "clause_id" in result["required"]


def _make_config(extraction_schema: dict, collection: str = "contracts") -> dict:
    """Minimal config dict with one extract-structured step and one structured collection."""
    return {
        "pipelines": [
            {
                "steps": [
                    {
                        "tool": "extract-structured",
                        "collection": collection,
                        "extractor": {
                            "extraction_schema": json.dumps(extraction_schema),
                        },
                    }
                ]
            }
        ],
        "structured_collections": [{"name": collection, "description": "test"}],
    }


class TestInjectPipelineRecordSchemas:
    def test_injects_schema_with_doc_id(self):
        ext_schema = {"type": "object", "properties": {"vendor": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        _inject_pipeline_record_schemas(cfg)
        sc = cfg["structured_collections"][0]
        assert "schema" in sc
        injected = json.loads(sc["schema"])
        assert "doc_id" in injected["properties"]
        assert injected["required"][0] == "doc_id"
        assert "vendor" in injected["properties"]

    def test_schema_matches_make_record_schema(self):
        ext_schema = {"type": "object", "properties": {"amount": {"type": "number"}}}
        cfg = _make_config(ext_schema)
        _inject_pipeline_record_schemas(cfg)
        injected = json.loads(cfg["structured_collections"][0]["schema"])
        expected = _make_record_schema(ext_schema)
        assert injected == expected

    def test_overwrites_existing_schema(self):
        ext_schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        cfg["structured_collections"][0]["schema"] = '{"stale": true}'
        _inject_pipeline_record_schemas(cfg)
        injected = json.loads(cfg["structured_collections"][0]["schema"])
        assert "doc_id" in injected["properties"]
        assert "title" in injected["properties"]

    def test_multiple_pipelines_and_collections(self):
        ext_a = {"type": "object", "properties": {"field_a": {"type": "string"}}}
        ext_b = {"type": "object", "properties": {"field_b": {"type": "number"}}}
        cfg = {
            "pipelines": [
                {
                    "steps": [
                        {
                            "tool": "extract-structured",
                            "collection": "col_a",
                            "extractor": {"extraction_schema": json.dumps(ext_a)},
                        }
                    ]
                },
                {
                    "steps": [
                        {
                            "tool": "extract-structured",
                            "collection": "col_b",
                            "extractor": {"extraction_schema": json.dumps(ext_b)},
                        }
                    ]
                },
            ],
            "structured_collections": [
                {"name": "col_a", "description": "a"},
                {"name": "col_b", "description": "b"},
            ],
        }
        _inject_pipeline_record_schemas(cfg)
        schema_a = json.loads(cfg["structured_collections"][0]["schema"])
        schema_b = json.loads(cfg["structured_collections"][1]["schema"])
        assert "field_a" in schema_a["properties"]
        assert "field_b" in schema_b["properties"]
        assert "doc_id" in schema_a["properties"]
        assert "doc_id" in schema_b["properties"]

    def test_unmatched_collection_not_touched(self):
        ext_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cfg = {
            "pipelines": [
                {
                    "steps": [
                        {
                            "tool": "extract-structured",
                            "collection": "col_a",
                            "extractor": {"extraction_schema": json.dumps(ext_schema)},
                        }
                    ]
                }
            ],
            "structured_collections": [
                {"name": "col_b", "description": "no matching extractor"},
            ],
        }
        _inject_pipeline_record_schemas(cfg)
        assert "schema" not in cfg["structured_collections"][0]

    def test_no_extract_structured_steps(self):
        cfg = {
            "pipelines": [{"steps": [{"tool": "chunk-embed-upsert", "collection": "chunks"}]}],
            "structured_collections": [{"name": "contracts", "description": "test"}],
        }
        _inject_pipeline_record_schemas(cfg)
        assert "schema" not in cfg["structured_collections"][0]

    def test_invalid_json_extraction_schema_raises(self):
        cfg = {
            "pipelines": [
                {
                    "steps": [
                        {
                            "tool": "extract-structured",
                            "collection": "contracts",
                            "extractor": {"extraction_schema": "not valid json {{{"},
                        }
                    ]
                }
            ],
            "structured_collections": [{"name": "contracts", "description": "test"}],
        }
        with pytest.raises(ValueError, match="extraction_schema is not valid JSON"):
            _inject_pipeline_record_schemas(cfg)

    def test_empty_config(self):
        cfg: dict = {}
        _inject_pipeline_record_schemas(cfg)  # must not raise

    def test_one_mode_sets_primary_fields_to_doc_id(self):
        ext_schema = {"type": "object", "properties": {"vendor": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        _inject_pipeline_record_schemas(cfg)
        sc = cfg["structured_collections"][0]
        assert sc["primary_fields"] == ["doc_id"]

    def test_many_mode_sets_primary_fields_to_doc_id_and_id_field(self):
        ext_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        cfg = _make_config(ext_schema, collection="contract_clauses")
        cfg["pipelines"][0]["steps"][0]["extractor"].update(
            {"record_mode": "many", "id_field": "clause_id"}
        )
        _inject_pipeline_record_schemas(cfg)
        sc = cfg["structured_collections"][0]
        assert sc["primary_fields"] == ["doc_id", "clause_id"]
        injected = json.loads(sc["schema"])
        assert "clause_id" in injected["properties"]
        assert "doc_id" in injected["properties"]

    def test_many_mode_without_id_field_omits_it(self):
        ext_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        cfg["pipelines"][0]["steps"][0]["extractor"]["record_mode"] = "many"
        _inject_pipeline_record_schemas(cfg)
        sc = cfg["structured_collections"][0]
        assert sc["primary_fields"] == ["doc_id"]

    def test_many_mode_strips_id_field_from_extraction_schema(self):
        # LLM mistakenly included clause_id in the extraction schema.
        ext_schema = {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "description": "per-record id"},
                "text": {"type": "string"},
            },
            "required": ["clause_id", "text"],
        }
        cfg = _make_config(ext_schema, collection="contract_clauses")
        cfg["pipelines"][0]["steps"][0]["extractor"].update(
            {"record_mode": "many", "id_field": "clause_id"}
        )
        _inject_pipeline_record_schemas(cfg)

        cleaned = json.loads(
            cfg["pipelines"][0]["steps"][0]["extractor"]["extraction_schema"]
        )
        assert "clause_id" not in cleaned["properties"]
        assert "clause_id" not in cleaned.get("required", [])

        sc = cfg["structured_collections"][0]
        record_schema = json.loads(sc["schema"])
        assert "clause_id" in record_schema["properties"]
        assert "doc_id" in record_schema["properties"]
        assert sc["primary_fields"] == ["doc_id", "clause_id"]

    def test_many_mode_strips_id_field_not_in_required(self):
        ext_schema = {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string"},
                "text": {"type": "string"},
            },
        }
        cfg = _make_config(ext_schema, collection="clauses")
        cfg["pipelines"][0]["steps"][0]["extractor"].update(
            {"record_mode": "many", "id_field": "clause_id"}
        )
        _inject_pipeline_record_schemas(cfg)

        cleaned = json.loads(
            cfg["pipelines"][0]["steps"][0]["extractor"]["extraction_schema"]
        )
        assert "clause_id" not in cleaned["properties"]
        assert "clause_id" not in cleaned.get("required", [])

    def test_overwrites_existing_primary_fields(self):
        ext_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        cfg["structured_collections"][0]["primary_fields"] = ["stale"]
        _inject_pipeline_record_schemas(cfg)
        assert cfg["structured_collections"][0]["primary_fields"] == ["doc_id"]


class TestInjectWorkflowOutputSchemas:
    def test_empty_config(self):
        cfg: dict = {}
        _inject_workflow_output_schemas(cfg, {})  # must not raise

    def test_workflow_save_target_uses_workflow_schema_as_is(self):
        workflow_schema = {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Source document id"},
                "clause_id": {"type": "string", "description": "Reviewed clause id"},
            },
        }
        cfg = {
            "workflows": [
                {
                    "steps": [
                        {
                            "tool": "structured-save",
                            "collection": "clause_compliance_findings",
                        }
                    ]
                }
            ],
            "structured_collections": [
                {
                    "name": "clause_compliance_findings",
                    "description": "findings",
                    "schema": '{"stale": true}',
                },
            ],
        }
        _inject_workflow_output_schemas(
            cfg,
            {"clause_compliance_findings": json.dumps(workflow_schema)},
        )
        injected = json.loads(cfg["structured_collections"][0]["schema"])
        assert injected == workflow_schema

    def test_invalid_json_workflow_schema_raises(self):
        cfg = {
            "workflows": [{"steps": [{"tool": "structured-save", "collection": "findings"}]}],
            "structured_collections": [{"name": "findings", "description": "test"}],
        }
        with pytest.raises(ValueError, match="workflow schema for collection 'findings' is not valid JSON"):
            _inject_workflow_output_schemas(cfg, {"findings": "not valid json {{{"})

    def test_unmatched_workflow_schema_not_applied(self):
        cfg = {
            "workflows": [
                {
                    "steps": [
                        {"tool": "structured-save", "collection": "other_findings"}
                    ]
                }
            ],
            "structured_collections": [
                {"name": "clause_compliance_findings", "description": "findings"},
            ],
        }
        _inject_workflow_output_schemas(
            cfg,
            {
                "clause_compliance_findings": json.dumps(
                    {"type": "object", "properties": {"status": {"type": "string"}}}
                )
            },
        )
        assert "schema" not in cfg["structured_collections"][0]

# ---------------------------------------------------------------------------
# _validate_extraction_schema
# ---------------------------------------------------------------------------


class TestValidateExtractionSchema:
    def test_valid_schema_returns_no_errors(self):
        schema = {"type": "object", "properties": {"amount": {"type": "number", "description": "Total"}}}
        assert _validate_extraction_schema(schema, "invoices") == []

    def test_non_dict_returns_error(self):
        errors = _validate_extraction_schema("not a dict", "col")
        assert len(errors) == 1
        assert "must be a JSON Schema object" in errors[0]

    def test_doc_id_in_properties_returns_error(self):
        schema = {"properties": {"doc_id": {"type": "string"}, "name": {"type": "string"}}}
        errors = _validate_extraction_schema(schema, "col")
        assert any("doc_id" in e for e in errors)

    def test_empty_properties_returns_error(self):
        schema = {"type": "object", "properties": {}}
        errors = _validate_extraction_schema(schema, "col")
        assert any("at least one field" in e for e in errors)

    def test_missing_properties_key_returns_error(self):
        schema = {"type": "object"}
        errors = _validate_extraction_schema(schema, "col")
        assert any("at least one field" in e for e in errors)

    def test_doc_id_error_returned_before_build_model(self):
        schema = {"properties": {"doc_id": {"type": "string"}, "x": {"type": "string"}}}
        errors = _validate_extraction_schema(schema, "col")
        assert not any("invalid JSON Schema" in e for e in errors)

    def test_unresolvable_ref_returns_error(self):
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/NonExistent"}},
        }
        errors = _validate_extraction_schema(schema, "col")
        assert any("invalid JSON Schema" in e for e in errors)

    def test_error_message_includes_collection_name(self):
        errors = _validate_extraction_schema("bad", "my_collection")
        assert all("my_collection" in e for e in errors)


# ---------------------------------------------------------------------------
# _validate_workflow_output_schema
# ---------------------------------------------------------------------------


class TestValidateWorkflowOutputSchema:
    def test_allows_doc_id_in_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Source doc"},
                "finding_id": {"type": "string", "description": "Finding id"},
            },
        }
        assert _validate_workflow_output_schema(schema, "findings") == []

    def test_empty_properties_returns_error(self):
        schema = {"type": "object", "properties": {}}
        errors = _validate_workflow_output_schema(schema, "findings")
        assert any("at least one field" in e for e in errors)


# ---------------------------------------------------------------------------
# _parse_and_validate_schemas
# ---------------------------------------------------------------------------


class TestParseAndValidateSchemas:
    def test_valid_yaml_returns_parsed_dict_and_no_errors(self):
        parsed, errors = _parse_and_validate_schemas(
            _MINIMAL_SCHEMA_YAML, validator=_validate_extraction_schema
        )
        assert errors == []
        assert isinstance(parsed, dict)
        assert "contracts" in parsed

    def test_invalid_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas(
            "key: [unclosed", validator=_validate_extraction_schema
        )
        assert parsed is None
        assert any("not valid" in e for e in errors)

    def test_non_mapping_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas(
            "- item1\n- item2\n", validator=_validate_extraction_schema
        )
        assert parsed is None
        assert errors

    def test_doc_id_in_collection_returns_parsed_and_errors(self):
        raw = "col:\n  properties:\n    doc_id: {type: string}\n    name: {type: string}\n"
        parsed, errors = _parse_and_validate_schemas(
            raw, validator=_validate_extraction_schema
        )
        assert parsed is not None  # still returns the parsed dict
        assert any("doc_id" in e for e in errors)

    def test_multiple_collections_errors_attributed_correctly(self):
        raw = (
            "valid_col:\n"
            "  type: object\n"
            "  properties:\n"
            "    title: {type: string, description: Title}\n"
            "bad_col:\n"
            "  properties:\n"
            "    doc_id: {type: string}\n"
            "    x: {type: string}\n"
        )
        _, errors = _parse_and_validate_schemas(
            raw, validator=_validate_extraction_schema
        )
        assert any("bad_col" in e for e in errors)
        assert not any("valid_col" in e for e in errors)

    def test_empty_string_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas(
            "", validator=_validate_extraction_schema
        )
        assert parsed is None
        assert errors

    def test_workflow_validator_allows_doc_id(self):
        raw = "findings:\n  type: object\n  properties:\n    doc_id: {type: string, description: source doc}\n    summary: {type: string, description: summary}\n"
        parsed, errors = _parse_and_validate_schemas(
            raw, validator=_validate_workflow_output_schema
        )
        assert parsed is not None
        assert errors == []


# ---------------------------------------------------------------------------
# _serialize_config
# ---------------------------------------------------------------------------


class TestSerializeConfig:
    def test_round_trips_through_from_yaml(self):
        config = AppConfig.from_yaml(_MINIMAL_CONFIG_YAML)
        serialized = config.to_yaml()
        config2 = AppConfig.from_yaml(serialized)
        assert config2.name == config.name
        assert len(config2.pipelines) == len(config.pipelines)

    def test_uses_schema_alias_not_private_name(self):
        record_json = json.dumps({"type": "object", "properties": {"doc_id": {"type": "string"}}})
        config = AppConfig.model_validate({
            "name": "alias-test",
            "structured_collections": [{
                "name": "col",
                "schema": record_json,
                "description": "test collection",
            }],
        })
        serialized = config.to_yaml()
        assert "schema:" in serialized
        assert "schema_:" not in serialized


# ---------------------------------------------------------------------------
# _run_propose_extraction_schemas
# ---------------------------------------------------------------------------


class TestRunProposeSchema:
    async def test_success_on_first_attempt(self):
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_extraction_schemas(llm, _CONVERSATION)
        assert message.startswith("Schemas validated.")
        assert schemas is not None
        assert "contracts" in schemas

    async def test_output_has_no_schema_record_line(self):
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_extraction_schemas(llm, _CONVERSATION)
        assert "  schema: '" not in message

    async def test_retry_then_success(self):
        llm = _make_llm("not: valid: yaml: [[[", _MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_extraction_schemas(llm, _CONVERSATION)
        assert message.startswith("Schemas validated.")
        assert schemas is not None
        assert llm.complete.call_count == 2

    async def test_exhausted_retries_returns_failure_message(self):
        llm = _make_llm("bad", "bad", "bad")
        message, schemas = await _run_propose_extraction_schemas(llm, _CONVERSATION)
        assert "failed after" in message
        assert schemas is None
        assert llm.complete.call_count == 3

    async def test_tool_call_messages_excluded_from_sub_messages(self):
        messages_with_tool_calls = [
            {"role": "user", "content": "build app"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "propose_extraction_schemas", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "1", "content": "Schemas validated."},
        ]
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        await _run_propose_extraction_schemas(llm, messages_with_tool_calls)
        sent_messages = llm.complete.call_args[0][0]
        roles = [m["role"] for m in sent_messages]
        assert "tool" not in roles
        assert not any(m.get("tool_calls") for m in sent_messages)

    async def test_each_collection_has_extraction_schema_line(self):
        two_collection_yaml = (
            "contracts:\n"
            "  type: object\n"
            "  properties:\n"
            "    vendor: {type: string, description: Vendor}\n"
            "clauses:\n"
            "  type: object\n"
            "  properties:\n"
            "    text: {type: string, description: Clause text}\n"
        )
        llm = _make_llm(two_collection_yaml)
        message, schemas = await _run_propose_extraction_schemas(llm, _CONVERSATION)
        assert schemas is not None
        assert set(schemas.keys()) == {"contracts", "clauses"}

# ---------------------------------------------------------------------------
# _run_propose_workflow_schemas
# ---------------------------------------------------------------------------


class TestRunProposeWorkflowSchemas:
    async def test_success_allows_doc_id(self):
        record_schemas = {"contract_clauses": json.dumps({
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "document identifier"},
                "clause_id": {"type": "string", "description": "record identifier"},
                "text": {"type": "string"},
            },
            "required": ["doc_id", "clause_id"],
        })}
        llm = _make_llm(_WORKFLOW_SCHEMA_YAML)
        message, schemas = await _run_propose_workflow_schemas(
            llm,
            _CONVERSATION,
            record_schemas,
        )
        assert message.startswith("Workflow schemas validated.")
        assert schemas is not None
        parsed = json.loads(schemas["clause_compliance_findings"])
        assert "doc_id" in parsed["properties"]

    async def test_includes_record_schemas_in_system_prompt(self):
        record_schemas = {"contract_clauses": json.dumps({
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "document identifier"},
                "clause_id": {"type": "string", "description": "record identifier"},
                "text": {"type": "string"},
            },
        })}
        llm = _make_llm(_WORKFLOW_SCHEMA_YAML)
        await _run_propose_workflow_schemas(llm, _CONVERSATION, record_schemas)
        sent_messages = llm.complete.call_args[0][0]
        assert "Validated pipeline record schemas" in sent_messages[0]["content"]
        assert "contract_clauses" in sent_messages[0]["content"]

    async def test_empty_mapping_is_valid(self):
        llm = _make_llm("{}")
        message, schemas = await _run_propose_workflow_schemas(llm, _CONVERSATION, {})
        assert "No workflow output collections" in message
        assert "propose_workflow_config" in message
        assert schemas == {}


# ---------------------------------------------------------------------------
# _run_propose_pipeline_config
# ---------------------------------------------------------------------------


class TestRunProposePipelineConfig:
    async def test_success_returns_config_dict_record_schemas_and_config_yaml(self):
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        message, config_dict, record_schemas, config_yaml = (
            await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        )
        assert message.startswith("Pipeline config validated.")
        assert config_dict is not None
        assert record_schemas == {}  # no structured collections
        assert config_yaml is not None  # no workflows → final config

    async def test_config_yaml_is_valid_app_config(self):
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        _, _, _, config_yaml = await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        config = AppConfig.from_yaml(config_yaml)
        assert config.name == "test-app"

    async def test_record_schemas_includes_full_schema_with_doc_id(self):
        llm = _make_llm(_CONFIG_YAML_WITH_STRUCTURED)
        _, _, record_schemas, _ = await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        assert "contracts" in record_schemas
        schema = json.loads(record_schemas["contracts"])
        assert "doc_id" in schema["properties"]
        assert "vendor" in schema["properties"]

    async def test_injects_doc_id_into_structured_collection_schema(self):
        llm = _make_llm(_CONFIG_YAML_WITH_STRUCTURED)
        _, _, _, stored_yaml = await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        data = yaml.safe_load(stored_yaml)
        sc = data["structured_collections"][0]
        record_schema = json.loads(sc["schema"])
        assert "doc_id" in record_schema["properties"]
        assert record_schema["required"][0] == "doc_id"

    async def test_validates_when_llm_omits_schema_and_primary_fields(self):
        yaml_without_schema = """\
name: test-app
vector_collections:
  - name: chunks
    description: Semantic search chunks.
structured_collections:
  - name: contracts
    description: Extracted contract facts.
pipelines:
  - name: main
    routing_description: Contract documents.
    steps:
      - tool: chunk-embed-upsert
        collection: chunks
      - tool: extract-structured
        collection: contracts
        extractor:
          type: llm
          extraction_schema: '{"type":"object","properties":{"vendor":{"type":"string","description":"Vendor name"}}}'
          prompt: Extract contract data.
"""
        llm = _make_llm(yaml_without_schema)
        message, _, _, stored_yaml = await _run_propose_pipeline_config(
            llm, _CONVERSATION, {}
        )
        assert message.startswith("Pipeline config validated.")
        data = yaml.safe_load(stored_yaml)
        sc = data["structured_collections"][0]
        assert "doc_id" in json.loads(sc["schema"])["properties"]
        assert sc["primary_fields"] == ["doc_id"]

    async def test_retry_then_success(self):
        llm = _make_llm("not: valid: yaml: [[[", _MINIMAL_CONFIG_YAML)
        message, _, _, _ = await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        assert message.startswith("Pipeline config validated.")
        assert llm.complete.call_count == 2

    async def test_exhausted_retries_returns_failure(self):
        llm = _make_llm("bad", "bad", "bad")
        message, config_dict, record_schemas, config_yaml = (
            await _run_propose_pipeline_config(llm, _CONVERSATION, {})
        )
        assert "failed after" in message
        assert config_dict is None
        assert record_schemas is None
        assert config_yaml is None
        assert llm.complete.call_count == 3

    async def test_extraction_schemas_in_system_prompt(self):
        extraction_schemas = {
            "contracts": '{"type":"object","properties":{"vendor":{"type":"string"}}}'
        }
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        await _run_propose_pipeline_config(llm, _CONVERSATION, extraction_schemas)
        sent = llm.complete.call_args[0][0]
        assert "Validated pipeline extraction schemas" in sent[0]["content"]
        assert "contracts" in sent[0]["content"]


# ---------------------------------------------------------------------------
# _extract_record_schemas
# ---------------------------------------------------------------------------


class TestExtractRecordSchemas:
    def test_returns_schemas_for_collections_with_schema(self):
        config_dict = {
            "structured_collections": [
                {"name": "contracts", "schema": '{"type":"object"}', "description": "x"},
                {"name": "clauses", "schema": '{"type":"object","properties":{}}', "description": "y"},
            ]
        }
        result = _extract_record_schemas(config_dict)
        assert set(result.keys()) == {"contracts", "clauses"}

    def test_excludes_collections_without_schema(self):
        config_dict = {
            "structured_collections": [
                {"name": "no_schema", "description": "missing schema"},
                {"name": "has_schema", "schema": '{"type":"object"}', "description": "x"},
            ]
        }
        result = _extract_record_schemas(config_dict)
        assert "no_schema" not in result
        assert "has_schema" in result

    def test_empty_config_returns_empty_dict(self):
        assert _extract_record_schemas({}) == {}


# ---------------------------------------------------------------------------
# _run_propose_workflow_config
# ---------------------------------------------------------------------------


class TestRunProposeWorkflowConfig:
    async def test_success_returns_validated_message_and_yaml(self):
        llm = _make_llm(_WORKFLOW_ADDITIONS_YAML)
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        assert message == "Config validated."
        assert stored_yaml is not None

    async def test_merged_config_contains_pipeline_and_workflow_collections(self):
        llm = _make_llm(_WORKFLOW_ADDITIONS_YAML)
        _, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        data = yaml.safe_load(stored_yaml)
        names = {sc["name"] for sc in data.get("structured_collections", [])}
        assert "contracts" in names
        assert "clause_compliance_findings" in names

    async def test_injects_workflow_schema_for_save_target(self):
        llm = _make_llm(_WORKFLOW_ADDITIONS_YAML)
        _, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        data = yaml.safe_load(stored_yaml)
        findings = next(
            sc for sc in data["structured_collections"]
            if sc["name"] == "clause_compliance_findings"
        )
        assert json.loads(findings["schema"]) == json.loads(_FINDING_WORKFLOW_SCHEMA)

    async def test_none_pipeline_config_returns_error_without_llm_call(self):
        llm = _make_llm(_WORKFLOW_ADDITIONS_YAML)
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, None, {}, {}
        )
        assert "propose_pipeline_config" in message
        assert stored_yaml is None
        assert llm.complete.call_count == 0

    async def test_empty_workflows_triggers_retry(self):
        no_workflows_yaml = (
            "structured_collections:\n"
            "  - name: clause_compliance_findings\n"
            "    description: findings\n"
            "    primary_fields: [clause_id]\n"
        )
        llm = _make_llm(no_workflows_yaml, _WORKFLOW_ADDITIONS_YAML)
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        assert message == "Config validated."
        assert llm.complete.call_count == 2

    async def test_retry_then_success(self):
        llm = _make_llm("not: valid: yaml: [[[", _WORKFLOW_ADDITIONS_YAML)
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        assert message == "Config validated."
        assert llm.complete.call_count == 2

    async def test_exhausted_retries_returns_failure_and_none(self):
        llm = _make_llm("bad", "bad", "bad")
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS, {}
        )
        assert "failed after" in message
        assert stored_yaml is None
        assert llm.complete.call_count == 3

    async def test_system_prompt_includes_all_three_context_sections(self):
        llm = _make_llm(_WORKFLOW_ADDITIONS_YAML)
        await _run_propose_workflow_config(
            llm, _CONVERSATION, _PIPELINE_CONFIG_DICT, _RECORD_SCHEMAS,
            {"clause_compliance_findings": _FINDING_WORKFLOW_SCHEMA},
        )
        system_prompt = llm.complete.call_args[0][0][0]["content"]
        assert "Validated pipeline config" in system_prompt
        assert "Validated pipeline record schemas" in system_prompt
        assert "Validated workflow output schemas" in system_prompt

    async def test_cross_pipeline_doc_id_filter_triggers_retry(self):
        # First response has a structured-query on pipe-b's collection filtered by
        # input.doc_id from pipe-a's driver. Second response corrects it.
        llm = _make_llm(_BAD_CROSS_PIPELINE_WORKFLOW_YAML, _GOOD_SAME_PIPELINE_WORKFLOW_YAML)
        message, stored_yaml = await _run_propose_workflow_config(
            llm, _CONVERSATION, _TWO_PIPELINE_CONFIG_DICT, _TWO_PIPELINE_RECORD_SCHEMAS,
            {"findings": _SIMPLE_FINDING_WORKFLOW_SCHEMA},
        )
        assert message == "Config validated."
        assert stored_yaml is not None
        assert llm.complete.call_count == 2
        # The error message sent back on retry names the bad collection and explains the fix.
        retry_user_msg = llm.complete.call_args_list[1][0][0][-1]["content"]
        assert "col_b" in retry_user_msg
        assert "vector-search" in retry_user_msg


# ---------------------------------------------------------------------------
# Fixtures for cross-pipeline doc_id filter tests
# ---------------------------------------------------------------------------

_TWO_PIPELINE_CONFIG_DICT = {
    "name": "two-pipeline-app",
    "vector_collections": [
        {"name": "docs_a", "description": "Docs A."},
        {"name": "docs_b", "description": "Docs B."},
    ],
    "structured_collections": [
        {
            "name": "col_a",
            "description": "Records from pipeline A.",
            "schema": '{"type":"object","properties":{"doc_id":{"type":"string","description":"doc id"},"value":{"type":"string","description":"value"}},"required":["doc_id"]}',
            "primary_fields": ["doc_id"],
        },
        {
            "name": "col_b",
            "description": "Records from pipeline B.",
            "schema": '{"type":"object","properties":{"doc_id":{"type":"string","description":"doc id"},"value":{"type":"string","description":"value"}},"required":["doc_id"]}',
            "primary_fields": ["doc_id"],
        },
    ],
    "pipelines": [
        {
            "name": "pipe-a",
            "routing_description": "Type A documents.",
            "steps": [
                {"tool": "chunk-embed-upsert", "collection": "docs_a"},
                {
                    "tool": "extract-structured",
                    "collection": "col_a",
                    "extractor": {
                        "type": "llm",
                        "extraction_schema": '{"type":"object","properties":{"value":{"type":"string","description":"Value"}}}',
                        "prompt": "Extract value.",
                    },
                },
            ],
        },
        {
            "name": "pipe-b",
            "routing_description": "Type B documents.",
            "steps": [
                {"tool": "chunk-embed-upsert", "collection": "docs_b"},
                {
                    "tool": "extract-structured",
                    "collection": "col_b",
                    "extractor": {
                        "type": "llm",
                        "extraction_schema": '{"type":"object","properties":{"value":{"type":"string","description":"Value"}}}',
                        "prompt": "Extract value.",
                    },
                },
            ],
        },
    ],
}

_TWO_PIPELINE_RECORD_SCHEMAS = {
    "col_a": '{"type":"object","properties":{"doc_id":{"type":"string"},"value":{"type":"string"}},"required":["doc_id"]}',
    "col_b": '{"type":"object","properties":{"doc_id":{"type":"string"},"value":{"type":"string"}},"required":["doc_id"]}',
}

_SIMPLE_FINDING_WORKFLOW_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "Source doc id"},
        "result": {"type": "string", "description": "Result"},
    },
})

# Workflow YAML that incorrectly filters a pipe-b collection by a pipe-a doc_id.
_BAD_CROSS_PIPELINE_WORKFLOW_YAML = """\
structured_collections:
  - name: findings
    description: Analysis findings.
workflows:
  - name: analyze
    trigger:
      type: manual
    params_from_collection:
      collection: col_a
      filters:
        doc_id: "{{ doc.doc_id }}"
      params:
        doc_id: "{{ record.doc_id }}"
    steps:
      - id: bad_step
        tool: structured-query
        collection: col_b
        filters:
          doc_id: "{{ input.doc_id }}"
      - id: judge
        tool: llm-structured
        prompt: "Judge."
        input:
          context: "{{ steps.bad_step.records }}"
        output_schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"Source doc id"},"result":{"type":"string","description":"Result"}}}'
      - id: save
        tool: structured-save
        collection: findings
        primary_fields: [doc_id]
        records:
          - "{{ steps.judge.output }}"
"""

# Corrected workflow using a same-pipeline query instead.
_GOOD_SAME_PIPELINE_WORKFLOW_YAML = """\
structured_collections:
  - name: findings
    description: Analysis findings.
workflows:
  - name: analyze
    trigger:
      type: manual
    params_from_collection:
      collection: col_a
      filters:
        doc_id: "{{ doc.doc_id }}"
      params:
        doc_id: "{{ record.doc_id }}"
    steps:
      - id: load_records
        tool: structured-query
        collection: col_a
        filters:
          doc_id: "{{ input.doc_id }}"
      - id: judge
        tool: llm-structured
        prompt: "Judge."
        input:
          context: "{{ steps.load_records.records }}"
        output_schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"Source doc id"},"result":{"type":"string","description":"Result"}}}'
      - id: save
        tool: structured-save
        collection: findings
        primary_fields: [doc_id]
        records:
          - "{{ steps.judge.output }}"
"""


# ---------------------------------------------------------------------------
# _build_collection_to_pipeline_map
# ---------------------------------------------------------------------------


class TestBuildCollectionToPipelineMap:
    def test_empty_config_returns_empty_map(self):
        assert _build_collection_to_pipeline_map({}) == {}

    def test_maps_each_collection_to_its_pipeline(self):
        cfg = {
            "pipelines": [
                {
                    "name": "pipe-a",
                    "steps": [
                        {"tool": "extract-structured", "collection": "col_a"},
                        {"tool": "chunk-embed-upsert", "collection": "chunks"},
                    ],
                },
                {
                    "name": "pipe-b",
                    "steps": [
                        {"tool": "extract-structured", "collection": "col_b"},
                    ],
                },
            ]
        }
        result = _build_collection_to_pipeline_map(cfg)
        assert result == {"col_a": "pipe-a", "chunks": "pipe-a", "col_b": "pipe-b"}

    def test_step_without_collection_key_is_ignored(self):
        cfg = {
            "pipelines": [
                {
                    "name": "pipe-a",
                    "steps": [{"tool": "some-tool"}],
                }
            ]
        }
        assert _build_collection_to_pipeline_map(cfg) == {}

    def test_no_pipelines_key_returns_empty_map(self):
        assert _build_collection_to_pipeline_map({"workflows": []}) == {}


# ---------------------------------------------------------------------------
# _validate_workflow_cross_pipeline_doc_id_filters
# ---------------------------------------------------------------------------


def _two_pipeline_base() -> dict:
    """Config with two pipelines: vendor-contracts (col_a) and policy-documents (col_b)."""
    return {
        "pipelines": [
            {
                "name": "vendor-contracts",
                "steps": [
                    {"tool": "extract-structured", "collection": "vendor_contract_clauses"},
                    {"tool": "extract-structured", "collection": "vendor_contract_metadata"},
                ],
            },
            {
                "name": "policy-documents",
                "steps": [
                    {"tool": "extract-structured", "collection": "policy_rules"},
                    {"tool": "extract-structured", "collection": "policy_documents"},
                    {"tool": "chunk-embed-upsert", "collection": "policy_chunks"},
                ],
            },
        ],
    }


class TestValidateWorkflowCrossPipelineDocIdFilters:
    def test_no_workflows_returns_no_errors(self):
        cfg = {**_two_pipeline_base(), "workflows": []}
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

    def test_empty_config_returns_no_errors(self):
        assert _validate_workflow_cross_pipeline_doc_id_filters({}) == []

    def test_same_pipeline_doc_id_filter_is_ok(self):
        # Querying vendor_contract_clauses by input.doc_id is fine — same pipeline.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "load_clauses",
                            "tool": "structured-query",
                            "collection": "vendor_contract_clauses",
                            "filters": {"doc_id": "{{ input.doc_id }}"},
                        }
                    ],
                }
            ],
        }
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

    def test_cross_pipeline_input_doc_id_returns_error(self):
        # Querying policy_rules with a contract doc_id: the exact bug from the live log.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "retrieve_policy_rules",
                            "tool": "structured-query",
                            "collection": "policy_rules",
                            "filters": {"doc_id": "{{ input.doc_id }}"},
                        }
                    ],
                }
            ],
        }
        errors = _validate_workflow_cross_pipeline_doc_id_filters(cfg)
        assert len(errors) == 1
        assert "policy_rules" in errors[0]
        assert "vendor-contracts" in errors[0]
        assert "policy-documents" in errors[0]
        assert "vector-search" in errors[0]

    def test_cross_pipeline_item_doc_id_in_foreach_returns_error(self):
        # Inside a foreach, filtering policy_rules by item.doc_id is equally wrong.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "load_clauses",
                            "tool": "structured-query",
                            "collection": "vendor_contract_clauses",
                            "filters": {"doc_id": "{{ input.doc_id }}"},
                        },
                        {
                            "id": "review_each_clause",
                            "foreach": "{{ steps.load_clauses.records }}",
                            "steps": [
                                {
                                    "id": "bad_nested_step",
                                    "tool": "structured-query",
                                    "collection": "policy_rules",
                                    "filters": {"doc_id": "{{ item.doc_id }}"},
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        errors = _validate_workflow_cross_pipeline_doc_id_filters(cfg)
        assert len(errors) == 1
        assert "bad_nested_step" in errors[0]
        assert "policy_rules" in errors[0]

    def test_multiple_cross_pipeline_steps_each_flagged(self):
        # Both policy_rules and policy_documents queried with contract doc_id.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "foreach_block",
                            "foreach": "{{ [] }}",
                            "steps": [
                                {
                                    "id": "bad_rules",
                                    "tool": "structured-query",
                                    "collection": "policy_rules",
                                    "filters": {"doc_id": "{{ input.doc_id }}"},
                                },
                                {
                                    "id": "bad_docs",
                                    "tool": "structured-query",
                                    "collection": "policy_documents",
                                    "filters": {"doc_id": "{{ input.doc_id }}"},
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        errors = _validate_workflow_cross_pipeline_doc_id_filters(cfg)
        assert len(errors) == 2
        flagged_ids = {e.split("step '")[1].split("'")[0] for e in errors}
        assert flagged_ids == {"bad_rules", "bad_docs"}

    def test_fixed_string_doc_id_filter_not_flagged(self):
        # A hard-coded doc_id value (no Jinja2) is not flagged.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "lookup",
                            "tool": "structured-query",
                            "collection": "policy_rules",
                            "filters": {"doc_id": "some-fixed-policy-id"},
                        }
                    ],
                }
            ],
        }
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

    def test_cross_pipeline_non_doc_id_filter_not_flagged(self):
        # Cross-pipeline query filtering by a field other than doc_id is not caught.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "lookup",
                            "tool": "structured-query",
                            "collection": "policy_rules",
                            "filters": {"topic": "{{ item.clause_type }}"},
                        }
                    ],
                }
            ],
        }
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

    def test_driver_collection_not_in_any_pipeline_skips_gracefully(self):
        # params_from_collection references a collection with no known pipeline.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "orphan",
                    "params_from_collection": {"collection": "unknown_collection"},
                    "steps": [
                        {
                            "id": "step",
                            "tool": "structured-query",
                            "collection": "policy_rules",
                            "filters": {"doc_id": "{{ input.doc_id }}"},
                        }
                    ],
                }
            ],
        }
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

    def test_vector_search_on_cross_pipeline_collection_not_flagged(self):
        # vector-search is the correct cross-pipeline tool — should never be flagged.
        cfg = {
            **_two_pipeline_base(),
            "workflows": [
                {
                    "name": "check-contracts",
                    "params_from_collection": {
                        "collection": "vendor_contract_metadata",
                        "params": {"doc_id": "{{ record.doc_id }}"},
                    },
                    "steps": [
                        {
                            "id": "retrieve_policy_context",
                            "tool": "vector-search",
                            "collection": "policy_chunks",
                            "query": "{{ item.clause_text }}",
                            "top_k": 5,
                        }
                    ],
                }
            ],
        }
        assert _validate_workflow_cross_pipeline_doc_id_filters(cfg) == []

