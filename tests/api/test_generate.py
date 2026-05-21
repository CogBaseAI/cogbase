"""Unit tests for helper functions in api/routers/generate.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import yaml

from api.routers.generate import (
    _chat_turn_events,
    _extract_record_schemas,
    _inject_pipeline_record_schemas,
    _inject_workflow_output_schemas,
    _make_record_schema,
    _parse_and_validate_schemas,
    _run_propose_extraction_schemas,
    _run_propose_pipeline_config,
    _run_propose_workflow_config,
    _run_propose_workflow_schemas,
    _serialize_config,
    _validate_extraction_schema,
    _validate_workflow_output_schema,
    chat,
)
from api.models import GenerateChatRequest
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
        # Schema as produced by the LLM in practice: has "type", nested "items",
        # and "additionalProperties": false.
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

        # top-level keywords preserved
        assert result["type"] == "object"
        assert result["additionalProperties"] is False

        # doc_id injected
        assert result["properties"]["doc_id"] == {
            "type": "string",
            "description": "document identifier",
        }
        # original fields intact
        assert result["properties"]["parties"] == {"type": "array", "items": {"type": "string"}}

        # doc_id leads required; original fields follow
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
        # doc_id leads required, id_field also required
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

        # id_field must be absent from the extractor's extraction_schema
        cleaned = json.loads(
            cfg["pipelines"][0]["steps"][0]["extractor"]["extraction_schema"]
        )
        assert "clause_id" not in cleaned["properties"]
        assert "clause_id" not in cleaned.get("required", [])

        # id_field must still appear in the collection schema (injected by _make_record_schema)
        sc = cfg["structured_collections"][0]
        record_schema = json.loads(sc["schema"])
        assert "clause_id" in record_schema["properties"]
        assert "doc_id" in record_schema["properties"]
        assert sc["primary_fields"] == ["doc_id", "clause_id"]

    def test_many_mode_strips_id_field_not_in_required(self):
        # id_field present in properties but not in required — still stripped cleanly.
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
        # doc_id error short-circuits before JSON Schema validation
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
        serialized = _serialize_config(config)
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
        serialized = _serialize_config(config)
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
        # assistant message with tool_calls is also excluded
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

    async def test_config_yaml_none_when_llm_includes_workflows(self):
        # If the LLM produces a workflow in the pipeline step, config_yaml is
        # withheld — the workflow section must be generated by propose_workflow_config.
        # Uses a self-contained config where all schemas are pre-populated so
        # full AppConfig validation succeeds.
        valid_workflow_yaml = """\
name: test-app
vector_collections:
  - name: chunks
    description: Chunks.
structured_collections:
  - name: contracts
    description: Contracts.
  - name: findings
    description: Findings.
    schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"doc id"},"status":{"type":"string","description":"status"}}}'
    primary_fields: [doc_id]
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
          extraction_schema: '{"type":"object","properties":{"vendor":{"type":"string","description":"Vendor"}}}'
          prompt: Extract.
workflows:
  - name: check
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
        prompt: Judge.
        input:
          doc_id: "{{ input.doc_id }}"
        output_schema: '{"type":"object","properties":{"doc_id":{"type":"string","description":"doc id"},"status":{"type":"string","description":"status"}}}'
      - id: save
        tool: structured-save
        collection: findings
        records:
          - "{{ steps.judge.output }}"
"""
        llm = _make_llm(valid_workflow_yaml)
        _, config_dict, _, config_yaml = await _run_propose_pipeline_config(
            llm, _CONVERSATION, {}
        )
        assert config_dict is not None  # pipeline section still validated
        assert config_yaml is None  # workflow app — defer to propose_workflow_config

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


# ---------------------------------------------------------------------------
# chat / chat stream
# ---------------------------------------------------------------------------


class TestChatTurn:
    async def test_chat_drains_shared_stream_and_returns_final_response(self):
        llm = _make_llm("A final response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        response = await chat(body, system_resources)

        assert response.content == "A final response"
        assert response.config_yaml is None
        assert llm.complete_stream.call_count == 1

    async def test_chat_turn_events_emit_result(self):
        llm = _make_llm("A final response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        events = []
        async for event in _chat_turn_events(body, system_resources, log_prefix="test/chat"):
            events.append(event)

        assert events[-1]["type"] == "result"
        assert events[-1]["result"]["content"] == "A final response"
