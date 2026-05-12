"""Unit tests for helper functions in api/routers/generate.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from api.routers.generate import (
    _chat_turn_events,
    _inject_record_schemas,
    _make_record_schema,
    _parse_and_validate_schemas,
    _run_propose_config,
    _run_propose_schema,
    _serialize_config,
    _validate_extraction_schema,
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


def _make_llm(*responses: str) -> MagicMock:
    """Return a mock LLMBase whose complete() yields each string in order."""
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[{"content": r, "tool_calls": None} for r in responses]
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


class TestInjectRecordSchemas:
    def test_injects_schema_with_doc_id(self):
        ext_schema = {"type": "object", "properties": {"vendor": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        _inject_record_schemas(cfg)
        sc = cfg["structured_collections"][0]
        assert "schema" in sc
        injected = json.loads(sc["schema"])
        assert "doc_id" in injected["properties"]
        assert injected["required"][0] == "doc_id"
        assert "vendor" in injected["properties"]

    def test_schema_matches_make_record_schema(self):
        ext_schema = {"type": "object", "properties": {"amount": {"type": "number"}}}
        cfg = _make_config(ext_schema)
        _inject_record_schemas(cfg)
        injected = json.loads(cfg["structured_collections"][0]["schema"])
        expected = _make_record_schema(ext_schema)
        assert injected == expected

    def test_overwrites_existing_schema(self):
        ext_schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        cfg = _make_config(ext_schema)
        cfg["structured_collections"][0]["schema"] = '{"stale": true}'
        _inject_record_schemas(cfg)
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
        _inject_record_schemas(cfg)
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
        _inject_record_schemas(cfg)
        assert "schema" not in cfg["structured_collections"][0]

    def test_no_extract_structured_steps(self):
        cfg = {
            "pipelines": [{"steps": [{"tool": "chunk-embed-upsert", "collection": "chunks"}]}],
            "structured_collections": [{"name": "contracts", "description": "test"}],
        }
        _inject_record_schemas(cfg)
        assert "schema" not in cfg["structured_collections"][0]

    def test_invalid_json_extraction_schema_skipped(self):
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
        _inject_record_schemas(cfg)
        assert "schema" not in cfg["structured_collections"][0]

    def test_empty_config(self):
        cfg: dict = {}
        _inject_record_schemas(cfg)  # must not raise


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
# _parse_and_validate_schemas
# ---------------------------------------------------------------------------


class TestParseAndValidateSchemas:
    def test_valid_yaml_returns_parsed_dict_and_no_errors(self):
        parsed, errors = _parse_and_validate_schemas(_MINIMAL_SCHEMA_YAML)
        assert errors == []
        assert isinstance(parsed, dict)
        assert "contracts" in parsed

    def test_invalid_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas("key: [unclosed")
        assert parsed is None
        assert any("not valid" in e for e in errors)

    def test_non_mapping_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas("- item1\n- item2\n")
        assert parsed is None
        assert errors

    def test_doc_id_in_collection_returns_parsed_and_errors(self):
        raw = "col:\n  properties:\n    doc_id: {type: string}\n    name: {type: string}\n"
        parsed, errors = _parse_and_validate_schemas(raw)
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
        _, errors = _parse_and_validate_schemas(raw)
        assert any("bad_col" in e for e in errors)
        assert not any("valid_col" in e for e in errors)

    def test_empty_string_yaml_returns_none_and_error(self):
        parsed, errors = _parse_and_validate_schemas("")
        assert parsed is None
        assert errors


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
# _run_propose_schema
# ---------------------------------------------------------------------------


class TestRunProposeSchema:
    async def test_success_on_first_attempt(self):
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_schema(llm, _CONVERSATION)
        assert message.startswith("Schemas validated.")
        assert schemas is not None
        assert "contracts" in schemas

    async def test_output_has_no_schema_record_line(self):
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_schema(llm, _CONVERSATION)
        assert "  schema: '" not in message

    async def test_retry_then_success(self):
        llm = _make_llm("not: valid: yaml: [[[", _MINIMAL_SCHEMA_YAML)
        message, schemas = await _run_propose_schema(llm, _CONVERSATION)
        assert message.startswith("Schemas validated.")
        assert schemas is not None
        assert llm.complete.call_count == 2

    async def test_exhausted_retries_returns_failure_message(self):
        llm = _make_llm("bad", "bad", "bad")
        message, schemas = await _run_propose_schema(llm, _CONVERSATION)
        assert "failed after" in message
        assert schemas is None
        assert llm.complete.call_count == 3

    async def test_tool_call_messages_excluded_from_sub_messages(self):
        messages_with_tool_calls = [
            {"role": "user", "content": "build app"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "propose_extraction_schema", "arguments": "{}"}]},
            {"role": "tool", "tool_call_id": "1", "content": "Schemas validated."},
        ]
        llm = _make_llm(_MINIMAL_SCHEMA_YAML)
        await _run_propose_schema(llm, messages_with_tool_calls)
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
        message, schemas = await _run_propose_schema(llm, _CONVERSATION)
        assert schemas is not None
        assert set(schemas.keys()) == {"contracts", "clauses"}


# ---------------------------------------------------------------------------
# _run_propose_config
# ---------------------------------------------------------------------------


class TestRunProposeConfig:
    async def test_success_returns_validated_message_and_yaml(self):
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        message, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        assert message == "Config validated."
        assert stored_yaml is not None

    async def test_stored_yaml_is_valid_app_config(self):
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        _, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        config = AppConfig.from_yaml(stored_yaml)
        assert config.name == "test-app"

    async def test_injects_doc_id_into_structured_collection_schema(self):
        llm = _make_llm(_CONFIG_YAML_WITH_STRUCTURED)
        _, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        data = yaml.safe_load(stored_yaml)
        sc = data["structured_collections"][0]
        record_schema = json.loads(sc["schema"])
        assert "doc_id" in record_schema["properties"]
        assert record_schema["required"][0] == "doc_id"

    async def test_retry_then_success(self):
        llm = _make_llm("not: valid: yaml: [[[", _MINIMAL_CONFIG_YAML)
        message, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        assert message == "Config validated."
        assert llm.complete.call_count == 2

    async def test_exhausted_retries_returns_failure_and_none(self):
        llm = _make_llm("bad", "bad", "bad")
        message, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        assert "failed after" in message
        assert stored_yaml is None
        assert llm.complete.call_count == 3

    async def test_config_without_structured_collections(self):
        llm = _make_llm(_MINIMAL_CONFIG_YAML)
        message, stored_yaml = await _run_propose_config(llm, _CONVERSATION, {})
        assert message == "Config validated."
        data = yaml.safe_load(stored_yaml)
        assert data.get("structured_collections", []) == []


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
        assert llm.complete.call_count == 1

    async def test_chat_turn_events_emit_result(self):
        llm = _make_llm("A final response")
        system_resources = MagicMock(llm=llm)
        body = GenerateChatRequest(text="hello", history=[])

        events = []
        async for event in _chat_turn_events(body, system_resources, log_prefix="test/chat"):
            events.append(event)

        assert events[-1]["type"] == "result"
        assert events[-1]["result"]["content"] == "A final response"
