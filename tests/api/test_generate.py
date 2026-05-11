"""Unit tests for helper functions in api/routers/generate.py."""

from __future__ import annotations

import pytest

from api.routers.generate import _make_record_schema


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
