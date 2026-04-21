"""Unit tests for cogbase/core/json_schema_to_basemodel.py."""

import json
import pytest
from pydantic import BaseModel, ValidationError

from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(properties: dict, required: list[str] | None = None, defs: dict | None = None) -> dict:
    """Minimal JSON Schema object with the given properties."""
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if defs:
        schema["$defs"] = defs
    return schema


# ---------------------------------------------------------------------------
# Input formats
# ---------------------------------------------------------------------------

class TestInputFormats:
    def test_accepts_dict(self):
        schema = _make({"x": {"type": "string"}})
        M = build_model_from_json_schema(schema)
        assert issubclass(M, BaseModel)

    def test_accepts_json_string(self):
        schema = _make({"x": {"type": "string"}})
        M = build_model_from_json_schema(json.dumps(schema))
        assert issubclass(M, BaseModel)

    def test_model_name_from_title(self):
        schema = {**_make({}), "title": "MyModel"}
        M = build_model_from_json_schema(schema)
        assert M.__name__ == "MyModel"

    def test_model_name_fallback_parameter(self):
        M = build_model_from_json_schema(_make({}), model_name="Fallback")
        assert M.__name__ == "Fallback"

    def test_empty_schema_produces_empty_model(self):
        M = build_model_from_json_schema(_make({}))
        assert M.model_fields == {}
        assert M()  # instantiates with no args


# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

class TestScalarTypes:
    def test_string_field(self):
        M = build_model_from_json_schema(_make({"s": {"type": "string"}}, required=["s"]))
        assert M(s="hello").s == "hello"

    def test_integer_field(self):
        M = build_model_from_json_schema(_make({"n": {"type": "integer"}}, required=["n"]))
        assert M(n=42).n == 42

    def test_number_field_maps_to_float(self):
        M = build_model_from_json_schema(_make({"f": {"type": "number"}}, required=["f"]))
        assert M(f=3.14).f == pytest.approx(3.14)

    def test_boolean_field(self):
        M = build_model_from_json_schema(_make({"b": {"type": "boolean"}}, required=["b"]))
        assert M(b=True).b is True

    def test_description_preserved(self):
        schema = _make({"x": {"type": "string", "description": "my desc"}}, required=["x"])
        M = build_model_from_json_schema(schema)
        assert M.model_fields["x"].description == "my desc"


# ---------------------------------------------------------------------------
# Defaults and required
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_required_field_must_be_supplied(self):
        M = build_model_from_json_schema(_make({"x": {"type": "string"}}, required=["x"]))
        with pytest.raises((ValidationError, TypeError)):
            M()

    def test_explicit_null_default(self):
        schema = _make({"x": {"type": "string", "default": None}})
        M = build_model_from_json_schema(schema)
        assert M().x is None

    def test_explicit_scalar_default(self):
        schema = _make({"x": {"type": "integer", "default": 7}})
        M = build_model_from_json_schema(schema)
        assert M().x == 7

    def test_empty_array_default_uses_factory(self):
        schema = _make({"tags": {"type": "array", "items": {"type": "string"}, "default": []}})
        M = build_model_from_json_schema(schema)
        a, b = M(), M()
        assert a.tags == []
        assert a.tags is not b.tags  # separate list per instance


# ---------------------------------------------------------------------------
# Nullable — anyOf form (Pydantic v2 output)
# ---------------------------------------------------------------------------

class TestNullableAnyOf:
    def test_anyof_string_null_is_optional(self):
        schema = _make({"r": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M().r is None
        assert M(r="hi").r == "hi"

    def test_anyof_integer_null(self):
        schema = _make({"n": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M(n=5).n == 5
        assert M().n is None

    def test_anyof_number_null(self):
        schema = _make({"f": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M(f=1.5).f == pytest.approx(1.5)

    def test_anyof_null_first_order(self):
        """null listed before the real type should still resolve correctly."""
        schema = _make({"x": {"anyOf": [{"type": "null"}, {"type": "string"}], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M(x="ok").x == "ok"


# ---------------------------------------------------------------------------
# Nullable — type-array form (draft-04+, hand-authored schemas)
# ---------------------------------------------------------------------------

class TestNullableTypeArray:
    def test_type_array_string_null(self):
        schema = _make({"r": {"type": ["string", "null"], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M().r is None
        assert M(r="hello").r == "hello"

    def test_type_array_integer_null(self):
        schema = _make({"n": {"type": ["integer", "null"], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M(n=3).n == 3
        assert M().n is None

    def test_type_array_null_first_order(self):
        schema = _make({"x": {"type": ["null", "string"], "default": None}})
        M = build_model_from_json_schema(schema)
        assert M(x="ok").x == "ok"

    def test_type_array_with_description(self):
        schema = _make({"r": {"type": ["string", "null"], "description": "role", "default": None}})
        M = build_model_from_json_schema(schema)
        assert M.model_fields["r"].description == "role"


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------

class TestArrayFields:
    def test_primitive_string_array(self):
        schema = _make({"tags": {"type": "array", "items": {"type": "string"}}})
        M = build_model_from_json_schema(schema)
        assert M(tags=["a", "b"]).tags == ["a", "b"]

    def test_primitive_integer_array(self):
        schema = _make({"ids": {"type": "array", "items": {"type": "integer"}}})
        M = build_model_from_json_schema(schema)
        assert M(ids=[1, 2, 3]).ids == [1, 2, 3]

    def test_array_defaults_to_empty_list(self):
        schema = _make({"tags": {"type": "array", "items": {"type": "string"}}})
        M = build_model_from_json_schema(schema)
        assert M().tags == []

    def test_array_instances_are_independent(self):
        schema = _make({"tags": {"type": "array", "items": {"type": "string"}}})
        M = build_model_from_json_schema(schema)
        a, b = M(), M()
        a.tags.append("x")
        assert b.tags == []

    def test_inline_object_array(self):
        schema = _make({
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        })
        M = build_model_from_json_schema(schema)
        result = M(items=[{"name": "Alice"}])
        assert result.items[0].name == "Alice"


# ---------------------------------------------------------------------------
# Nested objects via inline properties
# ---------------------------------------------------------------------------

class TestInlineNestedObjects:
    def test_inline_nested_object(self):
        schema = _make({
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["city"],
            }
        }, required=["address"])
        M = build_model_from_json_schema(schema)
        r = M(address={"city": "Portland", "zip": "97201"})
        assert r.address.city == "Portland"
        assert r.address.zip == "97201"

    def test_nullable_inline_nested_object(self):
        schema = _make({
            "meta": {
                "anyOf": [
                    {"type": "object", "properties": {"key": {"type": "string"}}},
                    {"type": "null"},
                ],
                "default": None,
            }
        })
        M = build_model_from_json_schema(schema)
        assert M().meta is None
        assert M(meta={"key": "v"}).meta.key == "v"


# ---------------------------------------------------------------------------
# $defs / $ref resolution
# ---------------------------------------------------------------------------

class TestDefsAndRef:
    def test_ref_to_defs_object(self):
        schema = {
            "type": "object",
            "properties": {"party": {"$ref": "#/$defs/Party"}},
            "required": ["party"],
            "$defs": {
                "Party": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": ["string", "null"], "default": None},
                    },
                    "required": ["name"],
                }
            },
        }
        M = build_model_from_json_schema(schema)
        r = M(party={"name": "Acme", "role": "buyer"})
        assert r.party.name == "Acme"
        assert r.party.role == "buyer"

    def test_ref_in_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "parties": {"type": "array", "items": {"$ref": "#/$defs/Party"}}
            },
            "$defs": {
                "Party": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        }
        M = build_model_from_json_schema(schema)
        r = M(parties=[{"name": "Alpha"}, {"name": "Beta"}])
        assert [p.name for p in r.parties] == ["Alpha", "Beta"]

    def test_anyof_ref_nullable(self):
        """anyOf: [$ref, null] — Pydantic v2 pattern for Optional[NestedModel]."""
        schema = {
            "type": "object",
            "properties": {
                "terms": {
                    "anyOf": [{"$ref": "#/$defs/Terms"}, {"type": "null"}],
                    "default": None,
                }
            },
            "$defs": {
                "Terms": {
                    "type": "object",
                    "properties": {"schedule": {"type": "string"}},
                }
            },
        }
        M = build_model_from_json_schema(schema)
        assert M().terms is None
        assert M(terms={"schedule": "net-30"}).terms.schedule == "net-30"

    def test_missing_ref_raises(self):
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
            "$defs": {},
        }
        with pytest.raises(ValueError, match="not found"):
            build_model_from_json_schema(schema)

    def test_definitions_key_as_alias_for_defs(self):
        """Support the older ``definitions`` key alongside ``$defs``."""
        schema = {
            "type": "object",
            "properties": {"tag": {"$ref": "#/definitions/Tag"}},
            "required": ["tag"],
            "definitions": {
                "Tag": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }
            },
        }
        M = build_model_from_json_schema(schema)
        assert M(tag={"value": "important"}).tag.value == "important"


# ---------------------------------------------------------------------------
# Round-trip with ContractExtraction.model_json_schema()
# ---------------------------------------------------------------------------

class TestContractExtractionRoundTrip:
    """Verify the builder faithfully reconstructs ContractExtraction from its own schema."""

    @pytest.fixture(scope="class")
    def DynModel(self):
        from examples.contract_analyst_demo.schema import ContractExtraction
        return build_model_from_json_schema(ContractExtraction.model_json_schema())

    def test_field_names_match(self, DynModel):
        from examples.contract_analyst_demo.schema import ContractExtraction
        assert set(DynModel.model_fields) == set(ContractExtraction.model_fields)

    def test_scalar_nullable_fields_default_to_none(self, DynModel):
        r = DynModel()
        assert r.contract_type is None
        assert r.effective_date is None
        assert r.contract_value is None

    def test_list_fields_default_to_empty(self, DynModel):
        r = DynModel()
        assert r.parties == []
        assert r.key_terms == []
        assert r.special_conditions == []

    def test_validate_full_payload(self, DynModel):
        payload = {
            "contract_type": "NDA",
            "effective_date": "2024-01-01",
            "parties": [{"name": "Acme", "role": "licensor"}],
            "payment_terms": {"schedule": "net-30"},
            "key_terms": ["perpetual license"],
            "contract_value": 50000.0,
        }
        r = DynModel.model_validate(payload)
        assert r.contract_type == "NDA"
        assert r.parties[0].name == "Acme"
        assert r.payment_terms.schedule == "net-30"
        assert r.key_terms == ["perpetual license"]
        assert r.contract_value == pytest.approx(50000.0)

    def test_model_validate_json(self, DynModel):
        raw = json.dumps({"contract_type": "SaaS", "governing_law": "New York"})
        r = DynModel.model_validate_json(raw)
        assert r.contract_type == "SaaS"
        assert r.governing_law == "New York"

    def test_invalid_json_raises(self, DynModel):
        with pytest.raises((ValidationError, ValueError)):
            DynModel.model_validate_json("not json")
