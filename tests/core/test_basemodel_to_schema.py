"""Tests for cogbase.core.basemodel_to_schema."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from cogbase.stores.schema import FieldType
from cogbase.core.basemodel_to_schema import cls_generate_schema, cls_json_schema_for_llm


# ---------------------------------------------------------------------------
# Helpers — small models used across multiple tests
# ---------------------------------------------------------------------------

class _Primitives(BaseModel):
    s: str
    i: int
    f: float
    b: bool


class _Optionals(BaseModel):
    s: str | None = None
    i: int | None = None
    f: float | None = None
    b: bool | None = None


class _Inner(BaseModel):
    x: str = Field(description="inner x")
    y: int = Field(description="inner y")


class _Nested(BaseModel):
    name: str
    inner: _Inner


class _Lists(BaseModel):
    tags: list[str]
    items: list[_Inner]


class _Mixed(BaseModel):
    title: str | None = None
    score: float | None = None
    count: int | None = None
    tags: list[str] = Field(default_factory=list)
    children: list[_Inner] = Field(default_factory=list)
    child: _Inner | None = None


# ---------------------------------------------------------------------------
# cls_generate_schema — primitive types
# ---------------------------------------------------------------------------

def test_str_maps_to_string():
    assert cls_generate_schema(_Primitives)["s"].type == FieldType.STRING


def test_int_maps_to_integer():
    assert cls_generate_schema(_Primitives)["i"].type == FieldType.INTEGER


def test_float_maps_to_float():
    assert cls_generate_schema(_Primitives)["f"].type == FieldType.FLOAT


def test_bool_maps_to_boolean():
    assert cls_generate_schema(_Primitives)["b"].type == FieldType.BOOLEAN


# ---------------------------------------------------------------------------
# cls_generate_schema — Optional / X | None unwrapping
# ---------------------------------------------------------------------------

def test_optional_str_maps_to_string():
    assert cls_generate_schema(_Optionals)["s"].type == FieldType.STRING


def test_optional_int_maps_to_integer():
    assert cls_generate_schema(_Optionals)["i"].type == FieldType.INTEGER


def test_optional_float_maps_to_float():
    assert cls_generate_schema(_Optionals)["f"].type == FieldType.FLOAT


def test_optional_bool_maps_to_boolean():
    assert cls_generate_schema(_Optionals)["b"].type == FieldType.BOOLEAN


def test_typing_optional_str_maps_to_string():
    class M(BaseModel):
        v: Optional[str] = None
    assert cls_generate_schema(M)["v"].type == FieldType.STRING


def test_typing_optional_float_maps_to_float():
    class M(BaseModel):
        v: Optional[float] = None
    assert cls_generate_schema(M)["v"].type == FieldType.FLOAT


# ---------------------------------------------------------------------------
# cls_generate_schema — list and nested BaseModel map to JSON
# ---------------------------------------------------------------------------

def test_list_of_str_maps_to_json():
    assert cls_generate_schema(_Lists)["tags"].type == FieldType.JSON


def test_list_of_basemodel_maps_to_json():
    assert cls_generate_schema(_Lists)["items"].type == FieldType.JSON


def test_nested_basemodel_maps_to_json():
    assert cls_generate_schema(_Nested)["inner"].type == FieldType.JSON


def test_nested_basemodel_json_schema_is_included():
    schema = cls_generate_schema(_Nested)
    assert schema["inner"].json_schema is not None
    assert '"x": "string, inner x"' in schema["inner"].json_schema


def test_list_of_str_json_schema_is_string_array():
    schema = cls_generate_schema(_Lists)
    assert schema["tags"].json_schema == '["string"]'


def test_list_of_int_json_schema_is_integer_array():
    class M(BaseModel):
        counts: list[int]
    schema = cls_generate_schema(M)
    assert schema["counts"].json_schema == '["integer"]'


# ---------------------------------------------------------------------------
# cls_generate_schema — all fields present and correct on a mixed model
# ---------------------------------------------------------------------------

def test_mixed_model_all_fields_present():
    result = cls_generate_schema(_Mixed)
    assert set(result.keys()) == {"title", "score", "count", "tags", "children", "child"}


def test_mixed_model_types():
    result = cls_generate_schema(_Mixed)
    assert result["title"].type == FieldType.STRING
    assert result["score"].type == FieldType.FLOAT
    assert result["count"].type == FieldType.INTEGER
    assert result["tags"].type == FieldType.JSON
    assert result["children"].type == FieldType.JSON
    assert result["child"].type == FieldType.JSON


# ---------------------------------------------------------------------------
# cls_json_schema_for_llm — primitive fields
# ---------------------------------------------------------------------------

def test_primitive_field_with_description():
    class M(BaseModel):
        name: str = Field(description="the name")
    out = cls_json_schema_for_llm(M)
    assert '"name": "string, the name"' in out


def test_primitive_field_without_description():
    class M(BaseModel):
        name: str
    out = cls_json_schema_for_llm(M)
    assert '"name": "string"' in out


def test_int_field_type_str():
    class M(BaseModel):
        count: int = Field(description="item count")
    out = cls_json_schema_for_llm(M)
    assert '"count": "int, item count"' in out


# ---------------------------------------------------------------------------
# cls_json_schema_for_llm — list of BaseModel renders nested schema
# ---------------------------------------------------------------------------

def test_list_of_basemodel_renders_nested():
    class M(BaseModel):
        items: list[_Inner]
    out = cls_json_schema_for_llm(M)
    assert '"items": [' in out
    assert '"x": "string, inner x"' in out
    assert '"y": "int, inner y"' in out


# ---------------------------------------------------------------------------
# cls_json_schema_for_llm — list of primitive uses field description
# ---------------------------------------------------------------------------

def test_list_of_str_renders_description():
    class M(BaseModel):
        tags: list[str] = Field(description="list of tags")
    out = cls_json_schema_for_llm(M)
    assert '"tags": ["list of tags"]' in out


# ---------------------------------------------------------------------------
# cls_json_schema_for_llm — nested BaseModel renders inline
# ---------------------------------------------------------------------------

def test_nested_basemodel_renders_inline():
    out = cls_json_schema_for_llm(_Nested)
    assert '"inner": {' in out
    assert '"x": "string, inner x"' in out


# ---------------------------------------------------------------------------
# cls_json_schema_for_llm — output is wrapped in braces
# ---------------------------------------------------------------------------

def test_output_starts_and_ends_with_braces():
    out = cls_json_schema_for_llm(_Primitives).strip()
    assert out.startswith("{")
    assert out.endswith("}")
