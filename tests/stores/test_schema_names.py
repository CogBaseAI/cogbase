"""Unit tests for validate_resource_name and its callers."""

import pytest
from pydantic import ValidationError

from cogbase.stores.schema import validate_resource_name, CollectionSchema, FieldSchema, FieldType
from cogbase.stores.vector.base import VectorCollectionSchema
from cogbase.config.config import AppConfig


# ---------------------------------------------------------------------------
# validate_resource_name — direct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "foo",
    "foo_bar",
    "foo-bar",
    "foo123",
    "_private",
    "CamelCase",
    "a",
    "_",
    "doc-chunks",
    "document_summary",
    "my-collection-v2",
])
def test_valid_names(name):
    assert validate_resource_name(name) == name


@pytest.mark.parametrize("name", [
    "",
    "123abc",       # starts with digit
    "-foo",         # starts with hyphen
    "foo bar",      # space
    "foo!",         # special char
    "foo.bar",      # dot
    "foo/bar",      # slash
])
def test_invalid_names(name):
    with pytest.raises(ValueError, match="invalid"):
        validate_resource_name(name)


# ---------------------------------------------------------------------------
# CollectionSchema
# ---------------------------------------------------------------------------

def _minimal_schema(name: str) -> CollectionSchema:
    return CollectionSchema(
        name=name,
        description="Test.",
        primary_fields=["id"],
        fields={"id": FieldSchema(type=FieldType.STRING)},
    )


@pytest.mark.parametrize("name", ["my_table", "my-table", "doc-chunks", "_internal"])
def test_collection_schema_valid_names(name):
    schema = _minimal_schema(name)
    assert schema.name == name


@pytest.mark.parametrize("name", ["123bad", "-bad", "bad name", "bad!name"])
def test_collection_schema_invalid_names(name):
    with pytest.raises(ValidationError, match="invalid"):
        _minimal_schema(name)


# ---------------------------------------------------------------------------
# VectorCollectionSchema
# ---------------------------------------------------------------------------

def _minimal_vector_schema(name: str) -> VectorCollectionSchema:
    return VectorCollectionSchema(name=name, dimensions=128, description="Test.")


@pytest.mark.parametrize("name", ["doc_chunks", "doc-chunks", "summary", "_vecs"])
def test_vector_schema_valid_names(name):
    schema = _minimal_vector_schema(name)
    assert schema.name == name


@pytest.mark.parametrize("name", ["123bad", "-bad", "bad name"])
def test_vector_schema_invalid_names(name):
    with pytest.raises(ValidationError, match="invalid"):
        _minimal_vector_schema(name)


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

def _minimal_app_config(name: str) -> AppConfig:
    return AppConfig.model_validate({"name": name})


@pytest.mark.parametrize("name", ["my-app", "my_app", "MyApp", "app123", "_hidden"])
def test_app_config_valid_names(name):
    cfg = _minimal_app_config(name)
    assert cfg.name == name


@pytest.mark.parametrize("name", ["123bad", "-bad", "bad name", "bad!name"])
def test_app_config_invalid_names(name):
    with pytest.raises(ValidationError, match="invalid"):
        _minimal_app_config(name)
