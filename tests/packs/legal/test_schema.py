"""Tests for build_contracts_schema."""

from __future__ import annotations

import pytest

from cogbase.stores.schema import FieldSchema, FieldType
from packs.legal.contract_analyst.schema import CONTRACTS_SCHEMA, build_contracts_schema


def test_no_args_returns_equivalent_schema():
    schema = build_contracts_schema()
    assert schema.fields.keys() == CONTRACTS_SCHEMA.fields.keys()
    assert schema.name == CONTRACTS_SCHEMA.name
    assert schema.id_field == CONTRACTS_SCHEMA.id_field


def test_does_not_mutate_default_schema():
    original_fields = set(CONTRACTS_SCHEMA.fields.keys())
    build_contracts_schema(
        extra_fields={"risk_score": FieldSchema(type=FieldType.FLOAT, nullable=True)},
        exclude={"liability_cap"},
    )
    assert set(CONTRACTS_SCHEMA.fields.keys()) == original_fields


def test_extra_fields_appended():
    extra = {"risk_score": FieldSchema(type=FieldType.FLOAT, nullable=True)}
    schema = build_contracts_schema(extra_fields=extra)
    assert "risk_score" in schema.fields
    assert schema.fields["risk_score"].type == FieldType.FLOAT


def test_extra_field_index_preserved():
    extra = {"jurisdiction": FieldSchema(type=FieldType.STRING, nullable=True, index=True)}
    schema = build_contracts_schema(extra_fields=extra)
    assert schema.fields["jurisdiction"].index is True


def test_exclude_removes_field():
    schema = build_contracts_schema(exclude={"liability_cap", "notice_period_days"})
    assert "liability_cap" not in schema.fields
    assert "notice_period_days" not in schema.fields


def test_non_excluded_fields_preserved():
    schema = build_contracts_schema(exclude={"liability_cap"})
    for name in CONTRACTS_SCHEMA.fields:
        if name != "liability_cap":
            assert name in schema.fields


def test_exclude_and_extra_combined():
    schema = build_contracts_schema(
        extra_fields={"jurisdiction": FieldSchema(type=FieldType.STRING, nullable=True)},
        exclude={"liability_cap"},
    )
    assert "jurisdiction" in schema.fields
    assert "liability_cap" not in schema.fields


def test_exclude_core_field_raises():
    for core in ("contract_id", "doc_id"):
        with pytest.raises(ValueError, match="core fields"):
            build_contracts_schema(exclude={core})


def test_extra_field_duplicates_existing_raises():
    with pytest.raises(ValueError, match="duplicates existing"):
        build_contracts_schema(
            extra_fields={"parties": FieldSchema(type=FieldType.JSON, nullable=True)}
        )


def test_returned_schema_is_valid():
    """CollectionSchema validation passes — name, id_field, and fields are consistent."""
    schema = build_contracts_schema(
        extra_fields={"score": FieldSchema(type=FieldType.FLOAT, nullable=True)},
        exclude={"currency"},
    )
    assert schema.id_field in schema.fields


def test_payment_terms_field_carries_nested_json_schema():
    field = CONTRACTS_SCHEMA.fields["payment_terms"]
    assert field.type == FieldType.JSON
    assert field.json_schema is not None
    assert '"schedule"' in field.json_schema
    assert '"due_date"' in field.json_schema
    assert '"late_penalty"' in field.json_schema
    assert '"verbatim"' in field.json_schema
