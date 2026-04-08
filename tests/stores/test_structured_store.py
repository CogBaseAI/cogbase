"""Contract tests for StructuredStoreBase — run against every concrete adapter."""

import pytest
from pydantic import BaseModel

from cogbase.core.models import Contradiction, Event, Fact
from cogbase.stores.filters import Col
from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType


# ------------------------------------------------------------------
# Record factories
# ------------------------------------------------------------------

def make_fact(**kw) -> Fact:
    defaults = dict(
        type="notice_period", value="60 days",
        raw_text="sixty (60) days written notice",
        doc_id="doc-1", confidence=0.95,
    )
    return Fact(**{**defaults, **kw})


def make_event(**kw) -> Event:
    defaults = dict(session_id="session-1", actor="user", action="query")
    return Event(**{**defaults, **kw})


def make_contradiction(fact_a: Fact, fact_b: Fact, **kw) -> Contradiction:
    return Contradiction(fact_a=fact_a, fact_b=fact_b, **{"conflict_type": "date", **kw})


# ------------------------------------------------------------------
# create_collection
# ------------------------------------------------------------------

def test_create_collection_is_idempotent(structured_store):
    from tests.stores.conftest import FACTS_SCHEMA
    structured_store.create_collection(FACTS_SCHEMA)


def test_create_collection_invalid_name():
    with pytest.raises(Exception, match="invalid"):
        CollectionSchema(
            name="bad name!",
            id_field="id",
            fields={"id": FieldSchema(type=FieldType.STRING)},
        )


def test_create_collection_id_field_must_be_in_fields():
    with pytest.raises(Exception, match="id_field"):
        CollectionSchema(
            name="things",
            id_field="missing_field",
            fields={"id": FieldSchema(type=FieldType.STRING)},
        )


def test_save_to_unknown_collection_raises(structured_store):
    with pytest.raises(KeyError, match="unknown"):
        structured_store.save("unknown", [make_fact()])


# ------------------------------------------------------------------
# Basic save / query
# ------------------------------------------------------------------

def test_save_and_query_no_filters(structured_store):
    fact = make_fact()
    structured_store.save("facts", [fact])
    results = structured_store.query("facts")
    assert len(results) == 1 and results[0]["fact_id"] == fact.fact_id


def test_query_as_deserialises_to_model(structured_store):
    fact = make_fact()
    structured_store.save("facts", [fact])
    results = structured_store.query_as("facts", None, Fact)
    assert isinstance(results[0], Fact) and results[0].fact_id == fact.fact_id


def test_save_upserts_by_id(structured_store):
    fact = make_fact(value="30 days")
    structured_store.save("facts", [fact])
    updated = Fact(
        fact_id=fact.fact_id, type=fact.type, value="60 days",
        raw_text=fact.raw_text, doc_id=fact.doc_id, confidence=fact.confidence,
    )
    structured_store.save("facts", [updated])
    results = structured_store.query("facts")
    assert len(results) == 1 and results[0]["value"] == "60 days"


# ------------------------------------------------------------------
# Equality  (=  and  !=)
# ------------------------------------------------------------------

def test_eq_filter(structured_store):
    structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
    ])
    results = structured_store.query("facts", [Col("type") == "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "notice_period"


def test_ne_filter(structured_store):
    structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
    ])
    results = structured_store.query("facts", [Col("type") != "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "termination_date"


# ------------------------------------------------------------------
# Comparisons  (<  <=  >  >=)
# ------------------------------------------------------------------

def test_gte_filter(structured_store):
    structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    results = structured_store.query("facts", [Col("confidence") >= 0.8])
    assert len(results) == 2
    assert all(r["confidence"] >= 0.8 for r in results)


def test_lt_filter(structured_store):
    structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
    ])
    results = structured_store.query("facts", [Col("confidence") < 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.5


def test_lte_filter(structured_store):
    structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    results = structured_store.query("facts", [Col("confidence") <= 0.8])
    assert len(results) == 2


def test_gt_filter(structured_store):
    structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.95),
    ])
    results = structured_store.query("facts", [Col("confidence") > 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.95


# ------------------------------------------------------------------
# IN / NOT IN
# ------------------------------------------------------------------

def test_in_filter(structured_store):
    structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
        make_fact(type="salary"),
    ])
    results = structured_store.query(
        "facts", [Col("type").in_(["notice_period", "termination_date"])]
    )
    assert len(results) == 2
    assert {r["type"] for r in results} == {"notice_period", "termination_date"}


def test_not_in_filter(structured_store):
    structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
        make_fact(type="salary"),
    ])
    results = structured_store.query(
        "facts", [Col("type").not_in(["notice_period", "termination_date"])]
    )
    assert len(results) == 1 and results[0]["type"] == "salary"


# ------------------------------------------------------------------
# LIKE
# ------------------------------------------------------------------

def test_like_filter_prefix(structured_store):
    structured_store.save("facts", [
        make_fact(value="30 days"),
        make_fact(value="60 days"),
        make_fact(value="annual bonus"),
    ])
    results = structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 2


def test_like_filter_case_insensitive(structured_store):
    structured_store.save("facts", [make_fact(value="Sixty Days")])
    results = structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 1


# ------------------------------------------------------------------
# IS NULL / IS NOT NULL
# ------------------------------------------------------------------

def test_is_null_filter(structured_store):
    structured_store.save("facts", [
        make_fact(page=None),
        make_fact(page=3),
    ])
    results = structured_store.query("facts", [Col("page").is_null()])
    assert len(results) == 1 and results[0]["page"] is None


def test_is_not_null_filter(structured_store):
    structured_store.save("facts", [
        make_fact(page=None),
        make_fact(page=3),
    ])
    results = structured_store.query("facts", [Col("page").is_not_null()])
    assert len(results) == 1 and results[0]["page"] == 3


# ------------------------------------------------------------------
# Combined filters (AND)
# ------------------------------------------------------------------

def test_multiple_filters_are_anded(structured_store):
    structured_store.save("facts", [
        make_fact(type="notice_period", doc_id="doc-1", confidence=0.9),
        make_fact(type="notice_period", doc_id="doc-2", confidence=0.9),
        make_fact(type="termination_date", doc_id="doc-1", confidence=0.9),
    ])
    results = structured_store.query("facts", [
        Col("type") == "notice_period",
        Col("doc_id") == "doc-1",
    ])
    assert len(results) == 1


def test_no_filters_returns_all(structured_store):
    structured_store.save("facts", [make_fact(), make_fact(), make_fact()])
    assert len(structured_store.query("facts")) == 3


def test_no_match_returns_empty(structured_store):
    structured_store.save("facts", [make_fact(type="notice_period")])
    assert structured_store.query("facts", [Col("type") == "nonexistent"]) == []


# ------------------------------------------------------------------
# JSON columns (payload, nested facts)
# ------------------------------------------------------------------

def test_json_payload_roundtrip(structured_store):
    e = make_event(payload={"query": "notice period?", "count": 3})
    structured_store.save("events", [e])
    results = structured_store.query("events", [Col("session_id") == "session-1"])
    assert results[0]["payload"] == {"query": "notice period?", "count": 3}


def test_contradiction_nested_facts_roundtrip(structured_store):
    fa = make_fact(doc_id="doc-1", value="30 days")
    fb = make_fact(doc_id="doc-2", value="60 days")
    structured_store.save("contradictions", [make_contradiction(fa, fb)])
    result = structured_store.query("contradictions")[0]
    assert result["fact_a"]["value"] == "30 days"
    assert result["fact_b"]["value"] == "60 days"


def test_boolean_filter(structured_store):
    fa, fb = make_fact(), make_fact()
    structured_store.save("contradictions", [
        make_contradiction(fa, fb, resolved=False),
        make_contradiction(fa, fb, resolved=True),
    ])
    assert len(structured_store.query("contradictions", [Col("resolved") == False])) == 1
    assert len(structured_store.query("contradictions", [Col("resolved") == True])) == 1


# ------------------------------------------------------------------
# delete_records
# ------------------------------------------------------------------

def test_delete_by_filter(structured_store):
    structured_store.save("facts", [
        make_fact(doc_id="doc-1"),
        make_fact(doc_id="doc-2"),
    ])
    structured_store.delete_records("facts", [Col("doc_id") == "doc-1"])
    remaining = structured_store.query("facts")
    assert len(remaining) == 1 and remaining[0]["doc_id"] == "doc-2"


def test_delete_with_range_filter(structured_store):
    structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    structured_store.delete_records("facts", [Col("confidence") < 0.8])
    remaining = structured_store.query("facts")
    assert len(remaining) == 2
    assert all(r["confidence"] >= 0.8 for r in remaining)


def test_delete_all_with_no_filters(structured_store):
    structured_store.save("facts", [make_fact(), make_fact()])
    structured_store.delete_records("facts")
    assert structured_store.query("facts") == []


def test_delete_no_match_is_noop(structured_store):
    structured_store.save("facts", [make_fact()])
    structured_store.delete_records("facts", [Col("type") == "nonexistent"])
    assert len(structured_store.query("facts")) == 1


# ------------------------------------------------------------------
# Custom collection (domain pack use case)
# ------------------------------------------------------------------

def test_custom_collection_with_rich_filters(structured_store):
    class RiskFlag(BaseModel):
        flag_id: str
        severity: str
        score: float
        description: str
        metadata: dict

    schema = CollectionSchema(
        name="risk_flags",
        id_field="flag_id",
        fields={
            "flag_id":     FieldSchema(type=FieldType.STRING,  nullable=False),
            "severity":    FieldSchema(type=FieldType.STRING,  index=True),
            "score":       FieldSchema(type=FieldType.FLOAT),
            "description": FieldSchema(type=FieldType.STRING),
            "metadata":    FieldSchema(type=FieldType.JSON),
        },
    )
    structured_store.create_collection(schema)

    structured_store.save("risk_flags", [
        RiskFlag(flag_id="f1", severity="high",   score=0.9, description="Missing indemnity", metadata={"page": 3}),
        RiskFlag(flag_id="f2", severity="medium", score=0.6, description="Vague termination",  metadata={"page": 7}),
        RiskFlag(flag_id="f3", severity="low",    score=0.2, description="Minor formatting",   metadata={"page": 1}),
    ])

    # Range + membership
    results = structured_store.query("risk_flags", [
        Col("score") >= 0.5,
        Col("severity").in_(["high", "medium"]),
    ])
    assert len(results) == 2

    # LIKE
    results = structured_store.query("risk_flags", [Col("description").like("%termination%")])
    assert len(results) == 1 and results[0]["flag_id"] == "f2"

    # query_as
    flags = structured_store.query_as("risk_flags", [Col("severity") == "high"], RiskFlag)
    assert flags[0].metadata["page"] == 3
