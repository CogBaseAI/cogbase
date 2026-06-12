"""Contract tests for StructuredStoreBase — run against every concrete adapter."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from cogbase.stores import Col, CollectionSchema, FieldSchema, FieldType
from cogbase.stores.filters import Filter, Op
from cogbase.stores.structured import InMemoryStructuredStore


# ------------------------------------------------------------------
# Record factories
# ------------------------------------------------------------------


class FactRecord(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    value: str
    raw_text: str
    doc_id: str
    page: int | None = None
    confidence: float


class EventRecord(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str
    action: str
    payload: dict = Field(default_factory=dict)


class ContradictionRecord(BaseModel):
    contradiction_id: str = Field(default_factory=lambda: str(uuid4()))
    fact_a: dict
    fact_b: dict
    conflict_type: str
    resolved: bool = False
    resolution_note: str | None = None


def make_fact(**kw) -> FactRecord:
    defaults = dict(
        type="notice_period", value="60 days",
        raw_text="sixty (60) days written notice",
        doc_id="doc-1", confidence=0.95,
    )
    return FactRecord(**{**defaults, **kw})


def make_event(**kw) -> EventRecord:
    defaults = dict(session_id="session-1", actor="user", action="query")
    return EventRecord(**{**defaults, **kw})


def make_contradiction(fact_a: FactRecord, fact_b: FactRecord, **kw) -> ContradictionRecord:
    return ContradictionRecord(
        fact_a=fact_a.model_dump(),
        fact_b=fact_b.model_dump(),
        **{"conflict_type": "date", **kw},
    )


# ------------------------------------------------------------------
# create_collection
# ------------------------------------------------------------------

async def test_create_collection_is_idempotent(structured_store):
    from tests.stores.conftest import FACTS_SCHEMA
    await structured_store.create_collection(FACTS_SCHEMA)


def test_create_collection_invalid_name():
    with pytest.raises(Exception, match="invalid"):
        CollectionSchema(
            name="bad name!",
            description="Test.",
            primary_fields=["id"],
            fields={"id": FieldSchema(type=FieldType.STRING)},
        )


def test_create_collection_primary_fields_must_be_in_fields():
    with pytest.raises(Exception, match="primary_fields"):
        CollectionSchema(
            name="things",
            description="Test.",
            primary_fields=["missing_field"],
            fields={"id": FieldSchema(type=FieldType.STRING)},
        )


async def test_save_to_unknown_collection_raises(structured_store):
    with pytest.raises(KeyError, match="unknown"):
        await structured_store.save("unknown", [make_fact().model_dump()])


# ------------------------------------------------------------------
# Basic save / query
# ------------------------------------------------------------------

async def test_save_and_query_no_filters(structured_store):
    fact = make_fact()
    await structured_store.save("facts", [fact.model_dump()])
    results = await structured_store.query("facts")
    assert len(results) == 1 and results[0]["fact_id"] == fact.fact_id


async def test_query_as_deserialises_to_model(structured_store):
    fact = make_fact()
    await structured_store.save("facts", [fact.model_dump()])
    results = await structured_store.query_as("facts", None, FactRecord)
    assert isinstance(results[0], FactRecord) and results[0].fact_id == fact.fact_id


async def test_save_upserts_by_id(structured_store):
    fact = make_fact(value="30 days")
    await structured_store.save("facts", [fact.model_dump()])
    updated = FactRecord(
        fact_id=fact.fact_id, type=fact.type, value="60 days",
        raw_text=fact.raw_text, doc_id=fact.doc_id, confidence=fact.confidence,
    )
    await structured_store.save("facts", [updated.model_dump()])
    results = await structured_store.query("facts")
    assert len(results) == 1 and results[0]["value"] == "60 days"


# ------------------------------------------------------------------
# Equality  (=  and  !=)
# ------------------------------------------------------------------

async def test_eq_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period").model_dump(),
        make_fact(type="termination_date").model_dump(),
    ])
    results = await structured_store.query("facts", [Col("type") == "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "notice_period"


async def test_ne_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period").model_dump(),
        make_fact(type="termination_date").model_dump(),
    ])
    results = await structured_store.query("facts", [Col("type") != "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "termination_date"


# ------------------------------------------------------------------
# Comparisons  (<  <=  >  >=)
# ------------------------------------------------------------------

async def test_gte_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5).model_dump(),
        make_fact(confidence=0.8).model_dump(),
        make_fact(confidence=0.95).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("confidence") >= 0.8])
    assert len(results) == 2
    assert all(r["confidence"] >= 0.8 for r in results)


async def test_lt_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5).model_dump(),
        make_fact(confidence=0.8).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("confidence") < 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.5


async def test_lte_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5).model_dump(),
        make_fact(confidence=0.8).model_dump(),
        make_fact(confidence=0.95).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("confidence") <= 0.8])
    assert len(results) == 2


async def test_gt_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5).model_dump(),
        make_fact(confidence=0.95).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("confidence") > 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.95


# ------------------------------------------------------------------
# IN / NOT IN
# ------------------------------------------------------------------

async def test_in_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period").model_dump(),
        make_fact(type="termination_date").model_dump(),
        make_fact(type="salary").model_dump(),
    ])
    results = await structured_store.query(
        "facts", [Col("type").in_(["notice_period", "termination_date"])]
    )
    assert len(results) == 2
    assert {r["type"] for r in results} == {"notice_period", "termination_date"}


async def test_not_in_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period").model_dump(),
        make_fact(type="termination_date").model_dump(),
        make_fact(type="salary").model_dump(),
    ])
    results = await structured_store.query(
        "facts", [Col("type").not_in(["notice_period", "termination_date"])]
    )
    assert len(results) == 1 and results[0]["type"] == "salary"


# ------------------------------------------------------------------
# OVERLAPS (JSON-array fields)
# ------------------------------------------------------------------

async def _seed_tagged_collection(store):
    schema = CollectionSchema(
        name="tagged_memories",
        description="Records with a JSON array of entity tags.",
        primary_fields=["memory_id"],
        fields={
            "memory_id": FieldSchema(type=FieldType.STRING, nullable=False),
            "status":    FieldSchema(type=FieldType.STRING, index=True),
            "entities":  FieldSchema(type=FieldType.JSON),
        },
    )
    await store.create_collection(schema)
    await store.save("tagged_memories", [
        {"memory_id": "m1", "status": "active",   "entities": ["acme", "jane"]},
        {"memory_id": "m2", "status": "active",   "entities": ["bob"]},
        {"memory_id": "m3", "status": "active",   "entities": []},
        {"memory_id": "m4", "status": "archived", "entities": ["acme"]},
    ])


async def test_overlaps_filter_matches_any_shared_element(structured_store):
    await _seed_tagged_collection(structured_store)
    results = await structured_store.query(
        "tagged_memories", [Col("entities").overlaps(["acme", "zoe"])]
    )
    assert {r["memory_id"] for r in results} == {"m1", "m4"}


async def test_overlaps_filter_no_shared_element(structured_store):
    await _seed_tagged_collection(structured_store)
    results = await structured_store.query(
        "tagged_memories", [Col("entities").overlaps(["zoe"])]
    )
    assert results == []


async def test_overlaps_filter_empty_array_never_matches(structured_store):
    await _seed_tagged_collection(structured_store)
    results = await structured_store.query(
        "tagged_memories", [Col("entities").overlaps(["acme", "jane", "bob"])]
    )
    assert "m3" not in {r["memory_id"] for r in results}


async def test_overlaps_filter_combines_with_and(structured_store):
    await _seed_tagged_collection(structured_store)
    results = await structured_store.query("tagged_memories", [
        Col("status") == "active",
        Col("entities").overlaps(["acme"]),
    ])
    assert len(results) == 1 and results[0]["memory_id"] == "m1"


def test_overlaps_on_scalar_sql_column_raises():
    from cogbase.stores.filters import to_sql_where

    with pytest.raises(ValueError, match="overlaps"):
        to_sql_where([Col("status").overlaps(["active"])], json_fields=set())


def test_overlaps_on_non_list_value_does_not_match():
    from cogbase.stores.filters import matches

    f = [Col("entities").overlaps(["acme"])]
    assert not matches({"entities": "acme"}, f)
    assert not matches({"entities": None}, f)
    assert matches({"entities": ["acme", "jane"]}, f)


# ------------------------------------------------------------------
# LIKE
# ------------------------------------------------------------------

async def test_like_filter_prefix(structured_store):
    await structured_store.save("facts", [
        make_fact(value="30 days").model_dump(),
        make_fact(value="60 days").model_dump(),
        make_fact(value="annual bonus").model_dump(),
    ])
    results = await structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 2


async def test_like_filter_case_insensitive(structured_store):
    await structured_store.save("facts", [make_fact(value="Sixty Days").model_dump()])
    results = await structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 1


# ------------------------------------------------------------------
# IS NULL / IS NOT NULL
# ------------------------------------------------------------------

async def test_is_null_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(page=None).model_dump(),
        make_fact(page=3).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("page").is_null()])
    assert len(results) == 1 and results[0]["page"] is None


async def test_is_not_null_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(page=None).model_dump(),
        make_fact(page=3).model_dump(),
    ])
    results = await structured_store.query("facts", [Col("page").is_not_null()])
    assert len(results) == 1 and results[0]["page"] == 3


# ------------------------------------------------------------------
# Combined filters (AND)
# ------------------------------------------------------------------

async def test_multiple_filters_are_anded(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period", doc_id="doc-1", confidence=0.9).model_dump(),
        make_fact(type="notice_period", doc_id="doc-2", confidence=0.9).model_dump(),
        make_fact(type="termination_date", doc_id="doc-1", confidence=0.9).model_dump(),
    ])
    results = await structured_store.query("facts", [
        Col("type") == "notice_period",
        Col("doc_id") == "doc-1",
    ])
    assert len(results) == 1


async def test_no_filters_returns_all(structured_store):
    await structured_store.save("facts", [make_fact().model_dump(), make_fact().model_dump(), make_fact().model_dump()])
    assert len(await structured_store.query("facts")) == 3


async def test_no_match_returns_empty(structured_store):
    await structured_store.save("facts", [make_fact(type="notice_period").model_dump()])
    assert await structured_store.query("facts", [Col("type") == "nonexistent"]) == []


# ------------------------------------------------------------------
# JSON columns (payload, nested facts)
# ------------------------------------------------------------------

async def test_json_payload_roundtrip(structured_store):
    e = make_event(payload={"query": "notice period?", "count": 3})
    await structured_store.save("events", [e.model_dump()])
    results = await structured_store.query("events", [Col("session_id") == "session-1"])
    assert results[0]["payload"] == {"query": "notice period?", "count": 3}


async def test_contradiction_nested_facts_roundtrip(structured_store):
    fa = make_fact(doc_id="doc-1", value="30 days")
    fb = make_fact(doc_id="doc-2", value="60 days")
    await structured_store.save("contradictions", [make_contradiction(fa, fb).model_dump()])
    result = (await structured_store.query("contradictions"))[0]
    assert result["fact_a"]["value"] == "30 days"
    assert result["fact_b"]["value"] == "60 days"


async def test_json_nested_field_eq_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query",  "count": 1}).model_dump(),
        make_event(session_id="s2", payload={"kind": "ingest", "count": 5}).model_dump(),
        make_event(session_id="s3", payload={"kind": "query",  "count": 5}).model_dump(),
    ])
    results = await memory_store.query("events", [Col("payload.kind") == "query"])
    assert {r["session_id"] for r in results} == {"s1", "s3"}


async def test_json_nested_field_comparison_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"count": 1}).model_dump(),
        make_event(session_id="s2", payload={"count": 3}).model_dump(),
        make_event(session_id="s3", payload={"count": 5}).model_dump(),
    ])
    results = await memory_store.query("events", [Col("payload.count") >= 3])
    assert {r["session_id"] for r in results} == {"s2", "s3"}


async def test_json_nested_field_combined_with_primitive_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query"}).model_dump(),
        make_event(session_id="s2", payload={"kind": "query"}).model_dump(),
        make_event(session_id="s3", payload={"kind": "ingest"}).model_dump(),
    ])
    results = await memory_store.query("events", [
        Col("payload.kind") == "query",
        Col("session_id") == "s1",
    ])
    assert len(results) == 1 and results[0]["session_id"] == "s1"


async def test_json_nested_missing_key_treated_as_null(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query"}).model_dump(),
        make_event(session_id="s2", payload={}).model_dump(),
    ])
    results = await memory_store.query("events", [Col("payload.kind").is_null()])
    assert len(results) == 1 and results[0]["session_id"] == "s2"


async def test_boolean_filter(structured_store):
    fa, fb = make_fact(), make_fact()
    await structured_store.save("contradictions", [
        make_contradiction(fa, fb, resolved=False).model_dump(),
        make_contradiction(fa, fb, resolved=True).model_dump(),
    ])
    assert len(await structured_store.query("contradictions", [Filter("resolved", Op.EQ, False)])) == 1
    assert len(await structured_store.query("contradictions", [Filter("resolved", Op.EQ, True)])) == 1


# ------------------------------------------------------------------
# delete_records
# ------------------------------------------------------------------

async def test_delete_by_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(doc_id="doc-1").model_dump(),
        make_fact(doc_id="doc-2").model_dump(),
    ])
    await structured_store.delete_records("facts", [Col("doc_id") == "doc-1"])
    remaining = await structured_store.query("facts")
    assert len(remaining) == 1 and remaining[0]["doc_id"] == "doc-2"


async def test_delete_with_range_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5).model_dump(),
        make_fact(confidence=0.8).model_dump(),
        make_fact(confidence=0.95).model_dump(),
    ])
    await structured_store.delete_records("facts", [Col("confidence") < 0.8])
    remaining = await structured_store.query("facts")
    assert len(remaining) == 2
    assert all(r["confidence"] >= 0.8 for r in remaining)


async def test_delete_all_with_no_filters(structured_store):
    await structured_store.save("facts", [make_fact().model_dump(), make_fact().model_dump()])
    await structured_store.delete_records("facts")
    assert await structured_store.query("facts") == []


async def test_delete_no_match_is_noop(structured_store):
    await structured_store.save("facts", [make_fact().model_dump()])
    await structured_store.delete_records("facts", [Col("type") == "nonexistent"])
    assert len(await structured_store.query("facts")) == 1


# ------------------------------------------------------------------
# delete_collection
# ------------------------------------------------------------------

async def test_delete_collection_removes_collection_and_data(structured_store):
    await structured_store.save("facts", [make_fact().model_dump(), make_fact().model_dump()])
    await structured_store.delete_collection("facts")
    with pytest.raises(KeyError):
        await structured_store.query("facts")


async def test_delete_collection_unknown_is_noop(structured_store):
    await structured_store.delete_collection("nonexistent")  # idempotent — must not raise


async def test_delete_collection_leaves_other_collections_intact(structured_store):
    await structured_store.save("facts", [make_fact().model_dump()])
    await structured_store.save("events", [make_event().model_dump()])
    await structured_store.delete_collection("facts")
    # "events" must still be queryable and contain its data
    results = await structured_store.query("events")
    assert len(results) == 1


# ------------------------------------------------------------------
# Custom collection (domain pack use case)
# ------------------------------------------------------------------

async def test_custom_collection_with_rich_filters(structured_store):
    class RiskFlag(BaseModel):
        flag_id: str
        severity: str
        score: float
        description: str
        metadata: dict

    schema = CollectionSchema(
        name="risk_flags",
        description="Extracted risk flags with severity, score, and metadata.",
        primary_fields=["flag_id"],
        fields={
            "flag_id":     FieldSchema(type=FieldType.STRING,  nullable=False),
            "severity":    FieldSchema(type=FieldType.STRING,  index=True),
            "score":       FieldSchema(type=FieldType.FLOAT),
            "description": FieldSchema(type=FieldType.STRING),
            "metadata":    FieldSchema(type=FieldType.JSON),
        },
    )
    await structured_store.create_collection(schema)

    await structured_store.save("risk_flags", [
        RiskFlag(flag_id="f1", severity="high",   score=0.9, description="Missing indemnity", metadata={"page": 3}).model_dump(),
        RiskFlag(flag_id="f2", severity="medium", score=0.6, description="Vague termination",  metadata={"page": 7}).model_dump(),
        RiskFlag(flag_id="f3", severity="low",    score=0.2, description="Minor formatting",   metadata={"page": 1}).model_dump(),
    ])

    # Range + membership
    results = await structured_store.query("risk_flags", [
        Col("score") >= 0.5,
        Col("severity").in_(["high", "medium"]),
    ])
    assert len(results) == 2

    # LIKE
    results = await structured_store.query("risk_flags", [Col("description").like("%termination%")])
    assert len(results) == 1 and results[0]["flag_id"] == "f2"

    # query_as
    flags = await structured_store.query_as("risk_flags", [Col("severity") == "high"], RiskFlag)
    assert flags[0].metadata["page"] == 3


# ------------------------------------------------------------------
# Schema migration — add / remove fields
# ------------------------------------------------------------------

async def test_add_field_to_existing_collection(structured_store):
    """A field added to the schema after initial create becomes queryable."""
    class FactV2(BaseModel):
        fact_id: str
        type: str
        value: str
        raw_text: str
        doc_id: str
        page: int | None = None
        confidence: float
        source: str | None = None  # new field

    schema_v2 = CollectionSchema(
        name="facts",
        description="Extracted facts with type, value, source document, and confidence.",
        primary_fields=["fact_id"],
        fields={
            "fact_id":    FieldSchema(type=FieldType.STRING,  nullable=False),
            "type":       FieldSchema(type=FieldType.STRING,  index=True),
            "value":      FieldSchema(type=FieldType.STRING),
            "raw_text":   FieldSchema(type=FieldType.STRING),
            "doc_id":     FieldSchema(type=FieldType.STRING,  index=True),
            "page":       FieldSchema(type=FieldType.INTEGER, nullable=True),
            "confidence": FieldSchema(type=FieldType.FLOAT),
            "source":     FieldSchema(type=FieldType.STRING,  nullable=True),
        },
    )

    # Save a record with the original schema
    await structured_store.save("facts", [make_fact().model_dump()])

    # Migrate: add the new field
    await structured_store.create_collection(schema_v2)

    # Write a record using the new schema
    new_fact = FactV2(
        fact_id="new-1", type="notice_period", value="30 days",
        raw_text="thirty days", doc_id="doc-new", confidence=0.9,
        source="upload",
    )
    await structured_store.save("facts", [new_fact.model_dump()])

    results = await structured_store.query("facts", [Col("fact_id") == "new-1"])
    assert results[0]["source"] == "upload"


async def test_existing_rows_get_null_for_added_field(structured_store):
    """Rows written before the migration have NULL for the new field."""
    old_fact = make_fact()
    await structured_store.save("facts", [old_fact.model_dump()])

    schema_v2 = CollectionSchema(
        name="facts",
        description="Extracted facts with type, value, source document, and confidence.",
        primary_fields=["fact_id"],
        fields={
            "fact_id":    FieldSchema(type=FieldType.STRING, nullable=False),
            "type":       FieldSchema(type=FieldType.STRING, index=True),
            "value":      FieldSchema(type=FieldType.STRING),
            "raw_text":   FieldSchema(type=FieldType.STRING),
            "doc_id":     FieldSchema(type=FieldType.STRING, index=True),
            "page":       FieldSchema(type=FieldType.INTEGER, nullable=True),
            "confidence": FieldSchema(type=FieldType.FLOAT),
            "source":     FieldSchema(type=FieldType.STRING, nullable=True),
        },
    )
    await structured_store.create_collection(schema_v2)

    results = await structured_store.query("facts", [Col("fact_id") == old_fact.fact_id])
    assert results[0]["source"] is None


async def test_remove_field_from_schema_is_ignored_on_read(structured_store):
    """Removing a field from the schema leaves its column in the store but
    the field is absent from query results — no error is raised."""
    await structured_store.save("facts", [make_fact().model_dump()])

    # New schema without the 'value' field
    schema_no_value = CollectionSchema(
        name="facts",
        description="Extracted facts with type, source document, and confidence.",
        primary_fields=["fact_id"],
        fields={
            "fact_id":    FieldSchema(type=FieldType.STRING, nullable=False),
            "type":       FieldSchema(type=FieldType.STRING, index=True),
            "raw_text":   FieldSchema(type=FieldType.STRING),
            "doc_id":     FieldSchema(type=FieldType.STRING, index=True),
            "page":       FieldSchema(type=FieldType.INTEGER, nullable=True),
            "confidence": FieldSchema(type=FieldType.FLOAT),
        },
    )
    await structured_store.create_collection(schema_no_value)

    results = await structured_store.query("facts")
    assert len(results) == 1
    assert "value" not in results[0]


async def test_migration_is_idempotent(structured_store):
    """Calling create_collection twice with the same schema is safe."""
    from tests.stores.conftest import FACTS_SCHEMA
    await structured_store.create_collection(FACTS_SCHEMA)
    await structured_store.create_collection(FACTS_SCHEMA)
    await structured_store.save("facts", [make_fact().model_dump()])
    assert len(await structured_store.query("facts")) == 1


# ------------------------------------------------------------------
# update_collection — explicit schema migration
# ------------------------------------------------------------------

def _facts_schema_with(**overrides) -> CollectionSchema:
    """Return a CollectionSchema for 'facts' with fields added/removed."""
    from tests.stores.conftest import FACTS_SCHEMA
    base = dict(FACTS_SCHEMA.fields)
    for name, field in overrides.items():
        if field is None:
            base.pop(name, None)
        else:
            base[name] = field
    return CollectionSchema(name="facts", description="Extracted facts with type, value, source document, and confidence.", primary_fields=["fact_id"], fields=base)


async def test_update_collection_add_field_existing_rows_get_null(structured_store):
    """Rows written before update_collection have None for the new field."""
    old_fact = make_fact()
    await structured_store.save("facts", [old_fact.model_dump()])

    new_schema = _facts_schema_with(source=FieldSchema(type=FieldType.STRING, nullable=True))
    await structured_store.update_collection(new_schema)

    results = await structured_store.query("facts", [Col("fact_id") == old_fact.fact_id])
    assert results[0]["source"] is None


async def test_update_collection_add_field_new_rows_can_populate_it(structured_store):
    """Rows written after update_collection can store a value in the new field."""
    class FactWithSource(BaseModel):
        fact_id: str
        type: str
        value: str
        raw_text: str
        doc_id: str
        page: int | None = None
        confidence: float
        source: str | None = None

    new_schema = _facts_schema_with(source=FieldSchema(type=FieldType.STRING, nullable=True))
    await structured_store.update_collection(new_schema)

    record = FactWithSource(
        fact_id="f-new", type="notice_period", value="30 days",
        raw_text="thirty days", doc_id="doc-x", confidence=0.9, source="upload",
    )
    await structured_store.save("facts", [record.model_dump()])

    results = await structured_store.query("facts", [Col("fact_id") == "f-new"])
    assert results[0]["source"] == "upload"


async def test_update_collection_remove_field_data_is_gone(structured_store):
    """Rows written before update_collection no longer expose the removed field."""
    await structured_store.save("facts", [make_fact(value="thirty days").model_dump()])

    new_schema = _facts_schema_with(value=None)  # drop 'value'
    await structured_store.update_collection(new_schema)

    results = await structured_store.query("facts")
    assert len(results) == 1
    assert "value" not in results[0]


async def test_update_collection_add_and_remove_simultaneously(structured_store):
    """Adding and removing fields in a single update_collection call both take effect."""
    await structured_store.save("facts", [make_fact().model_dump()])

    new_schema = _facts_schema_with(
        value=None,  # remove
        source=FieldSchema(type=FieldType.STRING, nullable=True),  # add
    )
    await structured_store.update_collection(new_schema)

    results = await structured_store.query("facts")
    assert len(results) == 1
    assert "value" not in results[0]
    assert "source" in results[0]
    assert results[0]["source"] is None


async def test_update_collection_surviving_fields_data_preserved(structured_store):
    """Data in fields that are neither added nor removed is untouched."""
    fact = make_fact(type="notice_period", doc_id="doc-99", confidence=0.88)
    await structured_store.save("facts", [fact.model_dump()])

    new_schema = _facts_schema_with(value=None)  # remove 'value', leave everything else
    await structured_store.update_collection(new_schema)

    results = await structured_store.query("facts", [Col("fact_id") == fact.fact_id])
    assert results[0]["type"] == "notice_period"
    assert results[0]["doc_id"] == "doc-99"
    assert results[0]["confidence"] == pytest.approx(0.88)


async def test_update_collection_unknown_collection_raises(structured_store):
    from tests.stores.conftest import FACTS_SCHEMA
    with pytest.raises(KeyError):
        await structured_store.update_collection(
            CollectionSchema(
                name="does_not_exist",
                description="Test.",
                primary_fields=["fact_id"],
                fields=FACTS_SCHEMA.fields,
            )
        )


async def test_update_collection_cannot_change_primary_fields(structured_store):
    new_schema = CollectionSchema(
        name="facts",
        description="Test.",
        primary_fields=["doc_id"],  # different primary key
        fields={
            "doc_id":     FieldSchema(type=FieldType.STRING, nullable=False),
            "type":       FieldSchema(type=FieldType.STRING),
            "confidence": FieldSchema(type=FieldType.FLOAT),
        },
    )
    with pytest.raises(ValueError, match="primary_fields"):
        await structured_store.update_collection(new_schema)


async def test_save_upserts_by_composite_primary_key(structured_store):
    class VersionedFact(BaseModel):
        doc_id: str
        type: str
        value: str
        confidence: float

    schema = CollectionSchema(
        name="versioned_facts",
        description="Facts versioned by document and type, with value and confidence.",
        primary_fields=["doc_id", "type"],
        fields={
            "doc_id":     FieldSchema(type=FieldType.STRING, nullable=False),
            "type":       FieldSchema(type=FieldType.STRING, nullable=False),
            "value":      FieldSchema(type=FieldType.STRING),
            "confidence": FieldSchema(type=FieldType.FLOAT),
        },
    )
    await structured_store.create_collection(schema)

    await structured_store.save("versioned_facts", [
        VersionedFact(doc_id="doc-1", type="notice_period", value="30 days", confidence=0.8).model_dump(),
        VersionedFact(doc_id="doc-1", type="termination", value="2026-01-01", confidence=0.6).model_dump(),
    ])
    await structured_store.save("versioned_facts", [
        VersionedFact(doc_id="doc-1", type="notice_period", value="60 days", confidence=0.9).model_dump(),
    ])

    results = await structured_store.query("versioned_facts")
    assert len(results) == 2
    by_key = {(row["doc_id"], row["type"]): row for row in results}
    assert by_key[("doc-1", "notice_period")]["value"] == "60 days"
    assert by_key[("doc-1", "termination")]["value"] == "2026-01-01"


async def test_update_collection_no_change_is_noop(structured_store):
    """Calling update_collection with the identical schema leaves data intact."""
    from tests.stores.conftest import FACTS_SCHEMA
    await structured_store.save("facts", [make_fact().model_dump()])
    await structured_store.update_collection(FACTS_SCHEMA)
    assert len(await structured_store.query("facts")) == 1


# ------------------------------------------------------------------
# InMemoryStructuredStore — persist / load
# ------------------------------------------------------------------

async def test_persist_and_load_roundtrip(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    fact = make_fact()
    await store.save("facts", [fact.model_dump()])
    await store.persist(tmp_path / "store")

    restored = InMemoryStructuredStore()
    await restored.load(tmp_path / "store")
    results = await restored.query("facts")
    assert len(results) == 1
    assert results[0]["fact_id"] == fact.fact_id


async def test_persist_restores_all_collections(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA, EVENTS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.create_collection(EVENTS_SCHEMA)
    await store.save("facts", [make_fact().model_dump()])
    await store.save("events", [make_event().model_dump()])
    await store.persist(tmp_path / "store")

    restored = InMemoryStructuredStore()
    await restored.load(tmp_path / "store")
    assert len(await restored.query("facts")) == 1
    assert len(await restored.query("events")) == 1


async def test_persist_creates_nested_directory(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.save("facts", [make_fact().model_dump()])
    nested = tmp_path / "a" / "b" / "store"
    await store.persist(nested)
    assert (nested / "_schemas.json").exists()
    assert (nested / "facts.pkl").exists()


async def test_load_replaces_existing_state(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA
    store_a = InMemoryStructuredStore()
    await store_a.create_collection(FACTS_SCHEMA)
    await store_a.save("facts", [make_fact().model_dump()])
    await store_a.persist(tmp_path / "store")

    store_b = InMemoryStructuredStore()
    await store_b.create_collection(FACTS_SCHEMA)
    await store_b.save("facts", [make_fact().model_dump(), make_fact().model_dump()])  # 2 facts
    await store_b.load(tmp_path / "store")

    assert len(await store_b.query("facts")) == 1


async def test_persist_roundtrip_all_field_types(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA, EVENTS_SCHEMA, CONTRADICTIONS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.create_collection(EVENTS_SCHEMA)
    await store.create_collection(CONTRADICTIONS_SCHEMA)

    fact = make_fact(page=5, confidence=0.75)
    event = make_event(payload={"query": "test", "count": 2})
    fa, fb = make_fact(), make_fact()
    contradiction = make_contradiction(fa, fb, resolved=True, resolution_note="agreed")
    await store.save("facts", [fact.model_dump()])
    await store.save("events", [event.model_dump()])
    await store.save("contradictions", [contradiction.model_dump()])
    await store.persist(tmp_path / "store")

    restored = InMemoryStructuredStore()
    await restored.load(tmp_path / "store")

    facts = await restored.query("facts")
    assert facts[0]["page"] == 5
    assert facts[0]["confidence"] == pytest.approx(0.75)

    events = await restored.query("events")
    assert events[0]["payload"] == {"query": "test", "count": 2}

    contradictions = await restored.query("contradictions")
    assert contradictions[0]["resolved"] is True
    assert contradictions[0]["resolution_note"] == "agreed"


async def test_load_restores_schema_so_save_works_without_create(tmp_path):
    """After load, collections are usable without calling create_collection again."""
    from tests.stores.conftest import FACTS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.persist(tmp_path / "store")

    restored = InMemoryStructuredStore()
    await restored.load(tmp_path / "store")
    await restored.save("facts", [make_fact().model_dump()])
    assert len(await restored.query("facts")) == 1


async def test_persist_empty_collection_restored_as_empty(tmp_path):
    from tests.stores.conftest import FACTS_SCHEMA
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.persist(tmp_path / "store")

    restored = InMemoryStructuredStore()
    await restored.load(tmp_path / "store")
    assert await restored.query("facts") == []


# ------------------------------------------------------------------
# Field projection (fields parameter)
# ------------------------------------------------------------------

async def test_query_fields_returns_only_requested_columns(structured_store):
    await structured_store.save("facts", [make_fact(type="notice_period", confidence=0.9).model_dump()])
    results = await structured_store.query("facts", fields=["type", "doc_id"])
    assert len(results) == 1
    assert set(results[0].keys()) == {"type", "doc_id"}


async def test_query_fields_unknown_field_silently_ignored(structured_store):
    await structured_store.save("facts", [make_fact().model_dump()])
    results = await structured_store.query("facts", fields=["type", "nonexistent_field"])
    assert len(results) == 1
    assert "type" in results[0]
    assert "nonexistent_field" not in results[0]


async def test_query_fields_none_returns_all_columns(structured_store):
    fact = make_fact()
    await structured_store.save("facts", [fact.model_dump()])
    results_all = await structured_store.query("facts")
    results_none = await structured_store.query("facts", fields=None)
    assert results_all == results_none


async def test_query_fields_with_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period", doc_id="doc-1", confidence=0.9).model_dump(),
        make_fact(type="termination_date", doc_id="doc-2", confidence=0.5).model_dump(),
    ])
    results = await structured_store.query(
        "facts",
        filters=[Col("type") == "notice_period"],
        fields=["doc_id", "confidence"],
    )
    assert len(results) == 1
    assert results[0] == {"doc_id": "doc-1", "confidence": pytest.approx(0.9)}


async def test_query_fields_with_json_filter_strips_json_col_from_result(memory_store):
    """When a JSON column is used only for filtering it must not appear in results."""
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query"}).model_dump(),
        make_event(session_id="s2", payload={"kind": "ingest"}).model_dump(),
    ])
    results = await memory_store.query(
        "events",
        filters=[Col("payload.kind") == "query"],
        fields=["session_id"],
    )
    assert len(results) == 1
    assert set(results[0].keys()) == {"session_id"}
    assert results[0]["session_id"] == "s1"


# ------------------------------------------------------------------
# register_schema — schema registry without DDL
# ------------------------------------------------------------------

async def test_register_schema_enables_query_without_create_collection(structured_store):
    """register_schema populates the in-memory registry so that an already-existing
    collection can be queried without calling create_collection first (restart case)."""
    from tests.stores.conftest import FACTS_SCHEMA

    # Save a fact through the normal path so the underlying table exists.
    await structured_store.save("facts", [make_fact(type="notice_period").model_dump()])

    # Simulate restart: wipe _schemas but leave backing store intact.
    structured_store._schemas.clear()

    # register_schema (no DDL) should re-hydrate the registry.
    structured_store.register_schema(FACTS_SCHEMA)
    results = await structured_store.query("facts")
    assert len(results) == 1
    assert results[0]["type"] == "notice_period"


async def test_register_schema_unknown_collection_still_raises_on_save(structured_store):
    """register_schema for a collection whose table does not exist lets save raise
    a DB-level error rather than a registry error — the contract is that the table
    must already exist when register_schema is used."""
    from cogbase.stores.structured.memory import InMemoryStructuredStore
    if not isinstance(structured_store, InMemoryStructuredStore):
        return  # only testable on in-memory where we can control table existence
    # Registering a schema for a non-existent collection is legal (no DDL error),
    # but save will fail because the frame doesn't exist.
    new_schema = CollectionSchema(
        name="ghost",
        description="Non-existent collection.",
        primary_fields=["id"],
        fields={"id": FieldSchema(type=FieldType.STRING, nullable=False)},
    )
    structured_store.register_schema(new_schema)
    # In InMemoryStructuredStore, save uses self._frames which won't have "ghost".
    with pytest.raises(KeyError):
        await structured_store.save("ghost", [{"id": "x"}])
