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

async def test_create_collection_is_idempotent(structured_store):
    from tests.stores.conftest import FACTS_SCHEMA
    await structured_store.create_collection(FACTS_SCHEMA)


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


async def test_save_to_unknown_collection_raises(structured_store):
    with pytest.raises(KeyError, match="unknown"):
        await structured_store.save("unknown", [make_fact()])


# ------------------------------------------------------------------
# Basic save / query
# ------------------------------------------------------------------

async def test_save_and_query_no_filters(structured_store):
    fact = make_fact()
    await structured_store.save("facts", [fact])
    results = await structured_store.query("facts")
    assert len(results) == 1 and results[0]["fact_id"] == fact.fact_id


async def test_query_as_deserialises_to_model(structured_store):
    fact = make_fact()
    await structured_store.save("facts", [fact])
    results = await structured_store.query_as("facts", None, Fact)
    assert isinstance(results[0], Fact) and results[0].fact_id == fact.fact_id


async def test_save_upserts_by_id(structured_store):
    fact = make_fact(value="30 days")
    await structured_store.save("facts", [fact])
    updated = Fact(
        fact_id=fact.fact_id, type=fact.type, value="60 days",
        raw_text=fact.raw_text, doc_id=fact.doc_id, confidence=fact.confidence,
    )
    await structured_store.save("facts", [updated])
    results = await structured_store.query("facts")
    assert len(results) == 1 and results[0]["value"] == "60 days"


# ------------------------------------------------------------------
# Equality  (=  and  !=)
# ------------------------------------------------------------------

async def test_eq_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
    ])
    results = await structured_store.query("facts", [Col("type") == "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "notice_period"


async def test_ne_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
    ])
    results = await structured_store.query("facts", [Col("type") != "notice_period"])
    assert len(results) == 1 and results[0]["type"] == "termination_date"


# ------------------------------------------------------------------
# Comparisons  (<  <=  >  >=)
# ------------------------------------------------------------------

async def test_gte_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    results = await structured_store.query("facts", [Col("confidence") >= 0.8])
    assert len(results) == 2
    assert all(r["confidence"] >= 0.8 for r in results)


async def test_lt_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
    ])
    results = await structured_store.query("facts", [Col("confidence") < 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.5


async def test_lte_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    results = await structured_store.query("facts", [Col("confidence") <= 0.8])
    assert len(results) == 2


async def test_gt_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.95),
    ])
    results = await structured_store.query("facts", [Col("confidence") > 0.8])
    assert len(results) == 1 and results[0]["confidence"] == 0.95


# ------------------------------------------------------------------
# IN / NOT IN
# ------------------------------------------------------------------

async def test_in_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
        make_fact(type="salary"),
    ])
    results = await structured_store.query(
        "facts", [Col("type").in_(["notice_period", "termination_date"])]
    )
    assert len(results) == 2
    assert {r["type"] for r in results} == {"notice_period", "termination_date"}


async def test_not_in_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period"),
        make_fact(type="termination_date"),
        make_fact(type="salary"),
    ])
    results = await structured_store.query(
        "facts", [Col("type").not_in(["notice_period", "termination_date"])]
    )
    assert len(results) == 1 and results[0]["type"] == "salary"


# ------------------------------------------------------------------
# LIKE
# ------------------------------------------------------------------

async def test_like_filter_prefix(structured_store):
    await structured_store.save("facts", [
        make_fact(value="30 days"),
        make_fact(value="60 days"),
        make_fact(value="annual bonus"),
    ])
    results = await structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 2


async def test_like_filter_case_insensitive(structured_store):
    await structured_store.save("facts", [make_fact(value="Sixty Days")])
    results = await structured_store.query("facts", [Col("value").like("%days%")])
    assert len(results) == 1


# ------------------------------------------------------------------
# IS NULL / IS NOT NULL
# ------------------------------------------------------------------

async def test_is_null_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(page=None),
        make_fact(page=3),
    ])
    results = await structured_store.query("facts", [Col("page").is_null()])
    assert len(results) == 1 and results[0]["page"] is None


async def test_is_not_null_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(page=None),
        make_fact(page=3),
    ])
    results = await structured_store.query("facts", [Col("page").is_not_null()])
    assert len(results) == 1 and results[0]["page"] == 3


# ------------------------------------------------------------------
# Combined filters (AND)
# ------------------------------------------------------------------

async def test_multiple_filters_are_anded(structured_store):
    await structured_store.save("facts", [
        make_fact(type="notice_period", doc_id="doc-1", confidence=0.9),
        make_fact(type="notice_period", doc_id="doc-2", confidence=0.9),
        make_fact(type="termination_date", doc_id="doc-1", confidence=0.9),
    ])
    results = await structured_store.query("facts", [
        Col("type") == "notice_period",
        Col("doc_id") == "doc-1",
    ])
    assert len(results) == 1


async def test_no_filters_returns_all(structured_store):
    await structured_store.save("facts", [make_fact(), make_fact(), make_fact()])
    assert len(await structured_store.query("facts")) == 3


async def test_no_match_returns_empty(structured_store):
    await structured_store.save("facts", [make_fact(type="notice_period")])
    assert await structured_store.query("facts", [Col("type") == "nonexistent"]) == []


# ------------------------------------------------------------------
# JSON columns (payload, nested facts)
# ------------------------------------------------------------------

async def test_json_payload_roundtrip(structured_store):
    e = make_event(payload={"query": "notice period?", "count": 3})
    await structured_store.save("events", [e])
    results = await structured_store.query("events", [Col("session_id") == "session-1"])
    assert results[0]["payload"] == {"query": "notice period?", "count": 3}


async def test_contradiction_nested_facts_roundtrip(structured_store):
    fa = make_fact(doc_id="doc-1", value="30 days")
    fb = make_fact(doc_id="doc-2", value="60 days")
    await structured_store.save("contradictions", [make_contradiction(fa, fb)])
    result = (await structured_store.query("contradictions"))[0]
    assert result["fact_a"]["value"] == "30 days"
    assert result["fact_b"]["value"] == "60 days"


async def test_json_nested_field_eq_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query",  "count": 1}),
        make_event(session_id="s2", payload={"kind": "ingest", "count": 5}),
        make_event(session_id="s3", payload={"kind": "query",  "count": 5}),
    ])
    results = await memory_store.query("events", [Col("payload.kind") == "query"])
    assert {r["session_id"] for r in results} == {"s1", "s3"}


async def test_json_nested_field_comparison_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"count": 1}),
        make_event(session_id="s2", payload={"count": 3}),
        make_event(session_id="s3", payload={"count": 5}),
    ])
    results = await memory_store.query("events", [Col("payload.count") >= 3])
    assert {r["session_id"] for r in results} == {"s2", "s3"}


async def test_json_nested_field_combined_with_primitive_filter(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query"}),
        make_event(session_id="s2", payload={"kind": "query"}),
        make_event(session_id="s3", payload={"kind": "ingest"}),
    ])
    results = await memory_store.query("events", [
        Col("payload.kind") == "query",
        Col("session_id") == "s1",
    ])
    assert len(results) == 1 and results[0]["session_id"] == "s1"


async def test_json_nested_missing_key_treated_as_null(memory_store):
    await memory_store.save("events", [
        make_event(session_id="s1", payload={"kind": "query"}),
        make_event(session_id="s2", payload={}),
    ])
    results = await memory_store.query("events", [Col("payload.kind").is_null()])
    assert len(results) == 1 and results[0]["session_id"] == "s2"


async def test_boolean_filter(structured_store):
    fa, fb = make_fact(), make_fact()
    await structured_store.save("contradictions", [
        make_contradiction(fa, fb, resolved=False),
        make_contradiction(fa, fb, resolved=True),
    ])
    assert len(await structured_store.query("contradictions", [Col("resolved") == False])) == 1
    assert len(await structured_store.query("contradictions", [Col("resolved") == True])) == 1


# ------------------------------------------------------------------
# delete_records
# ------------------------------------------------------------------

async def test_delete_by_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(doc_id="doc-1"),
        make_fact(doc_id="doc-2"),
    ])
    await structured_store.delete_records("facts", [Col("doc_id") == "doc-1"])
    remaining = await structured_store.query("facts")
    assert len(remaining) == 1 and remaining[0]["doc_id"] == "doc-2"


async def test_delete_with_range_filter(structured_store):
    await structured_store.save("facts", [
        make_fact(confidence=0.5),
        make_fact(confidence=0.8),
        make_fact(confidence=0.95),
    ])
    await structured_store.delete_records("facts", [Col("confidence") < 0.8])
    remaining = await structured_store.query("facts")
    assert len(remaining) == 2
    assert all(r["confidence"] >= 0.8 for r in remaining)


async def test_delete_all_with_no_filters(structured_store):
    await structured_store.save("facts", [make_fact(), make_fact()])
    await structured_store.delete_records("facts")
    assert await structured_store.query("facts") == []


async def test_delete_no_match_is_noop(structured_store):
    await structured_store.save("facts", [make_fact()])
    await structured_store.delete_records("facts", [Col("type") == "nonexistent"])
    assert len(await structured_store.query("facts")) == 1


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
        id_field="flag_id",
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
        RiskFlag(flag_id="f1", severity="high",   score=0.9, description="Missing indemnity", metadata={"page": 3}),
        RiskFlag(flag_id="f2", severity="medium", score=0.6, description="Vague termination",  metadata={"page": 7}),
        RiskFlag(flag_id="f3", severity="low",    score=0.2, description="Minor formatting",   metadata={"page": 1}),
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
        id_field="fact_id",
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
    await structured_store.save("facts", [make_fact()])

    # Migrate: add the new field
    await structured_store.create_collection(schema_v2)

    # Write a record using the new schema
    new_fact = FactV2(
        fact_id="new-1", type="notice_period", value="30 days",
        raw_text="thirty days", doc_id="doc-new", confidence=0.9,
        source="upload",
    )
    await structured_store.save("facts", [new_fact])

    results = await structured_store.query("facts", [Col("fact_id") == "new-1"])
    assert results[0]["source"] == "upload"


async def test_existing_rows_get_null_for_added_field(structured_store):
    """Rows written before the migration have NULL for the new field."""
    old_fact = make_fact()
    await structured_store.save("facts", [old_fact])

    schema_v2 = CollectionSchema(
        name="facts",
        id_field="fact_id",
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
    await structured_store.save("facts", [make_fact()])

    # New schema without the 'value' field
    schema_no_value = CollectionSchema(
        name="facts",
        id_field="fact_id",
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
    await structured_store.save("facts", [make_fact()])
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
    return CollectionSchema(name="facts", id_field="fact_id", fields=base)


async def test_update_collection_add_field_existing_rows_get_null(structured_store):
    """Rows written before update_collection have None for the new field."""
    old_fact = make_fact()
    await structured_store.save("facts", [old_fact])

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
    await structured_store.save("facts", [record])

    results = await structured_store.query("facts", [Col("fact_id") == "f-new"])
    assert results[0]["source"] == "upload"


async def test_update_collection_remove_field_data_is_gone(structured_store):
    """Rows written before update_collection no longer expose the removed field."""
    await structured_store.save("facts", [make_fact(value="thirty days")])

    new_schema = _facts_schema_with(value=None)  # drop 'value'
    await structured_store.update_collection(new_schema)

    results = await structured_store.query("facts")
    assert len(results) == 1
    assert "value" not in results[0]


async def test_update_collection_add_and_remove_simultaneously(structured_store):
    """Adding and removing fields in a single update_collection call both take effect."""
    await structured_store.save("facts", [make_fact()])

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
    await structured_store.save("facts", [fact])

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
                id_field="fact_id",
                fields=FACTS_SCHEMA.fields,
            )
        )


async def test_update_collection_cannot_change_id_field(structured_store):
    new_schema = CollectionSchema(
        name="facts",
        id_field="doc_id",  # different id_field
        fields={
            "doc_id":     FieldSchema(type=FieldType.STRING, nullable=False),
            "type":       FieldSchema(type=FieldType.STRING),
            "confidence": FieldSchema(type=FieldType.FLOAT),
        },
    )
    with pytest.raises(ValueError, match="id_field"):
        await structured_store.update_collection(new_schema)


async def test_update_collection_no_change_is_noop(structured_store):
    """Calling update_collection with the identical schema leaves data intact."""
    from tests.stores.conftest import FACTS_SCHEMA
    await structured_store.save("facts", [make_fact()])
    await structured_store.update_collection(FACTS_SCHEMA)
    assert len(await structured_store.query("facts")) == 1
