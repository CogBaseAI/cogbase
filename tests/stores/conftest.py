import pytest

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured import InMemoryStructuredStore, SQLiteStructuredStore

# ---------------------------------------------------------------------------
# Shared collection schemas used across tests
# ---------------------------------------------------------------------------

FACTS_SCHEMA = CollectionSchema(
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
    },
)

EVENTS_SCHEMA = CollectionSchema(
    name="events",
    id_field="event_id",
    fields={
        "event_id":   FieldSchema(type=FieldType.STRING, nullable=False),
        "session_id": FieldSchema(type=FieldType.STRING, index=True),
        "timestamp":  FieldSchema(type=FieldType.STRING),
        "actor":      FieldSchema(type=FieldType.STRING),
        "action":     FieldSchema(type=FieldType.STRING),
        "payload":    FieldSchema(type=FieldType.JSON),
    },
)

CONTRADICTIONS_SCHEMA = CollectionSchema(
    name="contradictions",
    id_field="contradiction_id",
    fields={
        "contradiction_id": FieldSchema(type=FieldType.STRING, nullable=False),
        "fact_a":           FieldSchema(type=FieldType.JSON),
        "fact_b":           FieldSchema(type=FieldType.JSON),
        "conflict_type":    FieldSchema(type=FieldType.STRING, index=True),
        "resolved":         FieldSchema(type=FieldType.BOOLEAN),
        "resolution_note":  FieldSchema(type=FieldType.STRING,  nullable=True),
    },
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["memory", "sqlite_file", "sqlite_memory"])
async def structured_store(request, tmp_path):
    if request.param == "memory":
        store = InMemoryStructuredStore()
    elif request.param == "sqlite_file":
        store = SQLiteStructuredStore(tmp_path / "test.db")
        request.addfinalizer(store.close)
    else:
        store = SQLiteStructuredStore(":memory:")
        request.addfinalizer(store.close)

    await store.create_collection(FACTS_SCHEMA)
    await store.create_collection(EVENTS_SCHEMA)
    await store.create_collection(CONTRADICTIONS_SCHEMA)
    return store


@pytest.fixture
async def memory_store():
    """InMemoryStructuredStore with all standard collections pre-created.

    Used for tests that exercise capabilities specific to the pandas backend
    (e.g. dotted-path JSON field queries) that SQLite does not support.
    """
    store = InMemoryStructuredStore()
    await store.create_collection(FACTS_SCHEMA)
    await store.create_collection(EVENTS_SCHEMA)
    await store.create_collection(CONTRADICTIONS_SCHEMA)
    return store
