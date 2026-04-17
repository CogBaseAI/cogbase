import subprocess
import time
import uuid

import pytest

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.structured import InMemoryStructuredStore, SQLiteStructuredStore

# ---------------------------------------------------------------------------
# Shared collection schemas used across tests
# ---------------------------------------------------------------------------

FACTS_SCHEMA = CollectionSchema(
    name="facts",
    primary_fields=["fact_id"],
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
    primary_fields=["event_id"],
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
    primary_fields=["contradiction_id"],
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


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL Docker container for the test session.

    Starts ``postgres:latest``, waits until ``pg_isready`` reports the server
    is accepting connections, and stops + removes the container when the
    session ends.  Docker must be available on the host.
    """
    container_name = f"cogbase_test_pg_{uuid.uuid4().hex[:8]}"
    db_user = "test"
    db_password = "test"
    db_name = "test"

    subprocess.run(
        [
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "-e", f"POSTGRES_USER={db_user}",
            "-e", f"POSTGRES_PASSWORD={db_password}",
            "-e", f"POSTGRES_DB={db_name}",
            "-p", "0:5432",  # let Docker choose a free host port
            "postgres:latest",
        ],
        check=True,
        capture_output=True,
    )

    # Resolve the host port Docker assigned.
    port = subprocess.check_output(
        [
            "docker", "inspect", container_name,
            "--format", "{{(index (index .NetworkSettings.Ports \"5432/tcp\") 0).HostPort}}",
        ],
        text=True,
    ).strip()

    # Wait until Postgres is truly ready (up to 30 s).
    # pg_isready only checks TCP-level availability; the actual database and
    # auth may not be set up yet.  Running a real query via psql confirms that
    # the specific test database is fully accessible before we yield.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "docker", "exec", container_name,
                "psql", "-U", db_user, "-d", db_name, "-c", "SELECT 1",
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        raise RuntimeError("Postgres container did not become ready within 30 s")

    dsn = f"postgresql://{db_user}:{db_password}@localhost:{port}/{db_name}?sslmode=disable"
    yield dsn

    subprocess.run(["docker", "stop", container_name], capture_output=True)


@pytest.fixture
async def postgres_store(postgres_container):
    """PostgresStructuredStore backed by the session-scoped Docker container.

    Tables are dropped and recreated between tests so each test starts clean.
    """
    from cogbase.stores.structured.postgres import PostgresStructuredStore

    store = PostgresStructuredStore(dsn=postgres_container)
    await store.connect()

    async with store._get_pool().acquire() as conn:
        for name in ("facts", "events", "contradictions"):
            await conn.execute(f'DROP TABLE IF EXISTS "{name}"')

    await store.create_collection(FACTS_SCHEMA)
    await store.create_collection(EVENTS_SCHEMA)
    await store.create_collection(CONTRADICTIONS_SCHEMA)

    yield store
    await store.close()


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
