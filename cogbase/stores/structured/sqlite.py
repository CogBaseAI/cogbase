"""SQLite implementation of StructuredStoreBase.

Persistence via a single SQLite file. Pass ``path=":memory:"`` for an
in-process SQLite database (useful when you want SQL semantics without a file).

Not thread-safe — wrap with a lock if sharing across threads.
"""

import json
import sqlite3
from pathlib import Path

from cogbase.core.models import Contradiction, Event, Fact
from cogbase.stores.base import StructuredStoreBase

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id     TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    value       TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    page        INTEGER,
    confidence  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id  TEXT PRIMARY KEY,
    fact_a            TEXT NOT NULL,
    fact_b            TEXT NOT NULL,
    conflict_type     TEXT NOT NULL,
    resolved          INTEGER NOT NULL DEFAULT 0,
    resolution_note   TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_facts_type     ON facts(type);
CREATE INDEX IF NOT EXISTS idx_facts_doc      ON facts(doc_id);
"""

# Columns that can be filtered directly in SQL for each table
_FACT_COLUMNS = {"fact_id", "type", "value", "doc_id", "page", "confidence"}
_CONTRADICTION_COLUMNS = {"contradiction_id", "conflict_type", "resolved", "resolution_note"}


class SQLiteStructuredStore(StructuredStoreBase):
    """SQLite-backed structured store.

    Args:
        path: Path to the SQLite database file, or ``":memory:"`` for an
              in-process database (data is lost when the connection closes).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SQLiteStructuredStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def save_facts(self, facts: list[Fact]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?)",
            [
                (f.fact_id, f.type, f.value, f.raw_text, f.doc_id, f.page, f.confidence)
                for f in facts
            ],
        )
        self._conn.commit()

    def query_facts(self, filters: dict) -> list[Fact]:
        sql_filters = {k: v for k, v in filters.items() if k in _FACT_COLUMNS}
        sql, params = _build_where("SELECT * FROM facts", sql_filters)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Fact(
                fact_id=row["fact_id"],
                type=row["type"],
                value=row["value"],
                raw_text=row["raw_text"],
                doc_id=row["doc_id"],
                page=row["page"],
                confidence=row["confidence"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def save_timeline(self, events: list[Event]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?)",
            [
                (
                    e.event_id,
                    e.session_id,
                    e.timestamp.isoformat(),
                    e.actor,
                    e.action,
                    json.dumps(e.payload),
                )
                for e in events
            ],
        )
        self._conn.commit()

    def query_timeline(self, session_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [
            Event(
                event_id=row["event_id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                actor=row["actor"],
                action=row["action"],
                payload=json.loads(row["payload"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def save_contradiction(self, c: Contradiction) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO contradictions VALUES (?,?,?,?,?,?)",
            (
                c.contradiction_id,
                c.fact_a.model_dump_json(),
                c.fact_b.model_dump_json(),
                c.conflict_type,
                int(c.resolved),
                c.resolution_note,
            ),
        )
        self._conn.commit()

    def query_contradictions(self, filters: dict) -> list[Contradiction]:
        # Translate `resolved` bool to int for SQL; filter on known columns only
        sql_filters: dict = {}
        for k, v in filters.items():
            if k not in _CONTRADICTION_COLUMNS:
                continue
            sql_filters[k] = int(v) if k == "resolved" else v

        sql, params = _build_where("SELECT * FROM contradictions", sql_filters)
        rows = self._conn.execute(sql, params).fetchall()

        results = [
            Contradiction(
                contradiction_id=row["contradiction_id"],
                fact_a=Fact.model_validate_json(row["fact_a"]),
                fact_b=Fact.model_validate_json(row["fact_b"]),
                conflict_type=row["conflict_type"],
                resolved=bool(row["resolved"]),
                resolution_note=row["resolution_note"],
            )
            for row in rows
        ]

        # doc_id is nested inside the JSON blobs — post-filter in Python
        if "doc_id" in filters:
            doc_id = filters["doc_id"]
            results = [
                c for c in results
                if c.fact_a.doc_id == doc_id or c.fact_b.doc_id == doc_id
            ]

        return results


# ------------------------------------------------------------------
# SQL helpers
# ------------------------------------------------------------------

def _build_where(base_sql: str, filters: dict) -> tuple[str, list]:
    if not filters:
        return base_sql, []
    clauses = [f"{col} = ?" for col in filters]
    return f"{base_sql} WHERE {' AND '.join(clauses)}", list(filters.values())
