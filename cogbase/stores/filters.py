"""Filter DSL for structured store queries.

Build filter expressions with ``Col``:

    Col("type") == "notice_period"
    Col("confidence") >= 0.8
    Col("doc_id").in_(["doc-1", "doc-2"])
    Col("page").is_null()

All filters in a list are combined with AND.

# Why a DSL instead of a raw SQL WHERE string?
#
# A raw WHERE string is tempting because it has zero learning curve:
#
#     store.query("facts", where="type = ? AND confidence >= ?",
#                 params=["notice_period", 0.8])
#
# But it has three problems:
#
# 1. Breaks the in-memory adapter. There is no way to evaluate a SQL string
#    against a Python dict without embedding a SQL parser or running SQLite
#    in-process. The only alternative is silently ignoring the filter, which
#    produces wrong results.
#
# 2. Not portable across backends. Raw SQL leaks dialect:
#    Postgres JSON:  WHERE metadata->>'key' = ?
#    SQLite JSON:    WHERE json_extract(metadata, '$.key') = ?
#    Any code using a raw WHERE string is silently coupled to one backend.
#
# 3. SQL injection risk if any user-supplied value is interpolated into the
#    string rather than passed as a parameter.
#
# The Col DSL solves all three: filters are backend-agnostic data structures
# that each adapter translates into whatever the backend natively supports —
# SQL WHERE clauses, Python predicates, or Postgres JSON operators. The API
# is intentionally modelled on SQLAlchemy column expressions, so developers
# familiar with SQLAlchemy Core will recognise it immediately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Op(str, Enum):
    EQ         = "="
    NE         = "!="
    LT         = "<"
    GT         = ">"
    LTE        = "<="
    GTE        = ">="
    IN         = "in"
    NOT_IN     = "not_in"
    LIKE       = "like"
    IS_NULL    = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass(frozen=True)
class Filter:
    """A single filter condition: ``field op value``."""
    field: str
    op: Op
    value: Any = None


class Col:
    """Column reference — use to build ``Filter`` expressions.

    Intentionally mirrors SQLAlchemy column expression syntax — if you've used
    SQLAlchemy Core, the operators work the same way.

    Example::

        Col("confidence") >= 0.8
        Col("type").in_(["date", "numeric"])
        Col("resolution_note").is_null()
    """

    def __init__(self, name: str) -> None:
        self.name = name

    # Comparison operators
    def __eq__(self, val: Any) -> Filter:           # type: ignore[override]
        return Filter(self.name, Op.EQ, val)

    def __ne__(self, val: Any) -> Filter:           # type: ignore[override]
        return Filter(self.name, Op.NE, val)

    def __lt__(self, val: Any) -> Filter:
        return Filter(self.name, Op.LT, val)

    def __gt__(self, val: Any) -> Filter:
        return Filter(self.name, Op.GT, val)

    def __le__(self, val: Any) -> Filter:
        return Filter(self.name, Op.LTE, val)

    def __ge__(self, val: Any) -> Filter:
        return Filter(self.name, Op.GTE, val)

    def __hash__(self) -> int:
        return hash(self.name)

    # Set membership
    def in_(self, values: list) -> Filter:
        return Filter(self.name, Op.IN, list(values))

    def not_in(self, values: list) -> Filter:
        return Filter(self.name, Op.NOT_IN, list(values))

    # Pattern match (SQL LIKE semantics: % = any sequence, _ = any char)
    def like(self, pattern: str) -> Filter:
        return Filter(self.name, Op.LIKE, pattern)

    # Null checks
    def is_null(self) -> Filter:
        return Filter(self.name, Op.IS_NULL)

    def is_not_null(self) -> Filter:
        return Filter(self.name, Op.IS_NOT_NULL)


# ---------------------------------------------------------------------------
# In-memory evaluation
# ---------------------------------------------------------------------------

def matches(record: dict, filters: list[Filter]) -> bool:
    """Return True if ``record`` satisfies all filters.

    Supports dot-notation for JSON sub-keys: ``"metadata.status"`` traverses
    into the ``metadata`` dict and extracts the ``status`` key.
    """
    for f in filters:
        if "." in f.field:
            col, key = f.field.split(".", 1)
            container = record.get(col)
            val = container.get(key) if isinstance(container, dict) else None
        else:
            val = record.get(f.field)
        if not _eval(val, f):
            return False
    return True


def _eval(val: Any, f: Filter) -> bool:
    match f.op:
        case Op.EQ:
            return val == f.value
        case Op.NE:
            return val != f.value
        case Op.LT:
            return val is not None and val < f.value
        case Op.GT:
            return val is not None and val > f.value
        case Op.LTE:
            return val is not None and val <= f.value
        case Op.GTE:
            return val is not None and val >= f.value
        case Op.IN:
            return val in f.value
        case Op.NOT_IN:
            return val not in f.value
        case Op.LIKE:
            return _like(val, f.value)
        case Op.IS_NULL:
            return val is None
        case Op.IS_NOT_NULL:
            return val is not None
    return False  # pragma: no cover


def _like(val: Any, pattern: str) -> bool:
    """SQL LIKE semantics: ``%`` = any sequence, ``_`` = any single char.

    Case-insensitive, matching SQLite's default behaviour.
    """
    if val is None:
        return False
    # Build regex char-by-char so literal parts are escaped independently
    parts = []
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return bool(re.match("^" + "".join(parts) + "$", str(val), re.IGNORECASE))


# ---------------------------------------------------------------------------
# SQL translation
# ---------------------------------------------------------------------------

def to_sql_where(
    filters: list[Filter],
    json_fields: set[str],
) -> tuple[str, list[Any]]:
    """Translate filters to a parameterized SQL WHERE clause, skipping ``json_fields``.

    Callers are responsible for handling skipped fields (e.g. post-filtering in
    Python, or translating them separately using database-specific JSON operators).

    Returns ``(where_clause, params)`` where ``where_clause`` is empty string if
    there are no applicable filters.
    """
    clauses: list[str] = []
    params: list[Any] = []

    for f in filters:
        base_field = f.field.split(".", 1)[0] if "." in f.field else f.field
        if base_field in json_fields:
            continue  # handled in Python (or by a DB-specific JSON translator)

        match f.op:
            case Op.EQ:
                clauses.append(f'"{f.field}" = ?')
                params.append(f.value)
            case Op.NE:
                clauses.append(f'"{f.field}" != ?')
                params.append(f.value)
            case Op.LT:
                clauses.append(f'"{f.field}" < ?')
                params.append(f.value)
            case Op.GT:
                clauses.append(f'"{f.field}" > ?')
                params.append(f.value)
            case Op.LTE:
                clauses.append(f'"{f.field}" <= ?')
                params.append(f.value)
            case Op.GTE:
                clauses.append(f'"{f.field}" >= ?')
                params.append(f.value)
            case Op.IN:
                placeholders = ", ".join(["?"] * len(f.value))
                clauses.append(f'"{f.field}" IN ({placeholders})')
                params.extend(f.value)
            case Op.NOT_IN:
                placeholders = ", ".join(["?"] * len(f.value))
                clauses.append(f'"{f.field}" NOT IN ({placeholders})')
                params.extend(f.value)
            case Op.LIKE:
                clauses.append(f'"{f.field}" LIKE ?')
                params.append(f.value)
            case Op.IS_NULL:
                clauses.append(f'"{f.field}" IS NULL')
            case Op.IS_NOT_NULL:
                clauses.append(f'"{f.field}" IS NOT NULL')

    return (" AND ".join(clauses), params)
