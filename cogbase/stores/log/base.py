"""Abstract contract for append-only log stores.

A log store is deliberately *separate* from :class:`DocumentStoreBase`.  The
document store is overwrite-oriented — ``save`` replaces whatever was there — and
exposing an ``append`` next to it invites a caller to ``save`` over a log object
and silently truncate it.  A log object must only ever grow.  Keeping the two
contracts distinct makes that invariant a type-level guarantee rather than a
convention.

This backs the episodic-memory NDJSON log: one append-only object per session,
one JSON event per line (see ``docs/episodic-memory.md``).  The store is
line-oriented on purpose — callers append and read *records*, never raw byte
ranges — so NDJSON framing (the trailing ``\\n``) is the store's responsibility,
not the caller's.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from cogbase.stores.scope import AppScope


class LogStoreBase(abc.ABC):
    """Append-only, line-oriented log keyed by ``log_type`` + ``log_id``.

    *log_type* names a log family (e.g. ``"episodic"``) so several families can
    share one backend without key collisions; *log_id* identifies a single
    append-only stream within it (e.g. a ``session_id``).  A log family is *not*
    a structured/vector collection — it has no schema or embeddings — hence the
    distinct name.  An optional :class:`AppScope` adds the
    app/namespace/account isolation layer above *log_type*, so several
    applications can share one backend without their log families colliding —
    mirroring the document/structured/vector stores.

    All methods are async.  Implementations that call blocking I/O must wrap it
    with ``run_in_executor``.

    Example::

        log = LocalFSLogStore("/var/cogbase/logs")
        await log.append("episodic", "session-abc", ['{"seq": 0}', '{"seq": 1}'])
        lines = await log.load_lines("episodic", "session-abc", tail=1)
        await log.delete("episodic", "session-abc")
    """

    def __init__(self, scope: AppScope | None = None) -> None:
        self._scope = scope

    def _c(self, log_type: str) -> str:
        """Return the backend-internal name for *log_type* (bare name → scoped name)."""
        prefix = self._scope.prefix() if self._scope else None
        return f"{prefix}__{log_type}" if prefix else log_type

    def with_scope(self, scope: AppScope) -> "LogStoreBase":
        """Return a scoped proxy that prefixes all log-type names with *scope*."""
        from cogbase.stores.scoped import ScopedLogStore
        return ScopedLogStore(self, scope)

    @abc.abstractmethod
    async def append(self, log_type: str, log_id: str, lines: Sequence[str]) -> None:
        """Append *lines* to *log_id*, creating it if absent.

        Each element becomes one newline-terminated record; the store owns the
        framing.  The whole batch lands as a single ordered, durable append (the
        episodic writer flushes a turn's events as one call), so an empty *lines*
        is a no-op.  Never overwrites — content is only ever appended.
        """

    @abc.abstractmethod
    async def load_lines(
        self, log_type: str, log_id: str, *, tail: int | None = None
    ) -> list[str]:
        """Return the log's records, newline terminators stripped.

        Returns an empty list if the log does not exist — a stream that has not
        been written to yet is empty, not missing.  When *tail* is given, only
        the last *tail* records are returned (short-term rehydrate reads the tail
        without fetching the whole object).
        """

    @abc.abstractmethod
    async def delete(self, log_type: str, log_id: str) -> None:
        """Delete the whole log.  No-op if it does not exist.

        Whole-object delete is the only supported mutation (retention/TTL,
        per-session erasure); the append-only log is never edited in place.
        """
