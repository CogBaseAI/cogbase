"""Local filesystem append-only log store."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Sequence

from cogbase.stores.log.base import LogStoreBase
from cogbase.stores.scope import AppScope


class LocalFSLogStore(LogStoreBase):
    """Stores each log as a UTF-8 NDJSON file under a root directory.

    Each log lives at ``<root>/<log_type>/<log_id>``.  Appends use ``O_APPEND``,
    which is atomic for line-sized writes across processes on the host, so two
    workers that briefly both own a session (a failover window) cannot interleave
    a partial line — the backend's own write serialization is the safety net, not
    process affinity.

    Args:
        root: Directory that will hold all log files.  Created on first append.
        scope: Optional :class:`AppScope` prefixing every log-type name.
    """

    def __init__(
        self, root: str | pathlib.Path, scope: AppScope | None = None
    ) -> None:
        super().__init__(scope)
        self._root = pathlib.Path(root).resolve()

    def _path(self, log_type: str, log_id: str) -> pathlib.Path:
        candidate = (self._root / self._c(log_type) / log_id).resolve()
        if not str(candidate).startswith(str(self._root)):
            raise ValueError(f"log_type/log_id {log_type!r}/{log_id!r} escapes the store root")
        return candidate

    async def append(self, log_type: str, log_id: str, lines: Sequence[str]) -> None:
        if not lines:
            return
        path = self._path(log_type, log_id)
        blob = "".join(f"{line}\n" for line in lines)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._append, path, blob)

    async def load_lines(
        self, log_type: str, log_id: str, *, tail: int | None = None
    ) -> list[str]:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_lines, path, tail)

    async def delete(self, log_type: str, log_id: str) -> None:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._unlink, path)

    # -- sync helpers -------------------------------------------------------

    @staticmethod
    def _append(path: pathlib.Path, blob: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One write() of the whole batch keeps it atomic under O_APPEND.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(blob)

    @staticmethod
    def _read_lines(path: pathlib.Path, tail: int | None) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        lines = text.splitlines()
        return lines[-tail:] if tail is not None else lines

    @staticmethod
    def _unlink(path: pathlib.Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
