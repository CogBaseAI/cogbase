"""Local filesystem append-only log store."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Sequence

from cogbase.stores.log.base import LogFenced, LogStoreBase
from cogbase.stores.scope import AppScope

try:  # POSIX advisory locking; absent on Windows.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX hosts
    fcntl = None  # type: ignore[assignment]


class LocalFSLogStore(LogStoreBase):
    """Stores each log as a UTF-8 NDJSON file under a root directory.

    Each log lives at ``<root>/<log_type>/<log_id>``.  Appends use ``O_APPEND``,
    which is atomic for line-sized writes across processes on the host, so two
    workers that briefly both own a session (a failover window) cannot interleave
    a partial line — the backend's own write serialization is the safety net, not
    process affinity.

    When the caller passes ``expected_offset`` the append also *fences*: the file
    is locked with ``flock`` and the write proceeds only if the on-disk size
    matches, so a deposed writer is rejected with :class:`LogFenced` instead of
    appending a ``seq``-colliding straggler.  This makes single-host fencing real
    (not merely detectable after the fact); the cross-node story is the S3
    directory-bucket backend.  On a non-POSIX host without ``flock`` the size
    check still runs but is best-effort (no cross-process atomicity).

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

    async def append(
        self,
        log_type: str,
        log_id: str,
        lines: Sequence[str],
        *,
        expected_offset: int | None = None,
    ) -> int:
        path = self._path(log_type, log_id)
        if not lines:
            # No-op write; report the current size without consulting the offset.
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._size, path)
        blob = "".join(f"{line}\n" for line in lines).encode("utf-8")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._append, path, blob, expected_offset
        )

    async def size(self, log_type: str, log_id: str) -> int:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._size, path)

    async def load_lines(
        self, log_type: str, log_id: str, *, tail: int | None = None
    ) -> list[str]:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_lines, path, tail)

    async def read_since(
        self, log_type: str, log_id: str, offset: int
    ) -> tuple[list[str], int]:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_since, path, offset)

    async def delete(self, log_type: str, log_id: str) -> None:
        path = self._path(log_type, log_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._unlink, path)

    # -- sync helpers -------------------------------------------------------

    @staticmethod
    def _size(path: pathlib.Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @staticmethod
    def _append(path: pathlib.Path, blob: bytes, expected_offset: int | None) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open for append in binary; O_APPEND keeps the batch write atomic, and the
        # flock makes the size-check-then-write atomic across processes so the
        # offset can fence a deposed writer.
        with open(path, "ab") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                size = fh.seek(0, 2)  # current end == byte length
                if expected_offset is not None and size != expected_offset:
                    raise LogFenced(
                        f"append to {path.name!r} fenced: "
                        f"expected offset {expected_offset}, log is at {size}"
                    )
                fh.write(blob)
                fh.flush()
                return size + len(blob)
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_lines(path: pathlib.Path, tail: int | None) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        lines = text.splitlines()
        return lines[-tail:] if tail is not None else lines

    @staticmethod
    def _read_since(path: pathlib.Path, offset: int) -> tuple[list[str], int]:
        # Seek to the caller's watermark and read only the tail bytes.  offset is
        # a prior size, so it falls on a record boundary; an offset at/past EOF
        # (nothing new, or the log shrank) reads nothing and reports the real size
        # so the caller can detect a shrink (size < offset) and rebuild.
        try:
            with open(path, "rb") as fh:
                size = fh.seek(0, 2)
                if offset >= size:
                    return [], size
                fh.seek(offset)
                data = fh.read()
        except FileNotFoundError:
            return [], 0
        return data.decode("utf-8").splitlines(), size

    @staticmethod
    def _unlink(path: pathlib.Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
