"""Amazon S3 (directory bucket) append-only log store.

Targets **S3 Express One Zone directory buckets**, which support *native* append
via offset-conditional ``PutObject`` — unlike standard buckets, which have no
append and would force a read-modify-write of the whole object.  Each append
passes ``WriteOffsetBytes`` equal to the current object size; S3 rejects the
write if the offset is stale, so the offset is a **fencing token**: a deposed or
stalled old writer that wakes after a handoff cannot append, and the live owner's
monotonic ``seq`` ordering is preserved (see
``docs/episodic-memory.md`` — single-writer and append safety).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from cogbase.stores.log.base import LogStoreBase
from cogbase.stores.scope import AppScope

# Stale-offset / lost-create races: re-read the size and retry against it.
_APPEND_MAX_RETRIES = 8
_OFFSET_CONFLICT_CODES = {
    "PreconditionFailed",  # IfNoneMatch=* create lost, or offset stale
    "InvalidWriteOffset",  # WriteOffsetBytes != current object size
    "ConditionalRequestConflict",
}

# TODO add unit test when start using it
class S3LogStore(LogStoreBase):
    """Stores each log as one append-only NDJSON object in a directory bucket.

    Each log is stored at ``<prefix>/<log_type>/<log_id>`` (or without the
    prefix when none is given).  Blocking boto3 calls are offloaded to a thread
    pool.

    Args:
        bucket:      Directory bucket name (e.g. ``mylogs--use1-az4--x-s3``).
        prefix:      Optional key prefix (no trailing slash needed).
        region:      AWS region.  ``None`` uses the default boto3 resolution chain.
        max_workers: Thread-pool size for blocking boto3 calls.
        scope:       Optional :class:`AppScope` prefixing every log-type name.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        max_workers: int = 8,
        scope: AppScope | None = None,
    ) -> None:
        super().__init__(scope)
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required for S3LogStore: pip install boto3") from exc

        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = boto3.client("s3", region_name=region)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _key(self, log_type: str, log_id: str) -> str:
        log_type = self._c(log_type)
        parts = [self._prefix, log_type, log_id] if self._prefix else [log_type, log_id]
        return "/".join(parts)

    async def append(self, log_type: str, log_id: str, lines: Sequence[str]) -> None:
        if not lines:
            return
        key = self._key(log_type, log_id)
        blob = "".join(f"{line}\n" for line in lines).encode("utf-8")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._append, key, blob)

    def _append(self, key: str, blob: bytes) -> None:
        from botocore.exceptions import ClientError

        for _ in range(_APPEND_MAX_RETRIES):
            size = self._object_size(key)
            try:
                if size is None:
                    # Create; IfNoneMatch=* fails if a racing writer created it
                    # first, so we never clobber an existing log with a plain put.
                    self._s3.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=blob,
                        IfNoneMatch="*",
                        ContentType="application/x-ndjson; charset=utf-8",
                    )
                else:
                    self._s3.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=blob,
                        WriteOffsetBytes=size,
                    )
                return
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code")
                if code in _OFFSET_CONFLICT_CODES:
                    continue  # lost the race; re-read size and retry
                raise
        raise RuntimeError(
            f"append to {key!r} failed after {_APPEND_MAX_RETRIES} offset-conflict retries"
        )

    def _object_size(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            resp = self._s3.head_object(Bucket=self._bucket, Key=key)
            return resp["ContentLength"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise

    async def load_lines(
        self, log_type: str, log_id: str, *, tail: int | None = None
    ) -> list[str]:
        key = self._key(log_type, log_id)
        loop = asyncio.get_event_loop()
        body = await loop.run_in_executor(self._executor, self._get, key)
        if body is None:
            return []
        lines = body.decode("utf-8").splitlines()
        return lines[-tail:] if tail is not None else lines

    def _get(self, key: str) -> bytes | None:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None

    async def delete(self, log_type: str, log_id: str) -> None:
        key = self._key(log_type, log_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._s3.delete_object(Bucket=self._bucket, Key=key),
        )
