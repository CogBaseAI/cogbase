"""Amazon S3 document store."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from cogbase.stores.document.base import DocumentStoreBase


class S3DocumentStore(DocumentStoreBase):
    """Stores document text as UTF-8 objects in an S3 bucket.

    Each document is stored at ``<prefix>/<doc_id>`` (or just ``<doc_id>`` when
    no prefix is given).  Blocking boto3 calls are offloaded to a thread pool.

    Args:
        bucket:      S3 bucket name.
        prefix:      Optional key prefix (no trailing slash needed).
        region:      AWS region name.  ``None`` uses the default boto3 resolution
                     chain (env vars, ``~/.aws/config``, instance profile, etc.).
        max_workers: Thread-pool size for blocking boto3 calls.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        max_workers: int = 8,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required for S3DocumentStore: pip install boto3") from exc

        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = boto3.client("s3", region_name=region)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _key(self, collection: str, doc_id: str) -> str:
        parts = [self._prefix, collection, doc_id] if self._prefix else [collection, doc_id]
        return "/".join(parts)

    async def save(self, collection: str, doc_id: str, content: str) -> None:
        key = self._key(collection, doc_id)
        body = content.encode("utf-8")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="text/plain; charset=utf-8",
            ),
        )

    async def load(self, collection: str, doc_id: str) -> str:
        key = self._key(collection, doc_id)
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                self._executor,
                lambda: self._s3.get_object(Bucket=self._bucket, Key=key),
            )
            body: bytes = await loop.run_in_executor(
                self._executor, response["Body"].read
            )
            return body.decode("utf-8")
        except self._s3.exceptions.NoSuchKey:
            raise KeyError(doc_id)

    async def delete(self, collection: str, doc_id: str) -> None:
        key = self._key(collection, doc_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._s3.delete_object(Bucket=self._bucket, Key=key),
        )

    async def exists(self, collection: str, doc_id: str) -> bool:
        key = self._key(collection, doc_id)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self._s3.head_object(Bucket=self._bucket, Key=key),
            )
            return True
        except Exception as exc:
            # ClientError with 404 means the key does not exist
            if getattr(exc, "response", {}).get("Error", {}).get("Code") == "404":
                return False
            raise

