"""Live tests for S3DocumentStore against a real AWS S3 bucket.

tests/deploy/test_s3_document_store.py in cogbase-service already exercises
the same store against MinIO, which proves the S3-compatible read/write
mechanics. What it cannot prove is the thing S3DocumentStore is actually
written for: real AWS credential resolution and a real bucket, with no
endpoint override (see that file's docstring — it takes no endpoint_url on
purpose). This is that other half, against the bucket named
COGBASE_TEST_S3_BUCKET (default "cogbase-test").

Marked pytest.mark.live and run with `pytest -m live`. Skips (rather than
failing) when no AWS credentials are resolvable at all — the same
"skip, don't substitute a stub" rule tests/live_setup.py documents for the
LLM/embedding live tests. Any other failure (missing bucket, wrong region,
access denied) is a real configuration problem and is left to fail loudly.

Every object this test writes lives under a per-run random prefix and is
deleted at module teardown — the bucket is a real, shared, persistent
resource, not a throwaway container.
"""

from __future__ import annotations

import os
import uuid

import boto3
import pytest
from botocore.exceptions import NoCredentialsError
from cogbase.stores.document.s3 import S3DocumentStore

BUCKET = os.environ.get("COGBASE_TEST_S3_BUCKET", "cogbase-test")
REGION = os.environ.get("COGBASE_TEST_S3_REGION", "us-west-2")


def _credentials_available() -> bool:
    try:
        boto3.client("s3", region_name=REGION).head_bucket(Bucket=BUCKET)
        return True
    except NoCredentialsError:
        return False
    except Exception:
        return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _credentials_available(), reason="No AWS credentials configured"),
]


@pytest.fixture(scope="module")
def prefix():
    return f"live-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module", autouse=True)
def cleanup(prefix):
    yield
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/"):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": objects})


@pytest.fixture
def store(prefix):
    return S3DocumentStore(bucket=BUCKET, prefix=prefix, region=REGION)


async def test_save_and_load_round_trips_text(store):
    await store.save("docs", "doc-1", "hello from the real S3 live test")
    assert await store.load("docs", "doc-1") == "hello from the real S3 live test"


async def test_load_missing_doc_raises_key_error(store):
    with pytest.raises(KeyError):
        await store.load("docs", "does-not-exist")


async def test_save_bytes_and_load_bytes_round_trips_binary(store):
    payload = b"\x00\x01\x02binary-content\xff"
    await store.save_bytes("docs", "doc-bin", payload)
    assert await store.load_bytes("docs", "doc-bin") == payload


async def test_exists_true_after_save_false_after_delete(store):
    await store.save("docs", "doc-2", "content")
    assert await store.exists("docs", "doc-2") is True
    await store.delete("docs", "doc-2")
    assert await store.exists("docs", "doc-2") is False


async def test_delete_collection_removes_every_doc_under_it(store):
    await store.save("scratch", "a", "one")
    await store.save("scratch", "b", "two")
    await store.delete_collection("scratch")
    assert await store.exists("scratch", "a") is False
    assert await store.exists("scratch", "b") is False


async def test_collections_are_isolated_by_prefix(store):
    await store.save("collection-x", "same-id", "x's content")
    await store.save("collection-y", "same-id", "y's content")
    assert await store.load("collection-x", "same-id") == "x's content"
    assert await store.load("collection-y", "same-id") == "y's content"
