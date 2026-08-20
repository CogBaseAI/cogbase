"""Live tests for S3LogStore against a real S3 Express One Zone directory bucket.

S3LogStore's append is only correct against a directory bucket: the fencing
token is WriteOffsetBytes, a header standard ("general purpose") S3 buckets
reject outright with NotImplemented — confirmed directly against
COGBASE_TEST_S3_BUCKET (cogbase-test, a standard bucket) before this test was
written. So this needs its own bucket, named by
COGBASE_TEST_S3_LOG_BUCKET (default "cogbase-log-test--usw2-az1--x-s3"), in
the availability zone its name encodes — a directory bucket's region is
pinned to that AZ's region at creation.

Marked pytest.mark.live and run with `pytest -m live`. Skips (rather than
failing) when no AWS credentials are resolvable at all — the same
"skip, don't substitute a stub" rule tests/live_setup.py documents for the
LLM/embedding live tests. Any other failure (missing bucket, wrong region,
access denied) is a real configuration problem and is left to fail loudly.

Every log this test writes lives under a per-run random prefix and is
deleted at module teardown — the bucket is a real, shared, persistent
resource, not a throwaway container.
"""

from __future__ import annotations

import os
import uuid

import boto3
import pytest
from botocore.exceptions import NoCredentialsError
from cogbase.stores.log.base import LogFenced
from cogbase.stores.log.s3 import S3LogStore

BUCKET = os.environ.get("COGBASE_TEST_S3_LOG_BUCKET", "cogbase-log-test--usw2-az1--x-s3")
REGION = os.environ.get("COGBASE_TEST_S3_LOG_REGION", "us-west-2")

LOG_TYPE = "episodic"
LOG = "session-abc"


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
    return S3LogStore(bucket=BUCKET, prefix=prefix, region=REGION)


async def test_load_lines_missing_returns_empty(store):
    assert await store.load_lines(LOG_TYPE, "never-written") == []


async def test_append_creates_then_accumulates(store):
    await store.append(LOG_TYPE, LOG, ['{"seq": 0}'])
    await store.append(LOG_TYPE, LOG, ['{"seq": 1}', '{"seq": 2}'])

    assert await store.load_lines(LOG_TYPE, LOG) == [
        '{"seq": 0}',
        '{"seq": 1}',
        '{"seq": 2}',
    ]


async def test_append_empty_batch_is_noop(store):
    await store.append(LOG_TYPE, "empty-batch", [])
    assert await store.load_lines(LOG_TYPE, "empty-batch") == []


async def test_size_reports_bytes_and_zero_when_missing(store):
    assert await store.size(LOG_TYPE, "never-written") == 0
    await store.append(LOG_TYPE, "sized", ["abc"])  # "abc\n" == 4 bytes
    assert await store.size(LOG_TYPE, "sized") == 4


async def test_append_returns_new_byte_offset(store):
    off1 = await store.append(LOG_TYPE, "offsets", ["abc"])  # 4 bytes
    assert off1 == 4
    off2 = await store.append(LOG_TYPE, "offsets", ["de", "f"])  # "de\n" + "f\n" == 5
    assert off2 == 9
    assert await store.size(LOG_TYPE, "offsets") == 9


async def test_read_since_returns_records_past_offset_with_size(store):
    off1 = await store.append(LOG_TYPE, "read-since", ["a", "b"])  # "a\nb\n" == 4 bytes
    await store.append(LOG_TYPE, "read-since", ["c", "d"])

    lines, size = await store.read_since(LOG_TYPE, "read-since", off1)
    assert lines == ["c", "d"]
    assert size == await store.size(LOG_TYPE, "read-since")


async def test_read_since_at_end_returns_nothing(store):
    end = await store.append(LOG_TYPE, "read-since-end", ["a"])
    lines, size = await store.read_since(LOG_TYPE, "read-since-end", end)
    assert lines == []
    assert size == end


async def test_conditional_append_succeeds_on_matching_offset(store):
    off = await store.append(LOG_TYPE, "conditional", ["a"], expected_offset=0)  # create
    off = await store.append(LOG_TYPE, "conditional", ["b"], expected_offset=off)
    assert await store.load_lines(LOG_TYPE, "conditional") == ["a", "b"]


async def test_conditional_append_fences_a_stale_writer(store):
    # Two writers cold-start the same session and both observe offset 0.
    first_off = await store.append(LOG_TYPE, "fenced-stale", ["live"], expected_offset=0)
    assert first_off > 0
    # The deposed writer still holds the stale offset and is rejected — its
    # colliding line never lands.
    with pytest.raises(LogFenced):
        await store.append(LOG_TYPE, "fenced-stale", ["straggler"], expected_offset=0)
    assert await store.load_lines(LOG_TYPE, "fenced-stale") == ["live"]


async def test_conditional_create_fences_second_session_owner(store):
    await store.append(LOG_TYPE, "fenced-create", ["owner"], expected_offset=0)
    # A second writer that also thinks the log is brand-new (offset 0) loses.
    with pytest.raises(LogFenced):
        await store.append(LOG_TYPE, "fenced-create", ["intruder"], expected_offset=0)


async def test_logs_isolated_across_log_types_and_ids(store):
    await store.append("app-a", LOG, ["a"])
    await store.append("app-b", LOG, ["b"])
    await store.append("app-a", "other", ["c"])

    assert await store.load_lines("app-a", LOG) == ["a"]
    assert await store.load_lines("app-b", LOG) == ["b"]
    assert await store.load_lines("app-a", "other") == ["c"]


async def test_delete_removes_log(store):
    await store.append(LOG_TYPE, "deletable", ["x"])
    await store.delete(LOG_TYPE, "deletable")
    assert await store.load_lines(LOG_TYPE, "deletable") == []


async def test_delete_missing_log_is_noop(store):
    await store.delete(LOG_TYPE, "missing")
