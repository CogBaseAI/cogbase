"""Tests for the append-only log store."""

import pytest

from cogbase.stores import LogStoreBase
from cogbase.stores.log.local_fs import LocalFSLogStore

LOG_TYPE = "episodic"
LOG = "session-abc"


def test_log_store_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LogStoreBase()  # type: ignore[abstract]


async def test_load_lines_missing_returns_empty(tmp_path):
    store = LocalFSLogStore(tmp_path)
    assert await store.load_lines(LOG_TYPE, "never-written") == []


async def test_append_creates_then_accumulates(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ['{"seq": 0}'])
    await store.append(LOG_TYPE, LOG, ['{"seq": 1}', '{"seq": 2}'])

    assert await store.load_lines(LOG_TYPE, LOG) == [
        '{"seq": 0}',
        '{"seq": 1}',
        '{"seq": 2}',
    ]


async def test_append_frames_lines_with_newlines_on_disk(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["a", "b"])
    raw = (tmp_path / LOG_TYPE / LOG).read_text()
    assert raw == "a\nb\n"


async def test_append_empty_batch_is_noop(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, [])
    # No file created, reads back empty.
    assert await store.load_lines(LOG_TYPE, LOG) == []
    assert not (tmp_path / LOG_TYPE / LOG).exists()


async def test_append_never_overwrites(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["line-1"])
    await store.append(LOG_TYPE, LOG, ["line-2"])
    assert await store.load_lines(LOG_TYPE, LOG) == ["line-1", "line-2"]


async def test_load_lines_tail_returns_last_n(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, [f"line-{i}" for i in range(5)])

    assert await store.load_lines(LOG_TYPE, LOG, tail=2) == ["line-3", "line-4"]
    # tail larger than the log returns everything, no error.
    assert len(await store.load_lines(LOG_TYPE, LOG, tail=99)) == 5


async def test_logs_isolated_across_log_types_and_ids(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append("app-a", LOG, ["a"])
    await store.append("app-b", LOG, ["b"])
    await store.append("app-a", "other", ["c"])

    assert await store.load_lines("app-a", LOG) == ["a"]
    assert await store.load_lines("app-b", LOG) == ["b"]
    assert await store.load_lines("app-a", "other") == ["c"]


async def test_delete_removes_log(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["x"])
    await store.delete(LOG_TYPE, LOG)
    assert await store.load_lines(LOG_TYPE, LOG) == []


async def test_delete_missing_log_is_noop(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.delete(LOG_TYPE, "missing")


async def test_rejects_path_escape(tmp_path):
    store = LocalFSLogStore(tmp_path)
    with pytest.raises(ValueError, match="escapes the store root"):
        await store.append(LOG_TYPE, "../../outside", ["bad"])
