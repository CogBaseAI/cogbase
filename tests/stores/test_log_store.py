"""Tests for the append-only log store."""

import pytest

from cogbase.stores import LogStoreBase
from cogbase.stores.log.base import LogFenced
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


async def test_read_since_returns_records_past_offset_with_size(tmp_path):
    store = LocalFSLogStore(tmp_path)
    off1 = await store.append(LOG_TYPE, LOG, ["a", "b"])  # "a\nb\n" == 4 bytes
    await store.append(LOG_TYPE, LOG, ["c", "d"])

    lines, size = await store.read_since(LOG_TYPE, LOG, off1)
    assert lines == ["c", "d"]
    assert size == await store.size(LOG_TYPE, LOG)


async def test_read_since_from_zero_returns_everything(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["a", "b", "c"])
    lines, size = await store.read_since(LOG_TYPE, LOG, 0)
    assert lines == ["a", "b", "c"]
    assert size == await store.size(LOG_TYPE, LOG)


async def test_read_since_at_end_returns_nothing(tmp_path):
    store = LocalFSLogStore(tmp_path)
    end = await store.append(LOG_TYPE, LOG, ["a"])
    lines, size = await store.read_since(LOG_TYPE, LOG, end)
    assert lines == []
    assert size == end


async def test_read_since_past_end_reports_shrink(tmp_path):
    # An offset beyond the log (it was truncated/recreated smaller) reads nothing
    # and reports the real, smaller size so the caller can detect the shrink.
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["a"])  # 2 bytes
    lines, size = await store.read_since(LOG_TYPE, LOG, 999)
    assert lines == []
    assert size == 2 and size < 999


async def test_read_since_missing_log_returns_empty_and_zero(tmp_path):
    store = LocalFSLogStore(tmp_path)
    assert await store.read_since(LOG_TYPE, "never-written", 0) == ([], 0)


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


# -- compare-and-append / fencing -------------------------------------------


async def test_size_reports_bytes_and_zero_when_missing(tmp_path):
    store = LocalFSLogStore(tmp_path)
    assert await store.size(LOG_TYPE, "never-written") == 0
    await store.append(LOG_TYPE, LOG, ["abc"])  # "abc\n" == 4 bytes
    assert await store.size(LOG_TYPE, LOG) == 4


async def test_append_returns_new_byte_offset(tmp_path):
    store = LocalFSLogStore(tmp_path)
    off1 = await store.append(LOG_TYPE, LOG, ["abc"])  # 4 bytes
    assert off1 == 4
    off2 = await store.append(LOG_TYPE, LOG, ["de", "f"])  # "de\n" + "f\n" == 5
    assert off2 == 9
    assert await store.size(LOG_TYPE, LOG) == 9


async def test_empty_append_returns_current_size_without_fencing(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["abc"])
    # An empty batch is a no-op even with a stale expected_offset.
    assert await store.append(LOG_TYPE, LOG, [], expected_offset=999) == 4


async def test_conditional_append_succeeds_on_matching_offset(tmp_path):
    store = LocalFSLogStore(tmp_path)
    off = await store.append(LOG_TYPE, LOG, ["a"], expected_offset=0)  # create
    off = await store.append(LOG_TYPE, LOG, ["b"], expected_offset=off)
    assert await store.load_lines(LOG_TYPE, LOG) == ["a", "b"]


async def test_conditional_append_fences_a_stale_writer(tmp_path):
    store = LocalFSLogStore(tmp_path)
    # Two writers cold-start the same session and both observe offset 0.
    first_off = await store.append(LOG_TYPE, LOG, ["live"], expected_offset=0)
    assert first_off > 0
    # The deposed writer still holds the stale offset and is rejected — its
    # colliding line never lands.
    with pytest.raises(LogFenced):
        await store.append(LOG_TYPE, LOG, ["straggler"], expected_offset=0)
    assert await store.load_lines(LOG_TYPE, LOG) == ["live"]


async def test_conditional_create_fences_second_session_owner(tmp_path):
    store = LocalFSLogStore(tmp_path)
    await store.append(LOG_TYPE, LOG, ["owner"], expected_offset=0)
    # A second writer that also thinks the log is brand-new (offset 0) loses.
    with pytest.raises(LogFenced):
        await store.append(LOG_TYPE, LOG, ["intruder"], expected_offset=0)
