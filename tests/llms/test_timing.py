"""Tests for context-local LLM completion timing."""

from __future__ import annotations

import asyncio

import pytest

from cogbase.llms.timing import (
    LLMTiming,
    measure_llm_call,
    record_llm_time,
    track_llm_time,
)


def test_record_outside_block_is_noop():
    """A completion recorded with no active tracker records nothing (no error)."""
    record_llm_time(0.5)  # must not raise
    with measure_llm_call():
        pass


def test_track_accumulates_calls_and_seconds():
    with track_llm_time() as timing:
        record_llm_time(0.1)
        record_llm_time(0.2)
    assert timing.calls == 2
    assert timing.seconds == pytest.approx(0.3)


def test_measure_llm_call_records_elapsed():
    with track_llm_time() as timing:
        with measure_llm_call():
            pass
    assert timing.calls == 1
    assert timing.seconds >= 0.0


def test_nested_track_isolates_inner_from_outer():
    """An inner tracker captures its own calls; the outer resumes untouched."""
    with track_llm_time() as outer:
        record_llm_time(0.1)
        with track_llm_time() as inner:
            record_llm_time(0.2)
            record_llm_time(0.3)
        record_llm_time(0.4)
    assert inner.calls == 2
    assert inner.seconds == pytest.approx(0.5)
    assert outer.calls == 2
    assert outer.seconds == pytest.approx(0.5)


def test_track_restores_previous_accumulator_on_exit():
    with track_llm_time() as outer:
        with track_llm_time():
            record_llm_time(1.0)
        # After the inner block exits, records land in the outer tracker again.
        record_llm_time(2.0)
    assert outer.calls == 1
    assert outer.seconds == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_concurrent_units_are_isolated():
    """Concurrent tasks each see only their own LLM time via contextvars."""

    async def fake_call(delay: float) -> None:
        with measure_llm_call():
            await asyncio.sleep(delay)

    async def unit(delays: list[float]) -> LLMTiming:
        with track_llm_time() as timing:
            # Concurrent completions (e.g. extraction windows) share the
            # accumulator and each add their wall time into it.
            await asyncio.gather(*(fake_call(d) for d in delays))
        return timing

    a, b = await asyncio.gather(unit([0.02, 0.02]), unit([0.01]))
    assert a.calls == 2
    assert b.calls == 1
    # a's two concurrent calls each contribute, so its summed seconds exceed b's.
    assert a.seconds > b.seconds
