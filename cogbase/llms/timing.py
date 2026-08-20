"""Context-local accounting of time and tokens spent in LLM completion calls.

Every :class:`~cogbase.llms.base.LLMBase` implementation records each
completion's wall time — and, when the provider reports it, token usage — into
the accumulator active for the current async context (if any). Callers wrap a
unit of work — a document ingest, a workflow run — in :func:`track_llm_time` to
measure how much of that unit's end-to-end time was spent waiting on the model,
and how many tokens it cost.

Accumulation is context-local (via ``contextvars``), so concurrent ingests each
see only their own LLM time even though they share one ``LLMBase``. Nested
concurrent completions (extraction windows, summarization chunks) run in child
contexts that inherit the *same* accumulator object and add into it, so their
overlapping wall times all count toward the enclosing unit's total. LLM calls
made outside any :func:`track_llm_time` block (e.g. the real-time query runner,
which surfaces tokens directly on its own response instead) find no accumulator
and record nothing.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

from cogbase.llms.base import TokenUsage


@dataclass
class LLMTiming:
    """Mutable running total of LLM completion time and tokens for one unit of work.

    ``seconds`` is summed wall time across (possibly concurrent) completions, so
    it can exceed the unit's end-to-end time when calls overlap; ``calls`` is the
    number of completions recorded. ``input_tokens``/``output_tokens`` sum
    whatever usage the provider reported — 0 for a completion that reported none,
    not a missing value, since a partial total is still the right total to bill.
    """

    seconds: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


_current: ContextVar[LLMTiming | None] = ContextVar("cogbase_llm_timing", default=None)


@contextlib.contextmanager
def track_llm_time() -> Iterator[LLMTiming]:
    """Accumulate LLM completion time within this block into a fresh accumulator.

    Sets a new :class:`LLMTiming` for the current context (isolating it from any
    enclosing tracker) and restores the previous one on exit. Read ``.seconds``
    after the block to get the total wall time spent in ``complete`` /
    ``complete_stream`` during it.
    """
    timing = LLMTiming()
    token = _current.set(timing)
    try:
        yield timing
    finally:
        _current.reset(token)


def record_llm_time(seconds: float) -> None:
    """Add one completion's wall time to the active accumulator, if any."""
    timing = _current.get()
    if timing is not None:
        timing.seconds += seconds
        timing.calls += 1


def record_llm_tokens(usage: TokenUsage | None) -> None:
    """Add one completion's token usage to the active accumulator, if any.

    Separate from :func:`record_llm_time` because usage is known only after a
    backend has parsed the provider's response — after ``measure_llm_call``'s
    ``with`` block has already recorded the call's wall time in the streaming
    case, or is about to exit it in the non-streaming case. A no-op when *usage*
    is ``None`` (the provider reported nothing) or no accumulator is active.
    """
    if usage is None:
        return
    timing = _current.get()
    if timing is not None:
        timing.input_tokens += usage.get("input_tokens", 0) or 0
        timing.output_tokens += usage.get("output_tokens", 0) or 0


@contextlib.contextmanager
def measure_llm_call() -> Iterator[None]:
    """Time the wrapped completion and record it into the active accumulator.

    A no-op when no :func:`track_llm_time` block is active. Backends wrap their
    provider call in this so every completion is accounted for uniformly.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        record_llm_time(time.perf_counter() - start)
