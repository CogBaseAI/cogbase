"""Transcript compaction primitives shared across tiers.

Two places compact conversation transcripts into bounded summaries:

  - ``QueryRunner.compact_messages`` (``cogbase/core/query_runner.py``) collapses
    the transient in-loop working message list when it outgrows the context
    budget, returning a fresh ``[system, summary]`` message list.
  - ``ShortTermMemory._compact_into_summary_locked`` (``cogbase/memory/short_term.py``)
    folds overflow turns into the durable per-session running summary.

Summarisation is inherently an LLM operation: ``summarize_transcript`` requires a
live ``LLMBase``. Policy for the no-LLM case and for transient LLM failures lives
at the call site, which has the context to degrade sensibly (e.g. dropping
overflow turns rather than fabricating a non-summary).

Both share the same underlying need — summarise an arbitrarily long transcript
within a token budget — so the reusable parts live here:

  - token estimation (``estimate_tokens`` / ``estimate_messages_tokens``)
  - transcript chunking (``split_by_tokens``) and rendering (``render_message``)
  - the map-reduce summariser (``summarize_transcript``)

The stateful orchestration (budget walks, lock-held state mutation, building the
final message list) stays at each call site; only the summarisation core is shared.
"""

from __future__ import annotations

import asyncio

from cogbase.llms.base import ChatMessage, LLMBase

# Max transcript tokens fed to a single summarisation call. Longer transcripts
# are split into chunks of this size, summarised in parallel, and merged.
DEFAULT_CHUNK_TOKENS = 4000

_COMPRESS_PROMPT = (
    "Compress the following conversation transcript into a concise bullet-point "
    "summary preserving all key facts, decisions, tool outputs, retrieved "
    "evidence, and conclusions. Be terse; output only the summary.\n\n"
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token)."""
    return len(text) // 4


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Rough token estimate for a message list, including tool-call arguments."""
    total = 0
    for m in messages:
        content = m.get("content")
        if content:
            total += estimate_tokens(str(content))
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(str(tc.get("function", {}).get("arguments", "")))
    return total


def render_message(m: ChatMessage) -> str:
    """Render one message as a transcript line, including any tool-call names/args."""
    content = str(m.get("content") or "")
    calls = m.get("tool_calls") or []
    if calls:
        call_str = "; ".join(
            f"{c.get('function', {}).get('name', '')}({c.get('function', {}).get('arguments', '')})"
            for c in calls
        )
        content = (f"{content} " if content else "") + f"<tool_calls: {call_str}>"
    return f"[{m.get('role', '')}] {content}"


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split *text* into chunks each within ~max_tokens, on line boundaries.

    A single line longer than the budget is hard-split by characters so no chunk
    ever exceeds the budget.
    """
    max_chars = max_tokens * 4
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        if len(line) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            for i in range(0, len(line), max_chars):
                chunks.append(line[i : i + max_chars])
            continue
        if size + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


async def summarize_transcript(
    llm: LLMBase,
    transcript: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    prior_summary: str | None = None,
) -> str:
    """Summarise *transcript* within *chunk_tokens*, returning a single summary.

    Long transcripts are split into budget-sized chunks, summarised in parallel,
    and merged — recursively, so an arbitrarily long transcript collapses to one
    bounded summary. When *prior_summary* is given, the new summary is folded into
    it (incremental running summary).

    LLM errors propagate; the caller owns any degradation policy.
    """
    new_summary = await _map_reduce(llm, transcript, chunk_tokens)
    if prior_summary:
        combined = f"Existing summary:\n{prior_summary}\n\nNew turns:\n{new_summary}"
        new_summary = await _map_reduce(llm, combined, chunk_tokens)
    return new_summary


async def _map_reduce(llm: LLMBase, transcript: str, chunk_tokens: int) -> str:
    if estimate_tokens(transcript) <= chunk_tokens:
        return await _summarize_chunk(llm, transcript)

    chunks = split_by_tokens(transcript, chunk_tokens)
    partials = await asyncio.gather(*(_summarize_chunk(llm, c) for c in chunks))
    merged = "\n".join(p for p in partials if p)

    # Merged partials can themselves exceed the budget for very long transcripts;
    # recurse until a single chunk fits.
    if estimate_tokens(merged) > chunk_tokens:
        return await _map_reduce(llm, merged, chunk_tokens)
    return merged


async def _summarize_chunk(llm: LLMBase, transcript: str) -> str:
    result = await llm.complete([{"role": "user", "content": _COMPRESS_PROMPT + transcript}])
    return result.get("content") or ""
