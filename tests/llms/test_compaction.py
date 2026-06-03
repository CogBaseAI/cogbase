"""Unit tests for cogbase.llms.compaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.llms.compaction import (
    DEFAULT_CHUNK_TOKENS,
    estimate_messages_tokens,
    estimate_tokens,
    render_message,
    split_by_tokens,
    summarize_transcript,
)


def _llm(content: str = "SUMMARY") -> MagicMock:
    """A fake LLM whose complete() always returns *content*."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": content})
    return llm


# ---------------------------------------------------------------------------
# estimate_tokens / estimate_messages_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_is_four_chars_per_token():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_estimate_messages_tokens_counts_content_and_tool_call_args():
    messages = [
        {"role": "user", "content": "abcd"},  # 1 token
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "vector_search", "arguments": "abcdabcd"}}  # 2 tokens
            ],
        },
        {"role": "tool", "content": "abcdabcdabcd"},  # 3 tokens
    ]
    assert estimate_messages_tokens(messages) == 1 + 2 + 3


def test_estimate_messages_tokens_handles_missing_fields():
    assert estimate_messages_tokens([{"role": "user"}]) == 0
    assert estimate_messages_tokens([]) == 0


# ---------------------------------------------------------------------------
# render_message
# ---------------------------------------------------------------------------

def test_render_plain_message():
    assert render_message({"role": "user", "content": "hello"}) == "[user] hello"


def test_render_none_content():
    assert render_message({"role": "assistant", "content": None}) == "[assistant] "


def test_render_message_with_tool_calls_only():
    m = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "vector_search", "arguments": '{"q":1}'}}],
    }
    assert render_message(m) == '[assistant] <tool_calls: vector_search({"q":1})>'


def test_render_message_with_content_and_tool_calls():
    m = {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [
            {"function": {"name": "a", "arguments": "1"}},
            {"function": {"name": "b", "arguments": "2"}},
        ],
    }
    assert render_message(m) == "[assistant] thinking <tool_calls: a(1); b(2)>"


# ---------------------------------------------------------------------------
# split_by_tokens
# ---------------------------------------------------------------------------

def test_split_short_text_single_chunk():
    assert split_by_tokens("a\nb\nc", 100) == ["a\nb\nc"]


def test_split_respects_budget_on_line_boundaries():
    # max_tokens=2 -> max_chars=8
    text = "\n".join(["aaaa", "bbbb", "cccc", "dddd"])
    chunks = split_by_tokens(text, 2)
    assert all(len(c) <= 8 for c in chunks)
    # no non-whitespace content dropped
    assert "".join("".join(chunks).split()) == "".join(text.split())


def test_split_hard_splits_oversized_single_line():
    # A single line longer than the budget is char-split.
    max_tokens = 2  # max_chars = 8
    line = "x" * 20
    chunks = split_by_tokens(line, max_tokens)
    assert chunks == ["x" * 8, "x" * 8, "x" * 4]
    assert all(len(c) <= 8 for c in chunks)


def test_split_oversized_line_flushes_pending_buffer_first():
    text = "ab\n" + "y" * 20
    chunks = split_by_tokens(text, 2)  # max_chars=8
    assert chunks[0] == "ab"  # pending buffer flushed before the big line
    assert all(len(c) <= 8 for c in chunks)


# ---------------------------------------------------------------------------
# summarize_transcript — single call / fold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_short_transcript_single_call():
    llm = _llm("S")
    out = await summarize_transcript(llm, "short transcript")
    assert out == "S"
    assert llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_prior_summary_triggers_fold_call():
    llm = _llm("S")
    out = await summarize_transcript(llm, "short", prior_summary="PRIOR")
    assert out == "S"
    # One call to summarise the transcript, one to fold into the prior summary.
    assert llm.complete.call_count == 2
    fold_prompt = llm.complete.call_args_list[1].args[0][0]["content"]
    assert "Existing summary:\nPRIOR" in fold_prompt


@pytest.mark.asyncio
async def test_empty_llm_content_returns_empty_string():
    llm = _llm("")
    assert await summarize_transcript(llm, "short") == ""


# ---------------------------------------------------------------------------
# summarize_transcript — map-reduce over long input
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_long_transcript_maps_over_chunks():
    chunk_tokens = 5  # max_chars=20
    # 5 chunks of 4 lines each; each summary is 1 char so the merged 5-char
    # result fits the budget -> exactly one pass, one call per chunk.
    transcript = "\n".join(["line"] * 20)
    n_chunks = len(split_by_tokens(transcript, chunk_tokens))
    assert n_chunks > 1  # sanity: the input really does split

    llm = _llm("s")
    out = await summarize_transcript(llm, transcript, chunk_tokens=chunk_tokens)
    assert llm.complete.call_count == n_chunks
    assert estimate_tokens(out) <= chunk_tokens


@pytest.mark.asyncio
async def test_recurses_until_merged_fits_budget():
    chunk_tokens = 10  # max_chars=40
    # ~25 chunks on the first pass; merging 25 single-char summaries still
    # exceeds the budget, forcing one recursive reduce that then fits.
    transcript = "\n".join(["word"] * 200)
    first_pass_chunks = len(split_by_tokens(transcript, chunk_tokens))
    assert first_pass_chunks > 10

    llm = _llm("x")
    out = await summarize_transcript(llm, transcript, chunk_tokens=chunk_tokens)

    assert estimate_tokens(out) <= chunk_tokens  # terminated within budget
    assert llm.complete.call_count > first_pass_chunks  # recursion happened


# ---------------------------------------------------------------------------
# summarize_transcript — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_error_propagates():
    # Summarisation owns no degradation policy; errors surface to the caller.
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await summarize_transcript(llm, "the transcript")
