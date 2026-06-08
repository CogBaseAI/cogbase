"""Transcript compaction primitives shared across tiers.

Two places compact conversation transcripts into bounded summaries:

  - ``QueryRunner.compact_messages`` (``cogbase/core/query_runner.py``) collapses
    the transient in-loop working message list when it outgrows the context
    budget, returning a fresh ``[system, summary]`` message list.
  - ``ShortTermMemory._maybe_compact_locked`` (``cogbase/memory/short_term.py``)
    folds overflow turns into the running summary, persisted as a
    ``session_compacted`` event in the episodic log.

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
import logging
import time

from cogbase.llms.base import ChatMessage, LLMBase

logger = logging.getLogger(__name__)


def _is_cjk(cp: int) -> bool:
    """Whether codepoint *cp* is a script BPE tokenizers split ~1 token/char.

    CJK ideographs and extensions, kana, bopomofo, Hangul, and half/fullwidth
    forms tokenize at roughly one token per character (often more), versus the
    ~4 chars/token of Latin text. Counting these separately keeps token
    estimates from undercounting Chinese/Japanese/Korean transcripts by ~4x,
    which would let a "within budget" chunk overflow the real context window.
    """
    return (
        0x3000 <= cp <= 0x9FFF       # CJK symbols/punct, kana, bopomofo, CJK Unified + ext A
        or 0xAC00 <= cp <= 0xD7A3    # Hangul syllables
        or 0xF900 <= cp <= 0xFAFF    # CJK compatibility ideographs
        or 0xFF00 <= cp <= 0xFFEF    # half/fullwidth forms
        or 0x20000 <= cp <= 0x2FA1F  # CJK extensions B-F and compatibility supplement
    )


# Max transcript tokens fed to a single summarisation call. Longer transcripts
# are split into chunks of this size, summarised in parallel, and merged.
#
# This is deliberately below the full context window of modern 128k-token models:
# summarisation still needs room for the compression prompt, output tokens, token
# estimation error, and smaller compatible backends.
DEFAULT_CHUNK_TOKENS = 64_000

# Default prompt, tuned for the query-runner working-context case: the transcript
# is an in-loop agent transcript (tool calls + tool outputs) and the summary is
# re-injected so the *same* loop can keep working. Tool outputs and retrieved
# evidence are first-class here.
WORKING_CONTEXT_PROMPT = (
    "You are compressing agent conversation history so execution can continue "
    "without the full transcript. Produce a compact working-memory snapshot that "
    "preserves everything needed to keep working on the user's request.\n\n"
    "Capture:\n"
    "- User goals: active task(s), constraints, preferences, success criteria.\n"
    "- Known facts: facts from the user, facts learned from tools, and confirmed "
    "assumptions.\n"
    "- Tool results: conclusions, extracted data, identifiers, URLs, file paths, "
    "API/database/search results, and retrieved evidence. Compress verbose "
    "outputs into the factual result; drop the reasoning that produced them.\n"
    "- Progress: actions completed, decisions made, and remaining work.\n"
    "- Open items: missing information, unresolved issues, and pending tool calls.\n"
    "- References: definitions, mappings, aliases, and terminology later steps "
    "may depend on.\n\n"
    "Drop greetings and chit-chat, repeated information, intermediate reasoning, "
    "and failed exploration paths unless they explain the current state.\n\n"
    "Output using exactly these sections, in this order, as concise bullet "
    "points. Omit a section if it has no content; never invent content to fill "
    "one. When updating an existing snapshot, merge new information into the "
    "existing sections rather than restructuring them.\n\n"
    "USER_GOALS:\n"
    "KNOWN_FACTS:\n"
    "TOOL_RESULTS:\n"
    "PROGRESS:\n"
    "OPEN_ITEMS:\n"
    "REFERENCES:\n\n"
    "The conversation transcript is provided in the user message."
)

# Prompt for the short-term memory running summary: the transcript is the
# user<->assistant conversation (no tool calls) and the summary is durable
# cross-turn context for *future* queries. Emphasise user intent, preferences,
# established facts, and unresolved threads over transient task mechanics.
CONVERSATION_SUMMARY_PROMPT = (
    "Summarize the conversation for use as long-term context in a future chat.\n\n"
    "Produce a structured summary using exactly the sections below, in this "
    "order. Under each section use concise bullet points. Omit any section that "
    "has no content; never invent content to fill a section.\n\n"
    "## Goals & Tasks\n"
    "The user's goals, intentions, and tasks.\n\n"
    "## Preferences & Constraints\n"
    "Stated preferences, requirements, and constraints.\n\n"
    "## Decisions & Conclusions\n"
    "Decisions made and conclusions reached.\n\n"
    "## Key Facts\n"
    "Important facts provided by the user, likely to be referenced later.\n\n"
    "## Entities\n"
    "Key people, products, projects, files, and topics discussed.\n\n"
    "## Open Items\n"
    "Open questions, pending actions, and unresolved issues.\n\n"
    "Guidelines:\n"
    "- Prefer stable facts over transient conversation details.\n"
    "- Omit small talk, greetings, acknowledgements, and repetition.\n"
    "- Omit detailed reasoning unless it affects future decisions.\n"
    "- Omit proposals that were explicitly rejected, unless the rejection "
    "itself matters.\n"
    "- When updating an existing summary, merge new information into the "
    "existing sections rather than duplicating or restructuring them.\n\n"
    "The conversation transcript is provided in the user message."
)

# Backwards-compatible alias.
_COMPRESS_PROMPT = WORKING_CONTEXT_PROMPT


def estimate_tokens(text: str) -> int:
    """Rough token estimate.

    Latin-script text runs ~4 chars/token; CJK characters are ~1 token each and
    are counted separately so CJK transcripts are not undercounted ~4x.
    """
    cjk = sum(1 for ch in text if _is_cjk(ord(ch)))
    return cjk + (len(text) - cjk) // 4


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

    Sizing uses :func:`estimate_tokens`, so CJK text is chunked by real token
    cost rather than raw character count. A single line longer than the budget is
    hard-split so no chunk ever exceeds it.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0  # estimated tokens of the lines buffered in `current`
    for line in text.split("\n"):
        line_tokens = estimate_tokens(line)
        if line_tokens > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            chunks.extend(_hard_split_line(line, max_tokens))
            continue
        if size + line_tokens + 1 > max_tokens and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_tokens + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _hard_split_line(line: str, max_tokens: int) -> list[str]:
    """Split one over-long line into pieces each within ~max_tokens tokens."""
    pieces: list[str] = []
    start = 0
    size = 0.0
    for i, ch in enumerate(line):
        cost = 1.0 if _is_cjk(ord(ch)) else 0.25
        if size + cost > max_tokens and i > start:
            pieces.append(line[start:i])
            start = i
            size = 0.0
        size += cost
    if start < len(line):
        pieces.append(line[start:])
    return pieces


async def summarize_transcript(
    llm: LLMBase,
    transcript: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    prior_summary: str | None = None,
    compress_prompt: str = WORKING_CONTEXT_PROMPT,
) -> str:
    """Summarise *transcript* within *chunk_tokens*, returning a single summary.

    Long transcripts are split into budget-sized chunks, summarised in parallel,
    and merged — recursively, so an arbitrarily long transcript collapses to one
    bounded summary. When *prior_summary* is given, the new summary is folded into
    it (incremental running summary).

    *compress_prompt* selects the summarisation emphasis (see
    :data:`WORKING_CONTEXT_PROMPT` / :data:`CONVERSATION_SUMMARY_PROMPT`); it is
    applied at every level, including the *prior_summary* fold, so the running
    summary is re-compressed with the same emphasis.

    LLM errors propagate; the caller owns any degradation policy.
    """
    in_tokens = estimate_tokens(transcript)
    prior_tokens = estimate_tokens(prior_summary) if prior_summary else 0
    stats = _CompactionStats()
    started = time.perf_counter()
    logger.info(
        "[compaction] start: transcript=%d tok, prior_summary=%d tok, chunk_budget=%d tok",
        in_tokens,
        prior_tokens,
        chunk_tokens,
    )

    new_summary = await _map_reduce(llm, transcript, chunk_tokens, compress_prompt, stats)
    if prior_summary:
        combined = f"Existing summary:\n{prior_summary}\n\nNew turns:\n{new_summary}"
        new_summary = await _map_reduce(
            llm, combined, chunk_tokens, compress_prompt, stats
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "[compaction] done in %.2fs: %d LLM call(s), max_depth=%d, folded_prior=%s, "
        "%d -> %d tok (%.0f%% reduction), llm_time=%.2fs",
        elapsed,
        stats.llm_calls,
        stats.max_depth,
        bool(prior_summary),
        in_tokens,
        estimate_tokens(new_summary),
        (1 - estimate_tokens(new_summary) / in_tokens) * 100 if in_tokens else 0.0,
        stats.llm_seconds,
    )
    return new_summary


class _CompactionStats:
    """Mutable accounting threaded through one ``summarize_transcript`` call."""

    __slots__ = ("llm_calls", "llm_seconds", "max_depth")

    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_seconds = 0.0  # summed wall time across (possibly concurrent) calls
        self.max_depth = 0


async def _map_reduce(
    llm: LLMBase,
    transcript: str,
    chunk_tokens: int,
    compress_prompt: str,
    stats: _CompactionStats,
    depth: int = 0,
) -> str:
    stats.max_depth = max(stats.max_depth, depth)
    if estimate_tokens(transcript) <= chunk_tokens:
        return await _summarize_chunk(llm, transcript, compress_prompt, stats)

    chunks = split_by_tokens(transcript, chunk_tokens)
    logger.debug(
        "[compaction] depth=%d: mapping %d chunk(s) (%d tok total)",
        depth,
        len(chunks),
        estimate_tokens(transcript),
    )
    partials = await asyncio.gather(
        *(_summarize_chunk(llm, c, compress_prompt, stats) for c in chunks)
    )
    merged = "\n".join(p for p in partials if p)

    # Merged partials can themselves exceed the budget for very long transcripts;
    # recurse until a single chunk fits.
    if estimate_tokens(merged) > chunk_tokens:
        logger.debug(
            "[compaction] depth=%d: merged partials still %d tok > %d budget; recursing",
            depth,
            estimate_tokens(merged),
            chunk_tokens,
        )
        return await _map_reduce(
            llm, merged, chunk_tokens, compress_prompt, stats, depth + 1
        )
    return merged


async def _summarize_chunk(
    llm: LLMBase, transcript: str, compress_prompt: str, stats: _CompactionStats
) -> str:
    started = time.perf_counter()
    # Instructions go in the system role (not concatenated into the user turn):
    # models follow system instructions more strongly, and the stable prompt
    # forms a clean message-boundary prefix for prompt caching while the variable
    # transcript stays isolated in the user turn.
    result = await llm.complete(
        [
            {"role": "system", "content": compress_prompt},
            {"role": "user", "content": transcript},
        ]
    )
    elapsed = time.perf_counter() - started
    stats.llm_calls += 1
    stats.llm_seconds += elapsed
    content = result.get("content") or ""
    logger.debug(
        "[compaction] chunk summarised in %.2fs: %d -> %d tok",
        elapsed,
        estimate_tokens(transcript),
        estimate_tokens(content),
    )
    return content
