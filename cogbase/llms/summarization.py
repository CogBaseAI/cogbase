"""Map-reduce text summarisation primitives shared across tiers.

Several places collapse arbitrarily long text into a single bounded summary:

  - ``QueryRunner.compact_messages`` (``cogbase/core/query_runner.py``) collapses
    the transient in-loop working message list when it outgrows the context
    budget, returning a fresh ``[system, summary]`` message list.
  - ``ShortTermMemory._commit_compaction`` (``cogbase/memory/short_term.py``)
    folds overflow turns into the running summary, persisted as a
    ``session_compacted`` event in the episodic log.
  - ``IngestionPipeline`` (``cogbase/pipeline/ingestion_pipeline.py``) summarises a
    full document into the single text embedded by a ``document-embed-upsert``
    step, map-reducing documents that overflow the summariser's context window.

Summarisation is inherently an LLM operation: ``summarize_text`` requires a
live ``LLMBase``. Policy for the no-LLM case and for transient LLM failures lives
at the call site, which has the context to degrade sensibly (e.g. dropping
overflow turns rather than fabricating a non-summary).

They share the same underlying need — summarise an arbitrarily long text
within a token budget — so the reusable parts live here:

  - token estimation (``estimate_tokens`` / ``estimate_messages_tokens``)
  - text chunking (``split_by_tokens``) and rendering (``render_message``)
  - the map-reduce summariser (``summarize_text``)

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


# Compaction sizes are *fractions of a model's context window*, not absolute
# token counts: the absolute budget is derived per-deployment from the configured
# window (see ``LLMBase.context_window``) so it can never exceed it. Two ratios,
# against two (possibly different) models:
#
#   - CONTEXT_BUDGET_RATIO  — fraction of the *answering* model's window kept as
#     live working context before compaction fires. Below 1.0 to leave room for
#     the system prompt, retrieval, skills, and output.
#   - SUMMARISE_CHUNK_RATIO — fraction of the *summariser* model's window fed to
#     one summarisation call. Smaller, to leave room for the compression prompt,
#     output tokens, and token-estimation error.
#
# The gap between them (budget > chunk) is load-bearing: with the call site
# keeping ~budget/2 of newest turns, the typical overflow lands under one chunk,
# so steady-state compaction is a single LLM round trip rather than a split.
CONTEXT_BUDGET_RATIO = 0.75
SUMMARISE_CHUNK_RATIO = 0.5

# Fallback absolute chunk size for callers without an LLM in hand (window
# unknown). Equals DEFAULT_CONTEXT_WINDOW * SUMMARISE_CHUNK_RATIO.
DEFAULT_CHUNK_TOKENS = 64_000


def context_budget_tokens(llm: LLMBase, model: str | None = None) -> int:
    """Working-context budget (tokens) before compaction, for *llm*'s window."""
    return int(llm.context_window(model) * CONTEXT_BUDGET_RATIO)


def summarise_chunk_tokens(llm: LLMBase, model: str | None = None) -> int:
    """Max tokens fed to one summarisation call, for *llm*'s window."""
    return int(llm.context_window(model) * SUMMARISE_CHUNK_RATIO)

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


async def summarize_text(
    llm: LLMBase,
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    prior_summary: str | None = None,
    compress_prompt: str = WORKING_CONTEXT_PROMPT,
    model: str | None = None,
) -> str:
    """Summarise *text* within *chunk_tokens*, returning a single summary.

    Long texts are split into budget-sized chunks, summarised in parallel,
    and merged — recursively, so an arbitrarily long text collapses to one
    bounded summary. When *prior_summary* is given, the new summary is folded into
    it (incremental running summary).

    *compress_prompt* selects the summarisation emphasis (see
    :data:`WORKING_CONTEXT_PROMPT` / :data:`CONVERSATION_SUMMARY_PROMPT`); it is
    applied at every level, including the *prior_summary* fold, so the running
    summary is re-compressed with the same emphasis.

    *model* is the ``complete`` model selector for every summarisation call (e.g.
    ``"mini"`` to run on the cheaper model); ``None`` uses the default model. Size
    *chunk_tokens* against the same model's window (see
    :func:`summarise_chunk_tokens`).

    LLM errors propagate; the caller owns any degradation policy.
    """
    in_tokens = estimate_tokens(text)
    prior_tokens = estimate_tokens(prior_summary) if prior_summary else 0
    stats = _SummarizationStats()
    started = time.perf_counter()
    logger.info(
        "[summarization] start: text=%d tok, prior_summary=%d tok, chunk_budget=%d tok",
        in_tokens,
        prior_tokens,
        chunk_tokens,
    )

    new_summary = await _map_reduce(llm, text, chunk_tokens, compress_prompt, stats, model=model)
    if prior_summary:
        combined = f"Existing summary:\n{prior_summary}\n\nNew turns:\n{new_summary}"
        new_summary = await _map_reduce(
            llm, combined, chunk_tokens, compress_prompt, stats, model=model
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "[summarization] done in %.2fs: %d LLM call(s), max_depth=%d, folded_prior=%s, "
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


class _SummarizationStats:
    """Mutable accounting threaded through one ``summarize_text`` call."""

    __slots__ = ("llm_calls", "llm_seconds", "max_depth")

    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_seconds = 0.0  # summed wall time across (possibly concurrent) calls
        self.max_depth = 0


async def _map_reduce(
    llm: LLMBase,
    text: str,
    chunk_tokens: int,
    compress_prompt: str,
    stats: _SummarizationStats,
    depth: int = 0,
    *,
    model: str | None = None,
) -> str:
    stats.max_depth = max(stats.max_depth, depth)
    if estimate_tokens(text) <= chunk_tokens:
        return await _summarize_chunk(llm, text, compress_prompt, stats, model=model)

    chunks = split_by_tokens(text, chunk_tokens)
    logger.info(
        "[summarization] depth=%d: mapping %d chunk(s) (%d tok total)",
        depth,
        len(chunks),
        estimate_tokens(text),
    )
    partials = await asyncio.gather(
        *(_summarize_chunk(llm, c, compress_prompt, stats, model=model) for c in chunks)
    )
    merged = "\n".join(p for p in partials if p)

    # Merged partials can themselves exceed the budget for very long transcripts;
    # recurse until a single chunk fits. Only recurse while we are making
    # progress: each pass must yield strictly fewer tokens than it consumed.
    # Without this guard a pathologically small budget (e.g. one below a single
    # summary's minimum size) loops forever, since the reduced output never drops
    # under the budget. Strictly-decreasing token counts bound the recursion.
    merged_tokens = estimate_tokens(merged)
    if merged_tokens > chunk_tokens:
        if merged_tokens >= estimate_tokens(text):
            logger.warning(
                "[summarization] depth=%d: reduction stalled at %d tok (budget=%d); "
                "returning best-effort summary without further recursion",
                depth,
                merged_tokens,
                chunk_tokens,
            )
            return merged
        logger.info(
            "[summarization] depth=%d: merged partials still %d tok > %d budget; recursing",
            depth,
            merged_tokens,
            chunk_tokens,
        )
        return await _map_reduce(
            llm, merged, chunk_tokens, compress_prompt, stats, depth + 1, model=model
        )
    return merged


async def _summarize_chunk(
    llm: LLMBase,
    text: str,
    compress_prompt: str,
    stats: _SummarizationStats,
    *,
    model: str | None = None,
) -> str:
    started = time.perf_counter()
    # Instructions go in the system role (not concatenated into the user turn):
    # models follow system instructions more strongly, and the stable prompt
    # forms a clean message-boundary prefix for prompt caching while the variable
    # transcript stays isolated in the user turn.
    result = await llm.complete(
        [
            {"role": "system", "content": compress_prompt},
            {"role": "user", "content": text},
        ],
        model=model,
    )
    elapsed = time.perf_counter() - started
    stats.llm_calls += 1
    stats.llm_seconds += elapsed
    content = result.get("content") or ""
    logger.info(
        "[summarization] chunk summarised in %.2fs: %d -> %d tok",
        elapsed,
        estimate_tokens(text),
        estimate_tokens(content),
    )
    return content
