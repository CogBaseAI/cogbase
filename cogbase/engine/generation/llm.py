"""LLM-backed generator — produces answers from retrieval evidence.

Accepts any OpenAI-compatible async client, matching the interface used by
``LLMRouter``.  Pattern A responses are formatted without an LLM call (the
architecture specifies "no LLM" for structured lookups).

Pattern-specific behaviour:

    A — Structured lookup: structured records are rendered as a plain-text
        table and returned directly.  No LLM call is made.
    B — Semantic search: retrieved chunks are passed as context; the LLM
        answers the query.
    C — Hybrid reasoning: both structured records and chunks are provided;
        the LLM reasons across them and returns a unified answer.
    D — Grounded generation: the LLM must produce exactly two sections —
        ``[FINDINGS]`` and ``[SUPPORTING_QUOTES]`` — separated by a blank
        line.  The generator parses these into structured fields on
        ``GenerationResult``.

Usage::

    import openai
    from cogbase.engine.generation.llm import LLMGenerator

    client = openai.AsyncOpenAI(api_key="...")
    generator = LLMGenerator(client, model="claude-sonnet-4-6")

    result = await generator.generate(query, retrieval_result)
    print(result.answer)

    # For Pattern D:
    print(result.findings)          # str — the [FINDINGS] block
    print(result.supporting_quotes) # list[str] — individual verbatim quotes
"""

from __future__ import annotations

import json
import re
from typing import Any

from cogbase.engine.generation.base import GenerationResult, GeneratorBase
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.router import QueryPattern


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PROMPT_B = """\
You are a document intelligence assistant.  Answer the user's query using ONLY
the context passages provided below.  If the context does not contain enough
information to answer, say so explicitly — do not invent facts.
"""

_PROMPT_C = """\
You are a document intelligence assistant.  Answer the user's query by reasoning
across both the structured records and the text passages provided below.  When
records and passages contradict each other, flag the discrepancy.  Do not invent
facts not present in the provided evidence.
"""

_PROMPT_D = """\
You are a document intelligence assistant producing a grounded report.  Using
ONLY the evidence provided, write a response with exactly two clearly labelled
sections:

[FINDINGS]
<Your conclusions, analysis, and synthesised answer here.>

[SUPPORTING_QUOTES]
<List each verbatim excerpt that supports your findings, one per line, prefixed
with a dash.  Quote only text that appears word-for-word in the provided passages
or record values.>

Do not add any text outside these two sections.  Do not invent quotes.
"""


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------

def _format_records(records: list[dict]) -> str:
    """Render structured records as a numbered JSON block."""
    if not records:
        return "(no structured records)"
    lines = ["Structured records:"]
    for i, rec in enumerate(records, 1):
        lines.append(f"  {i}. {json.dumps(rec, default=str)}")
    return "\n".join(lines)


def _format_chunks(chunks: list) -> str:  # chunks: list[Chunk]
    """Render vector chunks as numbered passages."""
    if not chunks:
        return "(no text passages)"
    lines = ["Text passages:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"  [{i}] (doc: {chunk.doc_id})\n  {chunk.text.strip()}")
    return "\n".join(lines)


def _format_records_as_text(records: list[dict]) -> str:
    """Render structured records as a plain human-readable answer (Pattern A — no LLM)."""
    if not records:
        return "No matching records found."
    lines = [f"Found {len(records)} record(s):"]
    for i, rec in enumerate(records, 1):
        pairs = ", ".join(f"{k}: {v}" for k, v in rec.items())
        lines.append(f"  {i}. {pairs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern D output parser
# ---------------------------------------------------------------------------

_FINDINGS_RE = re.compile(
    r"\[FINDINGS\]\s*(.*?)\s*(?=\[SUPPORTING_QUOTES\])",
    re.DOTALL | re.IGNORECASE,
)
_QUOTES_RE = re.compile(
    r"\[SUPPORTING_QUOTES\]\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_pattern_d(text: str) -> tuple[str, list[str]]:
    """Extract findings and quotes from a Pattern D LLM response.

    Returns ``(findings, quotes)`` where *findings* is the raw findings text
    and *quotes* is a list of individual quote strings (dash prefix stripped).
    Falls back gracefully when sections are absent.
    """
    findings = ""
    m = _FINDINGS_RE.search(text)
    if m:
        findings = m.group(1).strip()

    quotes: list[str] = []
    m = _QUOTES_RE.search(text)
    if m:
        raw_quotes = m.group(1).strip()
        for line in raw_quotes.splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                quotes.append(line)

    return findings, quotes


# ---------------------------------------------------------------------------
# LLMGenerator
# ---------------------------------------------------------------------------

class LLMGenerator(GeneratorBase):
    """Production generator backed by any OpenAI-compatible API.

    Accepts any async client that exposes ``client.chat.completions.create``
    with the OpenAI signature — OpenAI, Anthropic's compatibility endpoint,
    vLLM, Ollama, and any other compatible server.

    Pattern A responses are produced without an LLM call — structured records
    are formatted directly into a plain-text answer.

    Args:
        client:     Async OpenAI-compatible client.
        model:      Model name (e.g. ``"claude-sonnet-4-6"``, ``"gpt-4o"``).
        max_tokens: Maximum tokens to generate.  Defaults to 1024.

    Example::

        import openai
        from cogbase.engine.generation.llm import LLMGenerator

        client = openai.AsyncOpenAI(api_key="sk-...")
        generator = LLMGenerator(client, model="gpt-4o")
        result = await generator.generate("summarise the key clauses", retrieval)
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def generate(self, query: str, retrieval: RetrievalResult) -> GenerationResult:
        pattern = retrieval.route.pattern

        if pattern == QueryPattern.A:
            return self._generate_pattern_a(retrieval)

        system_prompt, user_content = self._build_prompt(query, retrieval, pattern)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        answer: str = response.choices[0].message.content.strip()

        findings: str | None = None
        supporting_quotes: list[str] = []
        if pattern == QueryPattern.D:
            findings, supporting_quotes = _parse_pattern_d(answer)

        return GenerationResult(
            answer=answer,
            pattern=pattern,
            findings=findings,
            supporting_quotes=supporting_quotes,
            retrieval=retrieval,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_pattern_a(self, retrieval: RetrievalResult) -> GenerationResult:
        """Format structured records as a plain answer without an LLM call."""
        answer = _format_records_as_text(retrieval.structured_records)
        return GenerationResult(
            answer=answer,
            pattern=QueryPattern.A,
            retrieval=retrieval,
        )

    def _build_prompt(
        self,
        query: str,
        retrieval: RetrievalResult,
        pattern: QueryPattern,
    ) -> tuple[str, str]:
        """Return ``(system_prompt, user_message)`` for patterns B, C, and D."""
        if pattern == QueryPattern.B:
            system = _PROMPT_B
            context = _format_chunks(retrieval.chunks)
        elif pattern == QueryPattern.C:
            system = _PROMPT_C
            context = "\n\n".join([
                _format_records(retrieval.structured_records),
                _format_chunks(retrieval.chunks),
            ])
        else:  # D
            system = _PROMPT_D
            context = "\n\n".join([
                _format_records(retrieval.structured_records),
                _format_chunks(retrieval.chunks),
            ])

        user_content = f"{context}\n\nQuery: {query.strip()}"
        return system, user_content
