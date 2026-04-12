"""Tests for cogbase.engine.generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.core.models import Chunk
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.generation.llm import (
    LLMGenerator,
    _format_chunks,
    _format_records,
    _format_records_as_text,
    _parse_pattern_d,
)
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.router import CollectionTarget, QueryPattern, RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route(
    pattern: QueryPattern,
    semantic_query: str = "test query",
) -> RouteResult:
    return RouteResult(
        pattern=pattern,
        semantic_query=semantic_query,
        structured_targets=[],
    )


def _retrieval(
    pattern: QueryPattern,
    records: list[dict] | None = None,
    chunks: list[Chunk] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        structured_records=records or [],
        chunks=chunks or [],
        route=_route(pattern),
    )


def _make_chunk(text: str, doc_id: str = "doc-1") -> Chunk:
    return Chunk(doc_id=doc_id, text=text)


def _mock_llm_client(answer: str) -> MagicMock:
    """Mock OpenAI-compatible async client returning *answer*."""
    message = MagicMock()
    message.content = answer
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# _format_records
# ---------------------------------------------------------------------------


def test_format_records_empty() -> None:
    assert _format_records([]) == "(no structured records)"


def test_format_records_single_entry() -> None:
    out = _format_records([{"id": "1", "value": "foo"}])
    assert "Structured records:" in out
    assert '"id": "1"' in out


def test_format_records_multiple_entries() -> None:
    out = _format_records([{"id": "1"}, {"id": "2"}])
    assert "1." in out
    assert "2." in out


# ---------------------------------------------------------------------------
# _format_chunks
# ---------------------------------------------------------------------------


def test_format_chunks_empty() -> None:
    assert _format_chunks([]) == "(no text passages)"


def test_format_chunks_includes_text_and_doc_id() -> None:
    chunk = _make_chunk("some passage", doc_id="doc-42")
    out = _format_chunks([chunk])
    assert "some passage" in out
    assert "doc-42" in out


def test_format_chunks_multiple() -> None:
    chunks = [_make_chunk(f"passage {i}") for i in range(3)]
    out = _format_chunks(chunks)
    assert "[1]" in out
    assert "[3]" in out


# ---------------------------------------------------------------------------
# _format_records_as_text
# ---------------------------------------------------------------------------


def test_format_records_as_text_no_records() -> None:
    assert _format_records_as_text([]) == "No matching records found."


def test_format_records_as_text_lists_records() -> None:
    out = _format_records_as_text([{"party": "Acme", "date": "2024-01-01"}])
    assert "Found 1 record(s):" in out
    assert "party: Acme" in out
    assert "date: 2024-01-01" in out


# ---------------------------------------------------------------------------
# _parse_pattern_d
# ---------------------------------------------------------------------------


def test_parse_pattern_d_extracts_findings_and_quotes() -> None:
    text = (
        "[FINDINGS]\n"
        "The contract was signed on 2024-01-01.\n\n"
        "[SUPPORTING_QUOTES]\n"
        "- signed on 2024-01-01\n"
        "- effective immediately\n"
    )
    findings, quotes = _parse_pattern_d(text)
    assert "2024-01-01" in findings
    assert len(quotes) == 2
    assert "signed on 2024-01-01" in quotes


def test_parse_pattern_d_strips_dash_prefix() -> None:
    text = "[FINDINGS]\nfoo\n[SUPPORTING_QUOTES]\n- quote one\n"
    _, quotes = _parse_pattern_d(text)
    assert quotes == ["quote one"]


def test_parse_pattern_d_case_insensitive() -> None:
    text = "[findings]\nfoo\n[supporting_quotes]\n- bar\n"
    findings, quotes = _parse_pattern_d(text)
    assert findings == "foo"
    assert quotes == ["bar"]


def test_parse_pattern_d_missing_sections_returns_empty() -> None:
    findings, quotes = _parse_pattern_d("just some text with no sections")
    assert findings == ""
    assert quotes == []


# ---------------------------------------------------------------------------
# LLMGenerator — Pattern A (no LLM call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_pattern_a_formats_records_without_llm() -> None:
    client = _mock_llm_client("should not be called")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(
        QueryPattern.A,
        records=[{"id": "1", "clause": "termination"}],
    )

    result = await generator.generate("list all clauses", retrieval)

    client.chat.completions.create.assert_not_called()
    assert isinstance(result, GenerationResult)
    assert result.pattern == QueryPattern.A
    assert "termination" in result.answer


@pytest.mark.asyncio
async def test_generator_pattern_a_no_records_returns_not_found() -> None:
    client = _mock_llm_client("")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.A, records=[])

    result = await generator.generate("list all clauses", retrieval)

    assert "No matching records found" in result.answer
    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# LLMGenerator — Pattern B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_pattern_b_calls_llm() -> None:
    client = _mock_llm_client("The notice period is 30 days.")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(
        QueryPattern.B,
        chunks=[_make_chunk("notice period is 30 days")],
    )

    result = await generator.generate("what is the notice period?", retrieval)

    client.chat.completions.create.assert_called_once()
    assert result.answer == "The notice period is 30 days."
    assert result.pattern == QueryPattern.B
    assert result.findings is None
    assert result.supporting_quotes == []


@pytest.mark.asyncio
async def test_generator_pattern_b_includes_chunks_in_prompt() -> None:
    client = _mock_llm_client("answer")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(
        QueryPattern.B,
        chunks=[_make_chunk("verbatim passage here", doc_id="doc-99")],
    )

    await generator.generate("query", retrieval)

    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"] if call_kwargs.kwargs else call_kwargs[1]["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "verbatim passage here" in user_content
    assert "doc-99" in user_content


# ---------------------------------------------------------------------------
# LLMGenerator — Pattern C
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_pattern_c_includes_both_records_and_chunks() -> None:
    client = _mock_llm_client("hybrid answer")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(
        QueryPattern.C,
        records=[{"id": "r1", "value": "structured-value"}],
        chunks=[_make_chunk("vector passage")],
    )

    await generator.generate("compare clauses", retrieval)

    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"] if call_kwargs.kwargs else call_kwargs[1]["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "structured-value" in user_content
    assert "vector passage" in user_content


@pytest.mark.asyncio
async def test_generator_pattern_c_result_has_no_findings() -> None:
    client = _mock_llm_client("combined answer")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.C)

    result = await generator.generate("query", retrieval)

    assert result.findings is None
    assert result.supporting_quotes == []


# ---------------------------------------------------------------------------
# LLMGenerator — Pattern D
# ---------------------------------------------------------------------------


_PATTERN_D_RESPONSE = (
    "[FINDINGS]\n"
    "The agreement was terminated on 2024-06-01 per clause 12.\n\n"
    "[SUPPORTING_QUOTES]\n"
    "- terminated on 2024-06-01\n"
    "- per clause 12 of the agreement\n"
)


@pytest.mark.asyncio
async def test_generator_pattern_d_parses_findings() -> None:
    client = _mock_llm_client(_PATTERN_D_RESPONSE)
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.D)

    result = await generator.generate("summarise termination", retrieval)

    assert result.pattern == QueryPattern.D
    assert result.findings is not None
    assert "2024-06-01" in result.findings


@pytest.mark.asyncio
async def test_generator_pattern_d_parses_supporting_quotes() -> None:
    client = _mock_llm_client(_PATTERN_D_RESPONSE)
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.D)

    result = await generator.generate("summarise termination", retrieval)

    assert len(result.supporting_quotes) == 2
    assert "terminated on 2024-06-01" in result.supporting_quotes


@pytest.mark.asyncio
async def test_generator_pattern_d_full_answer_preserved() -> None:
    client = _mock_llm_client(_PATTERN_D_RESPONSE)
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.D)

    result = await generator.generate("query", retrieval)

    assert result.answer == _PATTERN_D_RESPONSE.strip()


# ---------------------------------------------------------------------------
# GenerationResult — retrieval preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_preserves_retrieval_in_result() -> None:
    client = _mock_llm_client("answer")
    generator = LLMGenerator(client, model="test-model")
    retrieval = _retrieval(QueryPattern.B, chunks=[_make_chunk("passage")])

    result = await generator.generate("query", retrieval)

    assert result.retrieval is retrieval


# ---------------------------------------------------------------------------
# LLMGenerator — passes correct model and max_tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_passes_model_and_max_tokens() -> None:
    client = _mock_llm_client("answer")
    generator = LLMGenerator(client, model="my-model", max_tokens=512)
    retrieval = _retrieval(QueryPattern.B)

    await generator.generate("query", retrieval)

    call_kwargs = client.chat.completions.create.call_args
    kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kwargs["model"] == "my-model"
    assert kwargs["max_tokens"] == 512
