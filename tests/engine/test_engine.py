"""Tests for cogbase.engine.engine (Engine orchestrator)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogbase.engine.engine import Engine
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.router import QueryPattern, RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_route(pattern: QueryPattern = QueryPattern.B) -> RouteResult:
    return RouteResult(
        pattern=pattern,
        semantic_query="test query",
        structured_targets=[],
    )


def _mock_retrieval(route: RouteResult) -> RetrievalResult:
    return RetrievalResult(route=route)


def _mock_generation(retrieval: RetrievalResult, answer: str = "answer") -> GenerationResult:
    return GenerationResult(
        answer=answer,
        pattern=retrieval.route.pattern,
        retrieval=retrieval,
    )


def _mock_generate_stream(generation: GenerationResult):
    """Return an async generator that yields one token then the GenerationResult."""
    async def _gen(*args, **kwargs):
        yield generation.answer
        yield generation
    return _gen


def _build_engine(
    route: RouteResult | None = None,
    answer: str = "the answer",
) -> tuple[Engine, MagicMock, MagicMock, MagicMock]:
    route = route or _mock_route()
    retrieval = _mock_retrieval(route)
    generation = _mock_generation(retrieval, answer)

    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieval)

    generator = MagicMock()
    generator.generate_stream = _mock_generate_stream(generation)

    engine = Engine(router=router, retriever=retriever, generator=generator)
    return engine, router, retriever, generator


async def _collect(stream) -> GenerationResult:
    async for item in stream:
        if isinstance(item, GenerationResult):
            return item
    raise AssertionError("stream did not yield a GenerationResult")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_returns_generation_result() -> None:
    engine, *_ = _build_engine(answer="hello")

    result = await _collect(engine.query_stream("what is the notice period?"))

    assert isinstance(result, GenerationResult)
    assert result.answer == "hello"


@pytest.mark.asyncio
async def test_engine_calls_router_with_query_text() -> None:
    engine, router, retriever, generator = _build_engine()

    await _collect(engine.query_stream("find all clauses"))

    router.route.assert_called_once_with("find all clauses")


@pytest.mark.asyncio
async def test_engine_passes_route_to_retriever() -> None:
    route = _mock_route(QueryPattern.C)
    engine, router, retriever, generator = _build_engine(route=route)

    await _collect(engine.query_stream("query"))

    retriever.retrieve.assert_called_once_with(route)


@pytest.mark.asyncio
async def test_engine_passes_query_and_retrieval_to_generator() -> None:
    route = _mock_route()
    retrieval = _mock_retrieval(route)
    generation = _mock_generation(retrieval)

    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieval)

    calls: list[tuple] = []

    async def _tracking_stream(query, ret):
        calls.append((query, ret))
        yield generation.answer
        yield generation

    generator = MagicMock()
    generator.generate_stream = _tracking_stream

    engine = Engine(router=router, retriever=retriever, generator=generator)
    await _collect(engine.query_stream("my query text"))

    assert len(calls) == 1
    assert calls[0] == ("my query text", retrieval)


@pytest.mark.asyncio
async def test_engine_preserves_pattern_in_result() -> None:
    route = _mock_route(QueryPattern.D)
    engine, *_ = _build_engine(route=route)

    result = await _collect(engine.query_stream("generate a report"))

    assert result.pattern == QueryPattern.D


@pytest.mark.asyncio
async def test_engine_router_error_propagates() -> None:
    router = MagicMock()
    router.route = AsyncMock(side_effect=RuntimeError("router failed"))
    retriever = MagicMock()
    generator = MagicMock()
    engine = Engine(router=router, retriever=retriever, generator=generator)

    with pytest.raises(RuntimeError, match="router failed"):
        await _collect(engine.query_stream("query"))


@pytest.mark.asyncio
async def test_engine_retriever_error_propagates() -> None:
    route = _mock_route()
    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(side_effect=RuntimeError("retriever failed"))

    generator = MagicMock()
    engine = Engine(router=router, retriever=retriever, generator=generator)

    with pytest.raises(RuntimeError, match="retriever failed"):
        await _collect(engine.query_stream("query"))


@pytest.mark.asyncio
async def test_engine_generator_error_propagates() -> None:
    route = _mock_route()
    retrieval = _mock_retrieval(route)

    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieval)

    async def _failing_stream(*args, **kwargs):
        raise RuntimeError("generator failed")
        yield  # make it an async generator

    generator = MagicMock()
    generator.generate_stream = _failing_stream

    engine = Engine(router=router, retriever=retriever, generator=generator)

    with pytest.raises(RuntimeError, match="generator failed"):
        await _collect(engine.query_stream("query"))
