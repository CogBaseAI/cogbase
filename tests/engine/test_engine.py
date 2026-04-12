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
    generator.generate = AsyncMock(return_value=generation)

    engine = Engine(router=router, retriever=retriever, generator=generator)
    return engine, router, retriever, generator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_returns_generation_result() -> None:
    engine, *_ = _build_engine(answer="hello")

    result = await engine.query("what is the notice period?")

    assert isinstance(result, GenerationResult)
    assert result.answer == "hello"


@pytest.mark.asyncio
async def test_engine_calls_router_with_query_text() -> None:
    engine, router, retriever, generator = _build_engine()

    await engine.query("find all clauses")

    router.route.assert_called_once_with("find all clauses")


@pytest.mark.asyncio
async def test_engine_passes_route_to_retriever() -> None:
    route = _mock_route(QueryPattern.C)
    engine, router, retriever, generator = _build_engine(route=route)

    await engine.query("query")

    retriever.retrieve.assert_called_once_with(route)


@pytest.mark.asyncio
async def test_engine_passes_query_and_retrieval_to_generator() -> None:
    route = _mock_route()
    retrieval = _mock_retrieval(route)

    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieval)

    generation = _mock_generation(retrieval)
    generator = MagicMock()
    generator.generate = AsyncMock(return_value=generation)

    engine = Engine(router=router, retriever=retriever, generator=generator)
    await engine.query("my query text")

    generator.generate.assert_called_once_with("my query text", retrieval)


@pytest.mark.asyncio
async def test_engine_preserves_pattern_in_result() -> None:
    route = _mock_route(QueryPattern.D)
    engine, *_ = _build_engine(route=route)

    result = await engine.query("generate a report")

    assert result.pattern == QueryPattern.D


@pytest.mark.asyncio
async def test_engine_router_error_propagates() -> None:
    router = MagicMock()
    router.route = AsyncMock(side_effect=RuntimeError("router failed"))
    retriever = MagicMock()
    generator = MagicMock()
    engine = Engine(router=router, retriever=retriever, generator=generator)

    with pytest.raises(RuntimeError, match="router failed"):
        await engine.query("query")


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
        await engine.query("query")


@pytest.mark.asyncio
async def test_engine_generator_error_propagates() -> None:
    route = _mock_route()
    retrieval = _mock_retrieval(route)

    router = MagicMock()
    router.route = AsyncMock(return_value=route)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieval)

    generator = MagicMock()
    generator.generate = AsyncMock(side_effect=RuntimeError("generator failed"))

    engine = Engine(router=router, retriever=retriever, generator=generator)

    with pytest.raises(RuntimeError, match="generator failed"):
        await engine.query("query")
