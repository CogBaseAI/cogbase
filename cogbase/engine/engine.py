"""Engine — top-level query orchestrator.

Composes the three engine components into a single ``query`` call:

    1. ``QueryRouter``    — classifies the query into a pattern and extracts
                           structured filter targets.
    2. ``HybridRetriever`` — fetches evidence from the appropriate store(s).
    3. ``GeneratorBase``  — produces the final answer from the evidence.

Typical usage::

    import openai
    from cogbase.engine.engine import Engine
    from cogbase.engine.router import LLMRouter
    from cogbase.engine.retrieval.hybrid import HybridRetriever
    from cogbase.engine.generation.llm import LLMGenerator

    client = openai.AsyncOpenAI(api_key="...")

    engine = Engine(
        router=LLMRouter(client, model="claude-sonnet-4-6", schema=app.structured_schemas),
        retriever=HybridRetriever(
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
        ),
        generator=LLMGenerator(client, model="claude-sonnet-4-6"),
    )

    async for item in engine.query_stream("what are the termination clauses in the contracts?"):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print()
            print("pattern:", item.pattern.value)
            print("findings:", item.findings)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from cogbase.engine.generation.base import GenerationResult, GeneratorBase
from cogbase.engine.retrieval.base import RetrieverBase
from cogbase.engine.router import QueryRouter

logger = logging.getLogger(__name__)


class Engine:
    """Orchestrates routing, retrieval, and generation for a single query.

    Args:
        router:    A ``QueryRouter`` implementation (e.g. ``LLMRouter``).
        retriever: A ``RetrieverBase`` implementation.  ``HybridRetriever`` is
                   recommended as it dispatches automatically based on pattern.
        generator: A ``GeneratorBase`` implementation (e.g. ``LLMGenerator``).
    """

    def __init__(
        self,
        router: QueryRouter,
        retriever: RetrieverBase,
        generator: GeneratorBase,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._generator = generator

    async def query_stream(self, text: str) -> AsyncGenerator[str | GenerationResult, None]:
        """Route and retrieve, then stream tokens followed by a final GenerationResult."""
        logger.info("engine.query_stream.start query_len=%d", len(text))
        route = await self._router.route(text)
        logger.info(
            "engine.query_stream.routed pattern=%s structured_targets=%d",
            route.pattern.value,
            len(route.structured_targets),
        )
        retrieval = await self._retriever.retrieve(route)
        logger.info(
            "engine.query_stream.retrieved pattern=%s structured_records=%d chunks=%d",
            route.pattern.value,
            len(retrieval.structured_records),
            len(retrieval.chunks),
        )
        async for chunk in self._generator.generate_stream(text, retrieval):
            yield chunk
