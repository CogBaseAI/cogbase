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

    result = await engine.query("what are the termination clauses in the contracts?")
    print(result.answer)

    # For grounded generation (Pattern D) the structured fields are also set:
    print(result.findings)
    print(result.supporting_quotes)
"""

from __future__ import annotations

from cogbase.engine.generation.base import GenerationResult, GeneratorBase
from cogbase.engine.retrieval.base import RetrieverBase
from cogbase.engine.router import QueryRouter


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

    async def query(self, text: str) -> GenerationResult:
        """Answer a natural-language query end-to-end.

        Steps:
            1. Route: classify the query and extract any structured targets.
            2. Retrieve: fetch evidence from the appropriate store(s).
            3. Generate: produce the final answer from the evidence.

        Args:
            text: The user's natural-language query.

        Returns:
            ``GenerationResult`` containing the answer and, for Pattern D,
            structured ``findings`` and ``supporting_quotes``.

        Raises:
            Any router, retriever, or generator error propagates to the caller.
        """
        route = await self._router.route(text)
        retrieval = await self._retriever.retrieve(route)
        return await self._generator.generate(text, retrieval)
