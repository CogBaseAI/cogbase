"""Engine — top-level query orchestrator.

Delegates to a ``QueryRunner`` which drives an agentic retrieval loop: the LLM
calls ``structured_lookup`` and ``vector_search`` tools as needed, then
synthesises a final answer.  Large structured result sets are returned directly
without LLM synthesis (passthrough rule).

Typical usage::

    import openai
    from cogbase.llms import OpenAILLM
    from cogbase.engine.engine import Engine
    from cogbase.engine.query_runner import QueryRunner

    llm = OpenAILLM(openai.AsyncOpenAI(api_key="..."), model="claude-sonnet-4-6")
    runner = QueryRunner(
        llm=llm,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        default_vector_collection="legal_chunks",
        structured_schemas=schemas,
    )
    engine = Engine(runner)

    async for item in engine.query_stream("what are the termination clauses?"):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print()
            print("passthrough:", item.passthrough)
            print("records:", len(item.structured_records))
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from cogbase.engine.query_runner import QueryResult, QueryRunner

logger = logging.getLogger(__name__)


class Engine:
    """Thin orchestrator wrapping a ``QueryRunner``.

    Args:
        runner: The ``QueryRunner`` that drives the agentic retrieval loop.
    """

    def __init__(self, runner: QueryRunner) -> None:
        self._runner = runner

    async def query_stream(self, text: str) -> AsyncGenerator[str | QueryResult, None]:
        """Stream the answer token-by-token, then yield a final QueryResult."""
        logger.info("engine.query_stream.start query_len=%d", len(text))
        async for item in self._runner.query_stream(text):
            yield item
