"""Abstract contract for generation executors.

A generator takes a natural-language query and a ``RetrievalResult`` (the
evidence gathered by a retriever) and produces a ``GenerationResult`` — the
final answer returned to the caller.

Pattern mapping mirrors the retrieval layer:

    A — Structured lookup: records are formatted into a direct answer without
        calling an LLM (the architecture specifies "no LLM" for Pattern A).
    B — Semantic search: chunks are used as context; LLM answers the query.
    C — Hybrid reasoning: both records and chunks are provided; LLM reasons
        across them.
    D — Grounded generation: LLM produces structured output with a ``[FINDINGS]``
        section and a ``[SUPPORTING_QUOTES]`` section listing verbatim citations.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.router import QueryPattern


class GenerationResult(BaseModel):
    """The final answer produced for a single query.

    Attributes:
        answer:            Full response text.  For Pattern D this contains both
                           the findings and the supporting quotes formatted as a
                           single string; the structured fields below are also
                           populated for programmatic access.
        pattern:           The retrieval/generation pattern that was used.
        findings:          Pattern D only — the ``[FINDINGS]`` section extracted
                           from the LLM output.  ``None`` for other patterns.
        supporting_quotes: Pattern D only — individual verbatim quote strings
                           extracted from the ``[SUPPORTING_QUOTES]`` section.
                           Empty list for other patterns.
        retrieval:         The evidence that produced this answer, preserved so
                           callers can inspect which records/chunks were used.
    """

    answer: str
    pattern: QueryPattern
    findings: str | None = None
    supporting_quotes: list[str] = []
    retrieval: RetrievalResult

    model_config = {"arbitrary_types_allowed": True}


class GeneratorBase(abc.ABC):
    """Abstract generator — turns retrieval evidence into a final answer."""

    @abc.abstractmethod
    async def generate_stream(
        self, query: str, retrieval: RetrievalResult
    ) -> AsyncGenerator[str | GenerationResult, None]:
        """Stream the answer token-by-token, then yield the final GenerationResult.

        Yields ``str`` chunks as tokens arrive, followed by one ``GenerationResult``
        as the last item.  Subclasses must implement this method.
        """

    async def generate(self, query: str, retrieval: RetrievalResult) -> GenerationResult:
        """Produce an answer, blocking until complete.

        Drains ``generate_stream`` and returns the final ``GenerationResult``.
        """
        async for item in self.generate_stream(query, retrieval):
            if isinstance(item, GenerationResult):
                return item
        raise RuntimeError("generate_stream did not yield a GenerationResult")
