"""Query router — classifies a natural-language query into one of four retrieval patterns.

The router is the first component the engine calls at query time.  It decides
*how* to answer a query before touching any store or LLM.

Four patterns (from the CogBase architecture):

    A — Structured lookup: answered directly from the structured store; no LLM.
    B — Semantic search: answered from the vector store via embedding similarity.
    C — Hybrid: retrieval from both stores, then reasoning over combined results.
    D — Grounded generation: output separates [FINDINGS] from [SUPPORTING_QUOTES].

``LLMRouter`` is the only router.  It accepts any OpenAI-compatible async
client — the same interface is supported by vLLM, Ollama, and Anthropic's
compatibility endpoint, so no provider lock-in is required.

Usage::

    import openai
    from cogbase.engine.router import LLMRouter

    # Anthropic OpenAI-compatible endpoint
    client = openai.AsyncOpenAI(
        base_url="https://api.anthropic.com/v1",
        api_key="...",
    )
    router = LLMRouter(client, model="claude-opus-4-6")
    result = await router.route("does the review contradict the termination reason?")
    # result.pattern == QueryPattern.C

    # vLLM / Ollama / any OpenAI-compatible server
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    router = LLMRouter(client, model="llama3")
"""

from __future__ import annotations

import abc
import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

from cogbase.stores.filters import Filter


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class QueryPattern(str, Enum):
    """The four retrieval patterns supported by the reasoning engine.

    A — Structured lookup: answered directly from the structured store; no LLM.
    B — Semantic search: answered from the vector store via embedding similarity.
    C — Hybrid: retrieval from both stores, then reasoning over combined results.
    D — Grounded generation: output separates ``[FINDINGS]`` from ``[SUPPORTING_QUOTES]``.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class RouteResult(BaseModel):
    """Output of a query router.

    All fields except ``pattern`` and ``semantic_query`` are optional — routers
    that cannot extract structured information leave them as ``None``.
    """

    pattern: QueryPattern
    semantic_query: str
    """Cleaned query string to use for embedding or LLM prompting."""
    filters: list[Filter] | None = None
    """Structured filters parsed from the query (primarily useful for Pattern A).
    ``None`` when the router cannot extract filters from natural language.
    """
    collection: str | None = None
    """Target structured-store collection for Pattern A queries.
    ``None`` when the router cannot determine the collection from the query.
    """


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class QueryRouter(abc.ABC):
    """Abstract query router."""

    @abc.abstractmethod
    async def route(self, query: str) -> RouteResult:
        """Classify *query* and return routing instructions.

        Args:
            query: Raw natural-language query from a user or agent.

        Returns:
            ``RouteResult`` describing which pattern to use and how to execute it.
            Implementations must always populate ``pattern`` and ``semantic_query``;
            all other fields are optional.

        Raises:
            Any exception from the underlying LLM call or parse failure propagates
            to the caller — there is no silent fallback.
        """


# ---------------------------------------------------------------------------
# LLM-based router (production)
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """\
You are a query router for a document intelligence system. Classify the user's
query into exactly one of four retrieval patterns and return a JSON object.

Patterns:
  A — Structured lookup: the query asks for specific records from a structured
      store (filter by field, count, list all of a type, etc.). No LLM reasoning
      needed — pure data retrieval.
  B — Semantic search: the query asks an open-ended question best answered by
      finding similar passages in a vector store.
  C — Hybrid reasoning: the query requires retrieving from both stores and
      reasoning across the results (compare, reconcile, cross-reference, detect
      contradictions, analyse across multiple documents).
  D — Grounded generation: the query asks for a generated artefact (draft,
      summary, report, letter, etc.) that must be grounded in retrieved evidence.

Return ONLY valid JSON, no prose:
{
  "pattern": "<A|B|C|D>",
  "semantic_query": "<cleaned query for embedding or prompting>",
  "reasoning": "<one sentence explaining the classification>"
}
"""


class LLMRouter(QueryRouter):
    """Production query router backed by any OpenAI-compatible API.

    Accepts any async client that exposes ``client.chat.completions.create``
    with the OpenAI signature — this covers OpenAI, Anthropic's compatibility
    endpoint, vLLM, Ollama, and any other compatible server.

    If the LLM returns malformed JSON, the call is retried up to ``max_retries``
    times before the parse error is re-raised.  LLM API errors (network, auth,
    rate-limit) are not retried — those should be handled at a higher level.

    Args:
        client:      Async OpenAI-compatible client
                     (e.g. ``openai.AsyncOpenAI(...)``).
        model:       Model name to pass to the API (e.g. ``"claude-opus-4-6"``,
                     ``"gpt-5.4"``, ``"llama3"``).
        max_tokens:  Maximum tokens to generate. 256 is sufficient for the small
                     JSON response the router expects.
        max_retries: How many additional attempts to make when the response
                     cannot be parsed as valid JSON.  ``0`` disables retries.
                     Defaults to ``2`` (3 total attempts).

    Example::

        import openai
        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        router = LLMRouter(client, model="llama3")
        result = await router.route("compare the indemnity clauses across both contracts")
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 256,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def route(self, query: str) -> RouteResult:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query.strip()},
                ],
            )
            raw: str = response.choices[0].message.content
            try:
                return _parse_llm_response(raw, query)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]


def _parse_llm_response(raw: str, original_query: str) -> RouteResult:
    """Extract ``RouteResult`` from the LLM's JSON response.

    Strips markdown code fences if the model wraps the JSON in them.
    Raises ``ValueError`` on parse failure.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    data = json.loads(text)
    pattern = QueryPattern(data["pattern"].upper())
    raw_sq = str(data.get("semantic_query", "")).strip()
    semantic_query = raw_sq if raw_sq else original_query.strip()
    return RouteResult(pattern=pattern, semantic_query=semantic_query)

