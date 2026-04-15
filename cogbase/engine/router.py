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
    from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

    # Reuse the same CollectionSchema you pass to the structured store —
    # LLMRouter reads field types to explain valid filter operators to the LLM.
    schema = [
        CollectionSchema(
            name="contracts", id_field="doc_id",
            fields={
                "doc_id":         FieldSchema(type=FieldType.STRING),
                "party_a":        FieldSchema(type=FieldType.STRING),
                "effective_date": FieldSchema(type=FieldType.STRING),
            },
        ),
        CollectionSchema(
            name="facts", id_field="fact_id",
            fields={
                "fact_id":    FieldSchema(type=FieldType.STRING),
                "type":       FieldSchema(type=FieldType.STRING),
                "confidence": FieldSchema(type=FieldType.FLOAT),
                "metadata":   FieldSchema(type=FieldType.JSON),
            },
        ),
    ]

    client = openai.AsyncOpenAI(
        base_url="https://api.anthropic.com/v1",
        api_key="...",
    )
    router = LLMRouter(client, model="gpt-5.4", schema=schema)
    result = await router.route("does the review contradict the termination reason?")
    # result.pattern              == QueryPattern.C
    # result.structured_targets   — list of CollectionTarget (collection + filters)

    # Without a schema the router still classifies patterns but cannot populate
    # structured_targets.
    router = LLMRouter(client, model="gpt-5.4")

    # vLLM / Ollama / any OpenAI-compatible server
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    router = LLMRouter(client, model="llama3", schema=schema)
"""

from __future__ import annotations

import abc
import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

from cogbase.stores.filters import Filter, Op
from cogbase.stores.schema import CollectionSchema, FieldType

logger = logging.getLogger(__name__)


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


class CollectionTarget(BaseModel):
    """A single structured-store collection to query, with optional filters.

    One ``RouteResult`` may contain several targets (e.g. Pattern C spanning
    both a *contracts* and a *facts* collection).

    Args:
        collection: Collection name.
        filters:    Filter expressions to apply. Empty list means "return all".
    """

    collection: str
    filters: list[Filter] = []

    model_config = {"arbitrary_types_allowed": True}


class RouteResult(BaseModel):
    """Output of a query router.

    All fields except ``pattern`` and ``semantic_query`` are optional — routers
    that cannot extract structured information leave them as empty/default.
    """

    pattern: QueryPattern
    semantic_query: str
    """Cleaned query string to use for embedding or LLM prompting."""
    structured_targets: list[CollectionTarget] = []
    """Structured-store collections to query, each with its own filter set.

    Empty when the router cannot determine targets (pattern B, or no schema
    provided).  May contain multiple entries for patterns C and D when the
    query spans more than one collection.
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
# System prompt construction
# ---------------------------------------------------------------------------

_PATTERN_DESCRIPTIONS: dict[QueryPattern, str] = {
    QueryPattern.A: (
        "A — Structured lookup: the query asks for specific records from a structured\n"
        "      store (filter by field, count, list all of a type, etc.). No LLM reasoning\n"
        "      needed — pure data retrieval."
    ),
    QueryPattern.B: (
        "B — Semantic search: the query asks an open-ended question best answered by\n"
        "      finding similar passages in a vector store."
    ),
    QueryPattern.C: (
        "C — Hybrid reasoning: the query requires retrieving from both stores and\n"
        "      reasoning across the results (compare, reconcile, cross-reference, detect\n"
        "      contradictions, analyse across multiple documents)."
    ),
    QueryPattern.D: (
        "D — Grounded generation: the query asks for a generated artefact (draft,\n"
        "      summary, report, letter, etc.) that must be grounded in retrieved evidence."
    ),
}

_ROUTER_PROMPT_TEMPLATE = """\
You are a query router for a document intelligence system. Classify the user's
query into exactly one of the retrieval patterns listed below and return a JSON object.

Patterns:
{patterns_section}

{schema_section}\
Filter operator notes:
  in, not_in             value must be a JSON array
  like                   SQL LIKE pattern (% matches any sequence)
  is_null, is_not_null   no "value" key required
  json fields            never include in filters

Return ONLY valid JSON, no prose:
{{
  "pattern": "<{pattern_ids}>",
  "semantic_query": "<cleaned query for embedding or prompting>",
  "reasoning": "<one sentence explaining the classification>",
  "structured_targets": [
    {{
      "collection": "<collection name>",
      "filters": [
        {{"field": "<field>", "op": "<op>", "value": <value>}}
      ]
    }}
  ]
}}

Rules for structured_targets:
- Populate only for patterns A, C, and D when the query targets specific collections.
- A single query may target multiple collections — include one entry per collection.
- Leave as [] for pattern B or when no collection can be determined.
- Omit the "value" key for is_null / is_not_null operators.
"""


_TYPE_OPERATORS_LEGEND = """\
Field type → valid filter operators:
  string:  =, !=, like, in, not_in, is_null, is_not_null
  integer: =, !=, <, >, <=, >=, in, not_in, is_null, is_not_null
  float:   =, !=, <, >, <=, >=, in, not_in, is_null, is_not_null
  boolean: =, !=, is_null, is_not_null
  json:    (not filterable — never include in filters)
"""


def _build_system_prompt(
    schema: list[CollectionSchema] | None,
    available_patterns: list[QueryPattern] | None = None,
) -> str:
    """Return the router system prompt, optionally injecting collection schema.

    Args:
        schema:             Structured-store collection schemas to inject.
        available_patterns: Restrict the LLM to these patterns only.  ``None``
                            means all four patterns (A, B, C, D) are available.
    """
    patterns = available_patterns or list(QueryPattern)
    patterns_section = "\n".join(f"  {_PATTERN_DESCRIPTIONS[p]}" for p in patterns)
    pattern_ids = "|".join(p.value for p in patterns)

    if schema:
        lines = [_TYPE_OPERATORS_LEGEND, "Available collections (use these names exactly):\n"]
        for c in schema:
            fields_str = ", ".join(
                f"{fname} ({fschema.type.value})"
                for fname, fschema in c.fields.items()
            )
            lines.append(f"  {c.name}: {fields_str}\n")
        lines.append("\n")
        schema_section = "".join(lines)
    else:
        schema_section = ""
    return _ROUTER_PROMPT_TEMPLATE.format(
        patterns_section=patterns_section,
        pattern_ids=pattern_ids,
        schema_section=schema_section,
    )


# ---------------------------------------------------------------------------
# LLM-based router (production)
# ---------------------------------------------------------------------------


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
        schema:              Descriptions of available structured-store collections.
                             When provided, injected into the system prompt so the
                             LLM can populate ``structured_targets`` with the right
                             collection names and field-level filters.  ``None``
                             disables schema injection.
        available_patterns:  Restrict the LLM to this subset of patterns.  Use
                             ``[QueryPattern.A, QueryPattern.D]`` for
                             structured-only deployments (no vector store) to
                             prevent the LLM from selecting B or C.  ``None``
                             (default) allows all four patterns.
        max_tokens:          Maximum tokens to generate.  512 is a safe default for
                     responses that include several structured targets with
                     filters; increase for schemas with many collections.
        max_retries: How many additional attempts to make when the response
                     cannot be parsed as valid JSON.  ``0`` disables retries.
                     Defaults to ``2`` (3 total attempts).

    Example::

        import openai
        from cogbase.engine.router import CollectionSchema, LLMRouter

        schema = [
            CollectionSchema(name="contracts", fields=["party_a", "party_b", "effective_date"]),
            CollectionSchema(name="facts",     fields=["type", "value", "confidence"]),
        ]
        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        router = LLMRouter(client, model="llama3", schema=schema)
        result = await router.route("compare the indemnity clauses across both contracts")
    """

    def __init__(
        self,
        client: Any,
        model: str,
        schema: list[CollectionSchema] | None = None,
        available_patterns: list[QueryPattern] | None = None,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._system_prompt = _build_system_prompt(schema, available_patterns)

    async def route(self, query: str) -> RouteResult:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            logger.debug("router.route.attempt attempt=%d query_len=%d", attempt + 1, len(query))
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": query.strip()},
                ],
            )
            raw: str = response.choices[0].message.content
            try:
                result = _parse_llm_response(raw, query)
                logger.info(
                    "router.route.success pattern=%s structured_targets=%d",
                    result.pattern.value,
                    len(result.structured_targets),
                )
                return result
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                logger.exception(
                    "router.route.parse_failed attempt=%d/%d",
                    attempt + 1,
                    self._max_retries + 1,
                )
        logger.error(
            "router.route.exhausted_retries attempts=%d",
            self._max_retries + 1,
            exc_info=last_exc,
        )
        raise last_exc  # type: ignore[misc]


def _parse_filter(obj: dict) -> Filter:
    """Parse a single filter dict from the LLM response into a ``Filter``."""
    field = str(obj["field"])
    op = Op(obj["op"])
    value = obj.get("value")
    return Filter(field=field, op=op, value=value)


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

    structured_targets: list[CollectionTarget] = []
    for t in data.get("structured_targets", []) or []:
        filters = [_parse_filter(f) for f in t.get("filters", []) or []]
        structured_targets.append(
            CollectionTarget(collection=str(t["collection"]), filters=filters)
        )

    return RouteResult(
        pattern=pattern,
        semantic_query=semantic_query,
        structured_targets=structured_targets,
    )
