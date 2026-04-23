"""QueryRunner — agentic retrieval loop replacing the fixed route→retrieve→generate pipeline.

The runner drives an LLM agent loop where the LLM can call two store-backed
system tools:

  - structured_lookup: filter-based query against the structured store
  - vector_search:     semantic search against the vector store

The loop continues until the LLM produces a final answer (no tool calls) or
``max_rounds`` is exhausted.

Passthrough rule: if ``structured_lookup`` returns results whose JSON
serialisation exceeds ``passthrough_token_threshold`` tokens (estimated as
``len(json) // 4``), the formatted records are streamed directly and the loop
exits without an LLM synthesis step.

Usage::

    from cogbase.engine.query_runner import QueryRunner, QueryResult
    from cogbase.llms import OpenAILLM

    runner = QueryRunner(
        llm=OpenAILLM(client, model="claude-sonnet-4-6"),
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        default_vector_collection="legal_chunks",
        structured_schemas=schemas,
    )

    async for item in runner.query_stream("list all contracts expiring before 2026-01-01"):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print()
            print("passthrough:", item.passthrough)
            print("records:", len(item.structured_records))
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import ChatMessage, LLMBase, ToolDefinition
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.filters import Filter, Op
from cogbase.stores.schema import CollectionSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Final result of a QueryRunner query.

    Attributes:
        answer:              Full response text.
        structured_records:  All records returned by structured_lookup calls.
        chunks:              All chunks returned by vector_search calls.
        passthrough:         True when records were returned directly without
                             LLM synthesis (token threshold exceeded).
    """

    answer: str
    structured_records: list[dict] = []
    chunks: list[Chunk] = []
    passthrough: bool = False

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are a document intelligence assistant. Answer the user's query by retrieving
evidence from the available tools, then synthesising a final answer.

Rules:
- Call tools as needed to gather evidence before answering.
- Do not invent facts not present in retrieved evidence.
- When the evidence is sufficient, produce your final answer directly (no tool calls).
"""

_SCHEMA_HEADER = "\nAvailable structured collections (use these names and fields exactly):\n"

_FILTER_LEGEND = """
Filter operators: =, !=, <, >, <=, >=, like, in, not_in, is_null, is_not_null
  in / not_in:            value must be a JSON array
  like:                   SQL LIKE pattern (% = any sequence)
  is_null / is_not_null:  omit "value"
  JSON sub-keys:          use "field.subkey" notation
"""


def _build_system_prompt(schemas: list[CollectionSchema] | None) -> str:
    if not schemas:
        return _BASE_SYSTEM_PROMPT
    lines = [_BASE_SYSTEM_PROMPT, _SCHEMA_HEADER]
    for schema in schemas:
        fields_str = ", ".join(
            f"{name} ({fschema.type.value})"
            for name, fschema in schema.fields.items()
        )
        lines.append(f"  {schema.name}: {fields_str}")
    lines.append(_FILTER_LEGEND)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_STRUCTURED_LOOKUP_DEF: ToolDefinition = {
    "name": "structured_lookup",
    "description": (
        "Query the structured store for exact records. Use when the query asks for "
        "specific field values, counts, filtering by criteria, or listing records of a type."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": "Collection name to query.",
            },
            "filters": {
                "type": "array",
                "description": (
                    'Filter expressions (ANDed). Each item: {"field": "name", "op": "=", "value": ...}. '
                    "Omit 'value' for is_null / is_not_null."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "op": {"type": "string"},
                        "value": {},
                    },
                    "required": ["field", "op"],
                },
            },
            "fields": {
                "type": "array",
                "description": "Field names to return. Omit or leave empty for all fields.",
                "items": {"type": "string"},
            },
        },
        "required": ["collection"],
        "additionalProperties": False,
    },
}

_VECTOR_SEARCH_DEF: ToolDefinition = {
    "name": "vector_search",
    "description": (
        "Semantically search the document collection for relevant passages. Use for "
        "open-ended questions, conceptual queries, or finding similar text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 20).",
            },
            "collection": {
                "type": "string",
                "description": "Vector collection to search. Omit to use the default collection.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars ≈ 1 token."""
    return len(text) // 4


def _parse_filter(obj: dict) -> Filter:
    return Filter(field=str(obj["field"]), op=Op(obj["op"]), value=obj.get("value"))


def _format_records_as_text(records: list[dict]) -> str:
    if not records:
        return "No matching records found."
    lines = [f"Found {len(records)} record(s):"]
    for i, rec in enumerate(records, 1):
        pairs = ", ".join(f"{k}: {v}" for k, v in rec.items())
        lines.append(f"  {i}. {pairs}")
    return "\n".join(lines)


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no passages found)"
    lines = ["Passages:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"  [{i}] (doc: {chunk.doc_id})\n  {chunk.text.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QueryRunner
# ---------------------------------------------------------------------------


class QueryRunner:
    """Agentic query loop: LLM calls structured_lookup / vector_search until satisfied.

    Args:
        llm:                         LLM backend.
        structured_store:            Structured store, required for structured_lookup.
        vector_store:                Vector store, required for vector_search.
        embedder:                    Embedder, required for vector_search.
        default_vector_collection:   Collection used when vector_search omits "collection".
        structured_schemas:          Schema list injected into the system prompt so the
                                     LLM knows available collections and field types.
        passthrough_token_threshold: Estimated token count of structured_lookup results
                                     above which records are returned directly without LLM
                                     synthesis.  Defaults to 2000.
        max_rounds:                  Maximum tool-call rounds before giving up. Default 5.
    """

    def __init__(
        self,
        llm: LLMBase,
        structured_store: StructuredStoreBase | None = None,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        default_vector_collection: str | None = None,
        structured_schemas: list[CollectionSchema] | None = None,
        passthrough_token_threshold: int = 2000,
        max_rounds: int = 5,
    ) -> None:
        self._llm = llm
        self._structured_store = structured_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._default_vector_collection = default_vector_collection
        self._system_prompt = _build_system_prompt(structured_schemas)
        self._passthrough_token_threshold = passthrough_token_threshold
        self._max_rounds = max_rounds

        tool_defs: list[ToolDefinition] = []
        if structured_store is not None:
            tool_defs.append(_STRUCTURED_LOOKUP_DEF)
        if vector_store is not None and embedder is not None:
            tool_defs.append(_VECTOR_SEARCH_DEF)
        self._tool_defs = tool_defs

    async def query_stream(self, query: str) -> AsyncGenerator[str | QueryResult, None]:
        """Run the agentic retrieval loop, yielding str tokens then a final QueryResult."""
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        all_records: list[dict] = []
        all_chunks: list[Chunk] = []

        for round_num in range(self._max_rounds):
            logger.debug("query_runner.round round=%d query_len=%d", round_num, len(query))
            result = await self._llm.complete(messages, tools=self._tool_defs or None)

            tool_calls = result.get("tool_calls")
            if not tool_calls:
                answer = result.get("content") or ""
                yield answer
                yield QueryResult(
                    answer=answer,
                    structured_records=all_records,
                    chunks=all_chunks,
                    passthrough=False,
                )
                return

            messages.append({
                "role": "assistant",
                "content": result.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                inputs: dict = {}
                try:
                    inputs = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    pass

                name = tc["name"]
                logger.info("query_runner.tool_call name=%s", name)

                if name == "structured_lookup":
                    records, tool_output, passthrough = await self._run_structured_lookup(inputs)
                    all_records.extend(records)
                    if passthrough:
                        yield tool_output
                        yield QueryResult(
                            answer=tool_output,
                            structured_records=all_records,
                            chunks=all_chunks,
                            passthrough=True,
                        )
                        return
                elif name == "vector_search":
                    chunks, tool_output = await self._run_vector_search(inputs)
                    all_chunks.extend(chunks)
                else:
                    tool_output = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

        logger.error("query_runner.max_rounds_reached max_rounds=%d", self._max_rounds)
        answer = (
            "Unable to complete the query within the allowed number of retrieval rounds. "
            "Please try a more specific request."
        )
        yield answer
        yield QueryResult(
            answer=answer,
            structured_records=all_records,
            chunks=all_chunks,
            passthrough=False,
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _run_structured_lookup(
        self, inputs: dict
    ) -> tuple[list[dict], str, bool]:
        """Execute structured_lookup. Returns (records, tool_output, passthrough)."""
        if self._structured_store is None:
            return [], "structured_lookup is unavailable (no structured store configured)", False

        collection = str(inputs.get("collection", ""))
        raw_filters = inputs.get("filters") or []
        fields = inputs.get("fields") or []

        filters: list[Filter] = []
        for f in raw_filters:
            try:
                filters.append(_parse_filter(f))
            except (KeyError, ValueError) as exc:
                logger.warning("query_runner.structured_lookup.bad_filter filter=%s err=%s", f, exc)

        try:
            records = await self._structured_store.query(
                collection,
                filters or None,
                fields or None,
            )
        except Exception as exc:
            logger.exception("query_runner.structured_lookup.error collection=%s", collection)
            return [], f"structured_lookup error: {exc}", False

        json_str = json.dumps(records, default=str)
        estimated_tokens = _estimate_tokens(json_str)
        logger.info(
            "query_runner.structured_lookup.result collection=%s records=%d estimated_tokens=%d",
            collection,
            len(records),
            estimated_tokens,
        )

        if estimated_tokens > self._passthrough_token_threshold:
            logger.info(
                "query_runner.structured_lookup.passthrough estimated_tokens=%d threshold=%d",
                estimated_tokens,
                self._passthrough_token_threshold,
            )
            return records, _format_records_as_text(records), True

        return records, json_str, False

    async def _run_vector_search(self, inputs: dict) -> tuple[list[Chunk], str]:
        """Execute vector_search. Returns (chunks, tool_output)."""
        if self._vector_store is None or self._embedder is None:
            return [], "vector_search is unavailable (no vector store configured)"

        query_text = str(inputs.get("query", ""))
        top_k = min(int(inputs.get("top_k") or 5), 20)
        collection = str(inputs.get("collection") or self._default_vector_collection or "")

        if not collection:
            return [], "vector_search error: no collection specified and no default collection configured"

        try:
            (query_embedding,) = await self._embedder.embed([query_text])
            chunks = await self._vector_store.search(collection, query_embedding, top_k)
        except Exception as exc:
            logger.exception("query_runner.vector_search.error collection=%s", collection)
            return [], f"vector_search error: {exc}"

        logger.info(
            "query_runner.vector_search.result collection=%s chunks=%d",
            collection,
            len(chunks),
        )
        return chunks, _format_chunks(chunks)
