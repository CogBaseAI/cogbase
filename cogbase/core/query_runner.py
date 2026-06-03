"""QueryRunner — unified LLM agent loop with skill routing and retrieval tools.

The runner drives an LLM agent loop that handles two complementary concerns:

  Skill routing  — selects the best skill for each user request, builds a
                   skill-specific system prompt, and re-evaluates after every
                   tool-call round so the agent can switch skills mid-task.

  Retrieval      — exposes ``structured_lookup``, ``vector_search``, and
                   ``read_document`` as system tools when the corresponding
                   stores are configured.
                   The passthrough rule applies: if ``structured_lookup``
                   returns results whose estimated token count exceeds
                   ``passthrough_token_threshold``, the formatted records are
                   streamed directly and the loop exits without LLM synthesis.

Either concern can be used alone:

  - Skills only (no stores) — code-execution assistant, multi-step automation.
  - Stores only (no skills) — document retrieval and Q&A.
  - Both — skill-driven agents that can also query structured/vector data.

The loop yields ``str`` tokens during execution followed by a final
``QueryResult`` that carries accumulated records, chunks, and a passthrough flag.

Usage (retrieval mode)::

    runner = QueryRunner(
        llm=llm,
        structured_store=structured_store,
        vector_store=vector_store,
        embedder=embedder,
        structured_schemas=schemas,
    )
    async for item in runner.run("list all contracts expiring before 2026"):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print("passthrough:", item.passthrough)

Usage (skill mode)::

    runner = QueryRunner(llm=llm, skills=skills)
    async for item in runner.run("What's the weather in NYC?"):
        if isinstance(item, str):
            print(item, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import ChatMessage, CompletionResult, LLMBase, SystemTool, ToolDefinition
from cogbase.memory import MemoryRole, RetrievalKind, RetrievedItem, ShortTermMemory
from cogbase.stores import CollectionSchema, DocumentStoreBase, Filter, Op, StructuredStoreBase, VectorCollectionSchema, VectorStoreBase

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class DocumentSlice(BaseModel):
    """A slice of document text fetched by a read_document tool call."""

    doc_id: str
    offset: int
    length: int
    text: str

    @property
    def slice_id(self) -> str:
        return f"{self.doc_id}:{self.offset}:{self.length}"


class QueryResult(BaseModel):
    """Final result of a QueryRunner invocation.

    Attributes:
        answer:              Full response text.
        structured_records:  All records returned by structured_lookup calls.
        chunks:              All chunks returned by vector_search calls.
        document_slices:     Document text slices fetched by read_document calls.
        passthrough:         True when records were returned directly without
                             LLM synthesis (token threshold exceeded).
        input_tokens:        Total prompt tokens consumed across all LLM calls.
        output_tokens:       Total completion tokens generated across all LLM calls.
    """

    answer: str
    structured_records: list[dict] = []
    chunks: list[Chunk] = []
    document_slices: list[DocumentSlice] = []
    passthrough: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Execution tools (available when skills are present)
# ---------------------------------------------------------------------------

_BASE_TOOLS: list[ToolDefinition] = [
    {
        "name": "python",
        "description": (
            "Execute inline Python code and return stdout/stderr. "
            "Use for computation, data processing, or logic that does not need a separate script file."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code to execute"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "shell",
        "description": (
            "Run a bash command and return stdout/stderr. "
            "Use whenever the active skill instructs you to run a command, especially "
            "lines like 'python <script_path> ...' — those are shell commands, not inline code."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "A bash command to execute"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Retrieval tool definitions
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

_READ_DOCUMENT_DEF: ToolDefinition = {
    "name": "read_document",
    "description": (
        "Read a slice of a document's original text by character offset. "
        "Use after vector_search to get broader context around a relevant passage. "
        "Chunks returned by vector_search include char_offset and char_length showing "
        "where they appear in the source document. "
        "To read context before a chunk, set offset to an earlier integer; for example, "
        "if char_offset is 1200, use offset=700 to read 500 characters before it. "
        "To read context after a chunk, set offset to char_offset and length to an integer "
        "large enough to cover beyond char_length."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "Document ID to read from.",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset to start reading from (default: 0).",
            },
            "length": {
                "type": "integer",
                "description": "Number of characters to read (default: 2000, max: 10000).",
            },
        },
        "required": ["doc_id"],
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
            "collection": {
                "type": "string",
                "description": "Vector collection to search.",
            },
            "query": {
                "type": "string",
                "description": "Search query text.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 10, max: 20).",
            },
        },
        "required": ["collection", "query"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# System prompt helpers
# ---------------------------------------------------------------------------

_RETRIEVAL_BASE_PROMPT = """\
You are a document intelligence assistant. Answer the user's query by retrieving
evidence from the available tools, then synthesising a final answer.

Rules:
- Call tools as needed to gather evidence before answering.
- Do not invent facts not present in retrieved evidence.
- When the evidence is sufficient, produce your final answer directly (no tool calls).
- When your answer references a passage, cite its id in brackets, e.g. [contract_001_0].
"""

_SCHEMA_HEADER = "\nAvailable structured collections (use these names and fields exactly):\n"

_FILTER_LEGEND = """
Filter operators: =, !=, <, >, <=, >=, like, in, not_in, is_null, is_not_null
  in / not_in:            value must be a JSON array
  like:                   SQL LIKE pattern (% = any sequence)
  is_null / is_not_null:  omit "value"
  JSON sub-keys:          use "field.subkey" notation
"""


_VECTOR_COLLECTIONS_HEADER = "\nAvailable vector collections (pass name to vector_search 'collection' param):\n"


def _build_retrieval_prompt(
    schemas: list[CollectionSchema] | None,
    vector_schemas: list[VectorCollectionSchema] | None = None,
) -> str:
    lines = [_RETRIEVAL_BASE_PROMPT]
    if schemas:
        lines.append(_SCHEMA_HEADER)
        for schema in schemas:
            fields_str = ", ".join(
                f"{name} ({fschema.type.value})"
                for name, fschema in schema.fields.items()
            )
            header = f"  {schema.name} ({schema.description})" if schema.description else f"  {schema.name}"
            lines.append(f"{header}: {fields_str}")
        lines.append(_FILTER_LEGEND)
    if vector_schemas:
        lines.append(_VECTOR_COLLECTIONS_HEADER)
        for vs in vector_schemas:
            lines.append(f"  - {vs.name}: {vs.description}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
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

    # Group by doc_id, preserving first-occurrence (relevance) order across docs.
    doc_order: list[str] = []
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.doc_id not in by_doc:
            doc_order.append(c.doc_id)
            by_doc[c.doc_id] = []
        by_doc[c.doc_id].append(c)

    def _seq_key(c: Chunk) -> tuple:
        return (0, c.char_offset) if c.char_offset is not None else (1, c.chunk_id)

    lines = ["Passages:"]
    for doc_id in doc_order:
        for chunk in sorted(by_doc[doc_id], key=_seq_key):
            location = f"doc: {chunk.doc_id}"
            if chunk.char_offset is not None and chunk.char_length is not None:
                location += f", chars {chunk.char_offset}–{chunk.char_offset + chunk.char_length}"
            lines.append(f"  [{chunk.chunk_id}] ({location})\n  {chunk.text.strip()}")
    return "\n".join(lines)


def _extract_cited_ids(answer: str) -> set[str]:
    """Return all bracket-cited IDs from *answer*, e.g. [contract_001_0]."""
    return set(re.findall(r"\[([^\]]+)\]", answer))


def _filter_cited_chunks(all_chunks: list[Chunk], cited_ids: set[str]) -> list[Chunk]:
    """Return chunks whose chunk_id appears in *cited_ids*.

    Falls back to all chunks only when *cited_ids* is empty (LLM produced no
    citations at all), so the caller always gets something useful.  When the LLM
    did cite IDs but none matched chunks (e.g. it only cited slices), returns [].
    """
    matched = [c for c in all_chunks if c.chunk_id in cited_ids]
    return matched if (matched or cited_ids) else all_chunks


def _filter_cited_slices(all_slices: list[DocumentSlice], cited_ids: set[str]) -> list[DocumentSlice]:
    """Return slices whose slice_id appears in *cited_ids*.

    Falls back to all slices only when *cited_ids* is empty.  When the LLM cited
    IDs but none matched slices (e.g. it only cited chunks), returns [].
    """
    matched = [s for s in all_slices if s.slice_id in cited_ids]
    return matched if (matched or cited_ids) else all_slices


# ---------------------------------------------------------------------------
# QueryRunner
# ---------------------------------------------------------------------------


class QueryRunner:
    """Unified LLM agent loop with skill routing and retrieval tools.

    Args:
        app_name:                    Application name; used to scope document store reads.
        llm:                         LLM backend.
        document_store:              Document store; enables the ``read_document`` tool.
        structured_store:            Structured store; enables the ``structured_lookup`` tool.
        vector_store:                Vector store; enables the ``vector_search`` tool
                                     (requires *embedder*).
        embedder:                    Embedder for ``vector_search``.
        vector_schemas:              ``VectorCollectionSchema`` list for all vector
                                     collections, injected into the retrieval system
                                     prompt so the LLM can choose the right one.
        structured_schemas:          Schema list injected into the retrieval system prompt
                                     so the LLM knows available collections and field types.
        skills:                      Skills available for routing. Pass ``None`` or ``[]``
                                     to skip skill selection and use the retrieval system prompt.
        system_tools:                Custom store-backed or service tools injected by the
                                     caller. Available on every turn alongside retrieval
                                     tools and (when skills are present) execution tools.
        max_calls:                   Maximum LLM completion rounds per run. Default 10.
        passthrough_token_threshold: Estimated token count of ``structured_lookup`` results
                                     above which records are returned directly without LLM
                                     synthesis. None means disabled. Defaults None.
    """

    def __init__(
        self,
        app_name: str,
        llm: LLMBase,
        document_store: DocumentStoreBase,
        structured_store: StructuredStoreBase | None = None,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        vector_schemas: list[VectorCollectionSchema] | None = None,
        structured_schemas: list[CollectionSchema] | None = None,
        skills: list | None = None,
        system_tools: list[SystemTool] | None = None,
        max_calls: int = 10,
        passthrough_token_threshold: int | None = None,
        short_term: ShortTermMemory | None = None,
        context_token_budget: int | None = None,
    ) -> None:
        self._app_name = app_name
        self._llm = llm
        self._document_store = document_store
        self._structured_store = structured_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._retrieval_system_prompt = _build_retrieval_prompt(
            structured_schemas, vector_schemas
        )
        self._skills: list = skills or []
        self._system_tools: dict[str, SystemTool] = {t.name: t for t in (system_tools or [])}
        self._max_calls = max_calls
        self._passthrough_token_threshold = passthrough_token_threshold
        self._short_term = short_term
        self._context_token_budget = context_token_budget

        # Retrieval tool definitions — exposed as _tool_defs for introspection.
        self._tool_defs: list[ToolDefinition] = []
        if structured_schemas:
            self._tool_defs.append(_STRUCTURED_LOOKUP_DEF)
        if vector_schemas:
            self._tool_defs.append(_VECTOR_SEARCH_DEF)
        if document_store is not None and app_name is not None:
            self._tool_defs.append(_READ_DOCUMENT_DEF)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def structured_store(self):
        """The structured store, if configured."""
        return self._structured_store

    @property
    def vector_store(self):
        """The vector store, if configured."""
        return self._vector_store

    # ------------------------------------------------------------------
    # Skill selection helpers (used when skills are provided to run())
    # ------------------------------------------------------------------

    async def select(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
    ):
        """Ask the LLM to pick the best skill for *user_input*; returns None if none apply."""
        skills = self._skills
        if not skills:
            return None

        skill_list = "\n".join(
            f"{i + 1}. name={s.name!r}  description={s.description!r}"
            for i, s in enumerate(skills)
        )
        history_text = "\n".join(
            f"[{m['role']}] {m.get('content', '')}" for m in (history or [])
        )

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "You are a skill router. Given the conversation history, current user question, "
                    "and available skills, return the name of the single most relevant skill, "
                    "or 'none' if no skill applies. "
                    "Output only the skill name or 'none' — no explanation, no punctuation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Current user question:\n{user_input}\n\n"
                    f"Available skills:\n{skill_list}"
                ),
            },
        ]

        result = await self._llm.complete(messages)
        chosen = (result["content"] or "").lower().strip("'\"")

        if chosen == "none":
            logger.info("[runner] no skill selected for: %s", user_input[:100])
            return None

        for skill in skills:
            if skill.name.lower() == chosen:
                logger.info("[runner] selected skill '%s'", skill.name)
                return skill

        logger.error("[runner] router returned unknown skill '%s', ignoring", chosen)
        return None

    def build_system_prompt(self, base_prompt: str, skill=None) -> str:
        """Merge base_prompt, retrieval schema info, and (optionally) skill instructions."""
        parts = [base_prompt]

        if self._tool_defs:
            parts.append(self._retrieval_system_prompt)

        if skill is not None:
            base_dir = str(skill.source_path.parent) if skill.source_path else ""
            metadata_block = ""
            if skill.metadata:
                metadata_block = (
                    "Skill metadata:\n"
                    f"```json\n{json.dumps(skill.metadata, ensure_ascii=False, indent=2)}\n```\n\n"
                )
            skill_section = (
                f"## Active Skill: {skill.name}\n\n"
                + (f"Skill base directory: `{base_dir}`\n\n" if base_dir else "")
                + metadata_block
                + "Follow the skill's instructions below to complete the user's request. "
                "Use the `shell` tool to run any commands it suggests.\n\n"
                + skill.raw_markdown
            )
            parts.append(skill_section)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
        base_prompt: str = "You are a helpful assistant.",
        top_k: int = 10,
        session_id: str | None = None,
    ) -> AsyncGenerator[str | QueryResult, None]:
        """Drive the agent loop, yielding str tokens then a final QueryResult.

        Args:
            user_input:  The user's request.
            history:     Prior conversation messages.  Ignored when short-term
                         memory is active and ``session_id`` is supplied — the
                         session's assembled context is the source of truth.
            base_prompt: Base system prompt; merged with retrieval schema info and
                         skill instructions when those are configured.
            top_k:       Default number of chunks returned per vector_search call.
                         The LLM may request fewer; this value is used when the LLM
                         omits top_k from its tool arguments. Hard-capped at 20.
            session_id:  When set and short-term memory is configured, the runner
                         records the turn into the session and builds context from
                         it instead of ``history``.
        """
        skills = self._skills
        memory_active = self._short_term is not None and session_id is not None

        # Slot 0 is reserved for the system prompt; updated each iteration.
        messages: list[ChatMessage] = [{"role": "system", "content": ""}]
        if memory_active:
            # Server-side session context replaces caller-passed history.
            await self._short_term.append_message(session_id, MemoryRole.USER, user_input)
            context = await self._short_term.build_context(
                session_id=session_id,
                query=user_input,
                token_budget=self._context_token_budget,
            )
            messages.extend(context)
        else:
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_input})

        all_records: list[dict] = []
        all_chunks: list[Chunk] = []
        all_slices: list[DocumentSlice] = []
        total_input_tokens: int = 0
        total_output_tokens: int = 0

        # Select skill once before the loop. Support switching skill in the for loop when needed.
        current_skill = None
        if skills:
            current_skill = await self.select(user_input, history)
            if current_skill is not None:
                logger.info("[runner] active skill → '%s'", current_skill.name)
                yield f"Using skill: {current_skill.name}..."

        messages[0] = {"role": "system", "content": self.build_system_prompt(base_prompt, current_skill)}

        for _ in range(self._max_calls):
            # --- LLM completion (streaming) ---
            tools = self._all_tools(current_skill is not None)
            tokens = []
            final_result: CompletionResult | None = None
            async for chunk in self._llm.complete_stream(messages, tools=tools or None):
                if isinstance(chunk, str):
                    tokens.append(chunk)
                    yield chunk
                else:
                    final_result = chunk

            if final_result is not None:
                usage = final_result.get("usage")
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
            tool_calls = final_result.get("tool_calls") if final_result else None
            if not tool_calls:
                answer = "".join(tokens) + "\n"
                cited_ids = _extract_cited_ids(answer)
                if memory_active:
                    await self._short_term.append_message(
                        session_id, MemoryRole.ASSISTANT, answer
                    )
                yield QueryResult(
                    answer=answer,
                    structured_records=all_records,
                    chunks=_filter_cited_chunks(all_chunks, cited_ids),
                    document_slices=_filter_cited_slices(all_slices, cited_ids),
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
                return

            messages.append({
                "role": "assistant",
                "content": "".join(tokens) or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            })

            tool_names = ", ".join(tc["name"] for tc in tool_calls)
            logger.info("[runner] tool_calls (skill=%s): %s", current_skill.name if current_skill else "none", tool_names)

            yield f"Executing: {tool_names}...\n"

            # --- Tool execution ---
            for tc in tool_calls:
                inputs: dict = {}
                try:
                    inputs = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    pass

                name = tc["name"]
                logger.info("[runner] execute_tool %s(%s)", name, json.dumps(inputs)[:300])

                if name == "structured_lookup":
                    records, tool_output, passthrough = await self._run_structured_lookup(inputs)
                    all_records.extend(records)
                    if memory_active:
                        await self._record_record_retrievals(session_id, records)
                    if passthrough:
                        if memory_active:
                            await self._short_term.append_message(
                                session_id, MemoryRole.ASSISTANT, tool_output
                            )
                        yield QueryResult(
                            answer=tool_output,
                            structured_records=all_records,
                            chunks=all_chunks,
                            document_slices=all_slices,
                            passthrough=True,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                        )
                        return
                elif name == "vector_search":
                    seen_chunk_ids = {c.chunk_id for c in all_chunks}
                    chunks, tool_output = await self._run_vector_search(inputs, exclude_ids=seen_chunk_ids, default_top_k=top_k)
                    all_chunks.extend(chunks)
                    if memory_active:
                        await self._record_chunk_retrievals(session_id, chunks)
                elif name == "read_document":
                    doc_slice, tool_output = await self._run_read_document(inputs)
                    if doc_slice is not None:
                        all_slices.append(doc_slice)
                else:
                    tool_output = await self._execute_tool(name, inputs, current_skill)
                    logger.info("[runner] execute_tool done: %s", tool_output[:300])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

        logger.error("[runner] max_calls (%d) reached. skill=%s", self._max_calls, current_skill.name if current_skill else "none")
        answer = (
            "I was unable to complete your request within the allowed number of steps. "
            "Please try a simpler or more specific request."
        )
        # limit the number of references to avoid large context
        yield QueryResult(
            answer=answer,
            structured_records=all_records[:2],
            chunks=all_chunks[:2],
            document_slices=all_slices[:2],
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _all_tools(self, skill_active: bool) -> list[ToolDefinition]:
        tools: list[ToolDefinition] = []
        if skill_active:
            tools.extend(_BASE_TOOLS)
        tools.extend(t.definition for t in self._system_tools.values())
        tools.extend(self._tool_defs)
        return tools

    async def _execute_tool(self, name: str, inputs: dict, skill=None) -> str:
        if name in self._system_tools:
            try:
                result = self._system_tools[name].handler(inputs)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    return await result
                return result  # type: ignore[return-value]
            except Exception as e:
                return f"Tool error ({name}): {e}"

        env = self._tool_env(skill)
        if name == "python":
            return await self._run_python(inputs.get("code", ""), env)
        if name == "shell":
            return await self._run_shell(inputs.get("command", ""), env)
        return f"Unknown tool: {name}"

    async def _run_structured_lookup(
        self, inputs: dict
    ) -> tuple[list[dict], str, bool]:
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
                logger.warning("[runner] structured_lookup.bad_filter filter=%s err=%s", f, exc)

        try:
            records = await self._structured_store.query(
                collection,
                filters or None,
                fields or None,
            )
        except Exception as exc:
            logger.exception("[runner] structured_lookup.error collection=%s", collection)
            return [], f"structured_lookup error: {exc}", False

        json_str = json.dumps(records, default=str)
        logger.info(
            "[runner] structured_lookup.result collection=%s records=%d",
            collection, len(records),
        )

        if self._passthrough_token_threshold:
            estimated_tokens = _estimate_tokens(json_str)
            if estimated_tokens > self._passthrough_token_threshold:
                logger.info(
                    "[runner] structured_lookup.passthrough estimated_tokens=%d threshold=%d",
                    estimated_tokens, self._passthrough_token_threshold,
                )
                return records, _format_records_as_text(records), True

        return records, json_str, False

    async def _run_vector_search(self, inputs: dict, exclude_ids: set[str] | None = None, default_top_k: int = 10) -> tuple[list[Chunk], str]:
        if self._vector_store is None or self._embedder is None:
            return [], "vector_search is unavailable (no vector store configured)"

        collection = str(inputs.get("collection") or "")
        query_text = str(inputs.get("query", ""))
        top_k = min(int(inputs.get("top_k") or default_top_k), 20)
        search_top_k = top_k + len(exclude_ids or ())

        if not collection:
            return [], "vector_search error: no collection specified"

        try:
            (query_embedding,) = await self._embedder.embed([query_text])
            chunks = await self._vector_store.search(collection, query_text, query_embedding, search_top_k)
        except Exception as exc:
            logger.exception("[runner] vector_search.error collection=%s", collection)
            return [], f"vector_search error: {exc}"

        if exclude_ids:
            chunks = [c for c in chunks if c.chunk_id not in exclude_ids]
        chunks = chunks[:top_k]

        chunk_summary = ", ".join(
            f"{c.chunk_id}[{c.char_offset}:{c.char_length}]" for c in chunks
        )
        logger.info("[runner] vector_search.result collection=%s chunks=%d ids=[%s]", collection, len(chunks), chunk_summary)
        return chunks, _format_chunks(chunks)

    async def _run_read_document(self, inputs: dict) -> tuple[DocumentSlice | None, str]:
        if self._document_store is None or self._app_name is None:
            return None, "read_document is unavailable (no document store configured)"

        doc_id = str(inputs.get("doc_id", ""))
        if not doc_id:
            return None, "read_document error: doc_id is required"

        offset = max(0, int(inputs.get("offset") or 0))
        length = min(int(inputs.get("length") or 2000), 10000)

        try:
            text = await self._document_store.load(self._app_name, doc_id)
        except KeyError:
            return None, f"read_document error: document '{doc_id}' not found"
        except Exception as exc:
            logger.exception("[runner] read_document.error doc_id=%s", doc_id)
            return None, f"read_document error: {exc}"

        slice_text = text[offset : offset + length]
        total = len(text)
        logger.info("[runner] read_document doc_id=%s offset=%d length=%d total=%d", doc_id, offset, length, total)
        doc_slice = DocumentSlice(doc_id=doc_id, offset=offset, length=len(slice_text), text=slice_text)
        location = f"doc: {doc_id}, chars {offset}–{offset + len(slice_text)} of {total}"
        return doc_slice, f"Passage:\n  [{doc_slice.slice_id}] ({location})\n  {slice_text.strip()}"

    async def _run_python(self, code: str, env: dict) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp = f.name
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, tmp,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                return await self._read_proc(proc)
            finally:
                os.unlink(tmp)
        except Exception as e:
            return f"Python error: {e}"

    async def _run_shell(self, command: str, env: dict) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            return await self._read_proc(proc)
        except Exception as e:
            return f"Shell error: {e}"

    @staticmethod
    async def _read_proc(proc: asyncio.subprocess.Process) -> str:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Process timed out"
        return stdout.decode().strip() or stderr.decode().strip() or "(no output)"

    @staticmethod
    def _tool_env(skill) -> dict:
        env = os.environ.copy()
        if skill and skill.site_packages:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{skill.site_packages}:{existing}" if existing else skill.site_packages
        return env

    # ------------------------------------------------------------------
    # Short-term memory recording
    # ------------------------------------------------------------------

    async def _record_chunk_retrievals(self, session_id: str, chunks: list[Chunk]) -> None:
        """Record vector_search chunks as session retrievals (best-effort)."""
        if not chunks:
            return
        items = [
            RetrievedItem(
                kind=RetrievalKind.CHUNK,
                ref_id=c.chunk_id,
                text=c.text,
                source="vector_search",
            )
            for c in chunks
        ]
        await self._short_term.append_retrievals(session_id, items)

    async def _record_record_retrievals(self, session_id: str, records: list[dict]) -> None:
        """Record structured_lookup records as session retrievals (best-effort)."""
        if not records:
            return
        items = [
            RetrievedItem(
                kind=RetrievalKind.RECORD,
                text=json.dumps(rec, default=str)[:1000],
                source="structured_lookup",
            )
            for rec in records
        ]
        await self._short_term.append_retrievals(session_id, items)

    async def compact_messages(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Summarise *messages* into a minimal list to recover from context overflow."""
        transcript = "\n".join(
            f"[{m['role']}] {str(m.get('content', ''))[:500]}"
            for m in messages
        )
        result = await self._llm.complete([
            {
                "role": "user",
                "content": (
                    "Compress the following conversation transcript into a concise bullet-point "
                    "summary preserving all key decisions, tool outputs, and conclusions. Be terse.\n\n"
                    + transcript
                ),
            }
        ])
        summary = result.get("content") or "(empty summary)"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compacted context:\n\n{summary}\n\nContinue from this point."},
        ]
