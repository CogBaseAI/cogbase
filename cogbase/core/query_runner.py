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
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pydantic import BaseModel

from cogbase.llms.summarization import (
    WORKING_CONTEXT_PROMPT,
    estimate_messages_tokens,
    estimate_tokens,
    render_message,
    summarise_chunk_tokens,
    summarize_text,
)
from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import ChatMessage, CompletionResult, LLMBase, SystemTool, ToolDefinition
from cogbase.memory import EpisodicMemory, EventRef, LongTermMemory, LongTermRecord, MemoryKind, ShortTermMemory
from cogbase.stores import CollectionSchema, DocumentStoreBase, Filter, LogFenced, Op, StructuredStoreBase, VectorCollectionSchema, VectorStoreBase

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT = 30  # seconds

# Common stdlib modules pre-imported before every inline `python` snippet. The
# model frequently uses these (json.dump, re.sub, datetime, pathlib) without
# emitting the corresponding `import`, which fails as a NameError even though the
# module is always available. Re-imports are idempotent, so prepending this can
# never shadow or conflict with code that does import them. Kept on a single
# physical line so tracebacks only shift by one line relative to the model's code.
_PY_PREAMBLE = (
    "import json, os, sys, re, math, datetime, pathlib, collections, itertools\n"
)

# Root of the per-(app, session) local scratch dirs that skill subprocesses use
# for their working files (fetched documents, intermediate JSON, redline drafts).
# Deterministic and reusable so retries/follow-up turns reuse materialized files
# and the model can predict paths (no filesystem-wide `find`). Durable outputs
# still round-trip through the document store via save_artifact; this tree is a
# rebuildable local projection, so a fresh node re-materializes here from the
# store rather than requiring shared storage.
_DEFAULT_WORK_ROOT = os.environ.get("COGBASE_WORK_ROOT") or os.path.join(
    tempfile.gettempdir(), "cogbase-work"
)

# A traversal command (find/grep/fd/rg) rooted at the filesystem root ("/") — a
# whole-disk scan the model only reaches for when it has lost a path. Matches
# `find / -name x`, `grep -r foo /`, etc. but not a scoped `find /some/dir` or a
# relative/`$COGBASE_*`-anchored path. Used to refuse the scan before it runs.
_ROOT_FS_SCAN = re.compile(r"\b(?:find|grep|fd|rg)\b[^|;&]*?\s/(?:\s|$)")

# Attempts to land a turn's continuity events (user_message / final_answer) in the
# log before the turn is acknowledged.  The buffer is retained between attempts
# (it is the retry buffer), so each retry simply re-appends the same batch.
_CONTINUITY_FLUSH_RETRIES = 3


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


class ArtifactRef(BaseModel):
    """A file persisted by a save_artifact call, with its download link.

    ``download_path`` is the ready-to-use, app-scoped path served by the
    generated-artifact download endpoint; the runner both hands it back in the
    tool output and appends it — as a markdown link — to the final answer, so a
    downloadable file always surfaces an explicit link the caller can render.
    """

    artifact_id: str
    filename: str
    download_path: str

    @property
    def markdown_link(self) -> str:
        return f"[{self.filename}]({self.download_path})"


class QueryResult(BaseModel):
    """Final result of a QueryRunner invocation.

    Attributes:
        answer:              Full response text.
        structured_records:  All records returned by structured_lookup calls.
        chunks:              All chunks returned by vector_search calls.
        document_slices:     Document text slices fetched by read_document calls.
        memories:            Long-term memories the answer actually used — the
                             records of every injected memory block (recall and
                             memory_lookup) whose block id the answer cited in
                             brackets. A block is all-or-nothing. Empty when no
                             memory block was cited.
        passthrough:         True when records were returned directly without
                             LLM synthesis (token threshold exceeded).
        input_tokens:        Total prompt tokens consumed across all LLM calls.
        output_tokens:       Total completion tokens generated across all LLM calls.
    """

    answer: str
    structured_records: list[dict] = []
    chunks: list[Chunk] = []
    document_slices: list[DocumentSlice] = []
    memories: list[LongTermRecord] = []
    passthrough: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Dependency bundles
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResources:
    """Stores, embedder, and schemas the runner retrieves evidence from.

    ``document_store`` is required (it backs ``read_document`` and is scoped by
    ``app_id``); the rest are optional and each enables its corresponding tool
    when present: ``structured_schemas`` → ``structured_lookup``,
    ``vector_schemas`` (+ ``vector_store`` + ``embedder``) → ``vector_search``.
    """

    document_store: DocumentStoreBase
    structured_store: StructuredStoreBase | None = None
    vector_store: VectorStoreBase | None = None
    embedder: EmbeddingBase | None = None
    structured_schemas: list[CollectionSchema] | None = None
    vector_schemas: list[VectorCollectionSchema] | None = None


@dataclass
class MemoryTiers:
    """The three persistent memory tiers; any subset may be wired.

    ``short_term`` and ``episodic`` engage only when a ``session_id`` is passed
    to ``run()`` and must share a log store (see ``QueryRunner`` docstring);
    ``long_term`` is session-independent and drives recall + ``memory_lookup``.
    """

    short_term: ShortTermMemory | None = None
    episodic: EpisodicMemory | None = None
    long_term: LongTermMemory | None = None


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
    {
        "name": "fetch_document",
        "description": (
            "Materialize a stored document's original uploaded file (e.g. its .docx) to a "
            "local path so a skill script can process the raw file — unlike read_document, "
            "which returns extracted text. Returns the local file path to operate on."
        ),
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "description": "Document ID whose original file to fetch."}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_artifact",
        "description": (
            "Persist a locally-produced file (e.g. a generated or edited .docx) so the user "
            "can download it. Call this after a skill script writes its output file. Returns "
            "an artifact id and a ready-to-use markdown download link — include that exact link "
            "in your answer so the user can download the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local path of the file to persist."},
                "filename": {"type": "string", "description": "Suggested download filename (default: the path's basename)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_artifact",
        "description": (
            "Materialize a previously save_artifact'd file (by its artifact id) back to a "
            "local path so a skill script can reload and patch it across turns — the inbound "
            "half of save_artifact. Use this to reopen working state (e.g. an ops.json) that a "
            "prior turn produced. Returns the local file path to operate on."
        ),
        "parameters": {
            "type": "object",
            "properties": {"artifact_id": {"type": "string", "description": "Artifact id returned by a prior save_artifact call."}},
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_artifact",
        "description": (
            "Delete a previously save_artifact'd file (by its artifact id) from the document "
            "store. Use to clean up working state (e.g. an ops.json) once a task is complete. "
            "Idempotent: deleting a missing artifact is treated as success."
        ),
        "parameters": {
            "type": "object",
            "properties": {"artifact_id": {"type": "string", "description": "Artifact id returned by a prior save_artifact call."}},
            "required": ["artifact_id"],
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

_MEMORY_LOOKUP_DEF: ToolDefinition = {
    "name": "memory_lookup",
    "description": (
        "Search the assistant's long-term memory: durable facts, preferences, "
        "corrections, and retrieval hints remembered from past sessions. Use when "
        "the user references something previously told to the assistant, or asks "
        "what is known/remembered about a person, project, or topic — or when the "
        "memory already provided in context is insufficient. Results are "
        "memory-derived background knowledge, NOT cited document evidence. "
        "Provide at least one of query / kind / entities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search over memory content.",
            },
            "kind": {
                "type": "string",
                "enum": ["preference", "fact", "correction", "retrieval_hint"],
                "description": "Restrict to one memory kind.",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict to memories about these entities (people, projects, "
                    "organizations, systems), e.g. [\"acme corp\"]."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max memories to return (default: 5, max: 20).",
            },
        },
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
Filter operators: =, !=, <, >, <=, >=, like, in, not_in, overlaps, is_null, is_not_null
  in / not_in:            value must be a JSON array
  overlaps:               json-typed array field shares >=1 element with value (a JSON array)
  like:                   SQL LIKE pattern (% = any sequence)
  is_null / is_not_null:  omit "value"
  JSON sub-keys:          use "field.subkey" notation
"""


_VECTOR_COLLECTIONS_HEADER = "\nAvailable vector collections (pass name to vector_search 'collection' param):\n"


# Precedence policy carried with any injected long-term memory.  Memory is
# framed as dated, user-attributed claims; on a conflict with retrieved
# documents the model prefers the documents (the fresher, authoritative source
# for subject-matter facts) but surfaces the discrepancy — which keeps a user
# correction (memory that contradicts the corpus because the corpus was wrong)
# from being silently buried.  See docs/long-term-memory.md.
_MEMORY_EVIDENCE_POLICY = (
    "Each item is dated, memory-derived background the user asserted or confirmed "
    "as of the given date — NOT cited document evidence; do not present it as a "
    "sourced fact. When a memory conflicts with what the documents currently say, "
    "prefer the document evidence as the fresher, authoritative source and note "
    "the discrepancy, rather than relying on the older memory."
)


def _format_memory_line(m: LongTermRecord) -> str:
    """One recalled memory as a dated, attributed line within a memory block.

    A whole memory block is cited as one passage (its block id leads the block
    header — see ``_recall_memory_block`` / ``_run_memory_lookup``), so the
    individual records need no per-record id; the opaque ``memory_id`` UUID would
    only add noise the model cannot meaningfully cite.

    The "as of" anchor is the memory's ``observed_at`` (when its source turn was
    asserted — see LongTermRecord.observed_at), which is always set on a promoted
    record.  Using the observation date keeps a fact dated by when it was observed,
    not when distillation happened to run — the two coincide for an immediately-
    distilled live session and diverge for a delayed/batched distill or a replayed
    past dialogue.
    """
    as_of = m.observed_at.date().isoformat()
    line = f"- [{m.kind.value}, as of {as_of}] {m.content}"
    if m.entities:
        line += f" (entities: {', '.join(m.entities)})"
    return line


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


def _append_download_links(answer: str, artifacts: list[ArtifactRef]) -> str:
    """Guarantee every saved artifact surfaces an explicit download link.

    The model is told the ready-made link in the ``save_artifact`` tool output,
    but whether it reproduces it verbatim in its prose is not reliable — so the
    runner appends a stable markdown link for any artifact whose download path
    isn't already in the answer.  An artifact the model already linked is skipped
    to avoid a duplicate; only genuinely missing ones get the appended block.
    """
    missing = [a for a in artifacts if a.download_path not in answer]
    if not missing:
        return answer
    lines = "\n".join(f"- {a.markdown_link}" for a in missing)
    sep = "" if answer.endswith("\n") else "\n"
    return f"{answer}{sep}\n**Download:**\n{lines}\n"


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


def _format_memory_block(title: str, block_id: str, memories: list[LongTermRecord]) -> str:
    """Render *memories* as one citable block led by *block_id*.

    The block is a single passage: the header tells the model to cite ``[block_id]``
    if it uses any of the block, and ``_cited_block_memories`` then surfaces all of
    the block's records on the QueryResult when that id appears in the answer.

    Lines are rendered oldest -> newest so the dated entries read as a timeline: the
    model can follow how a belief evolved (a correction sits after what it corrects)
    and the most-current fact lands last, lining up with the recency the precedence
    policy tells it to prefer.  Stable sort on the relevance-ordered input keeps
    vector relevance as the tiebreaker within a single date; only the rendering is
    reordered — the returned records keep recall/lookup's native order, the contract
    callers expect.
    """
    ordered = sorted(memories, key=lambda m: m.observed_at)
    lines = "\n".join(_format_memory_line(m) for m in ordered)
    return (
        f"{title} Cite [{block_id}] in your answer if you use any of it.\n"
        + _MEMORY_EVIDENCE_POLICY + "\n" + lines
    )


def _cited_block_memories(
    memory_blocks: dict[str, list[LongTermRecord]], cited_ids: set[str]
) -> list[LongTermRecord]:
    """Return the records of every memory block the answer cited.

    Each injected memory block is one citable passage keyed by its block id, so a
    block is all-or-nothing: cite the block id and you get all its records.  Unlike
    chunks/slices there is no fall-back to all — memory is memory-derived
    background, not the document evidence the answer is grounded in, so a block the
    answer never cites was not used and is dropped from the QueryResult entirely.
    Records shared across cited blocks are de-duplicated by ``memory_id``.
    """
    out: list[LongTermRecord] = []
    seen: set[str] = set()
    for block_id, records in memory_blocks.items():
        if block_id not in cited_ids:
            continue
        for m in records:
            if m.memory_id not in seen:
                out.append(m)
                seen.add(m.memory_id)
    return out


def _serialize_references(
    structured_records: list[dict],
    chunks: list[Chunk],
    document_slices: list[DocumentSlice],
    memories: list[LongTermRecord],
) -> dict:
    """Serialize an answer's references to the plain-dict payload persisted on the
    ``final_answer`` event (and re-hydrated by the transcript view).

    Mirrors how the API builds the live ``QueryResponse`` — chunks drop their
    embedding, memories keep only the query-facing projection — so a replayed
    transcript surfaces the same evidence shape the live response returned.
    """
    return {
        "structured_records": structured_records,
        "chunks": [c.model_dump(exclude={"embedding"}) for c in chunks],
        "document_slices": [s.model_dump() for s in document_slices],
        "memories": [
            {
                "memory_id": m.memory_id,
                "kind": m.kind.value,
                "content": m.content,
                "entities": list(m.entities),
            }
            for m in memories
        ],
    }


# ---------------------------------------------------------------------------
# QueryRunner
# ---------------------------------------------------------------------------


class QueryRunner:
    """Unified LLM agent loop with skill routing and retrieval tools.

    Args:
        app_id:                      Stable internal application id; used to scope document store reads
                                     and to build the download URL for artifacts persisted by
                                     ``save_artifact`` (keyed by app_id so the link survives a rename).
        llm:                         LLM backend.
        resources:                   ``RetrievalResources`` bundle: document store (required),
                                     optional structured/vector stores, embedder, and the
                                     structured/vector schemas injected into the retrieval
                                     system prompt. Each store/schema enables its tool when
                                     present (``structured_lookup``, ``vector_search``,
                                     ``read_document``).
        memory:                      ``MemoryTiers`` bundle: short-term (session working
                                     context, projected from the episodic log), episodic
                                     (durable append-only event log), and long-term (curated
                                     cross-session memory driving recall + ``memory_lookup``).
                                     Short-term and episodic must share a log store and engage
                                     only when a ``session_id`` is passed to ``run()``; long-term
                                     recall is injected as memory-derived context, kept distinct
                                     from document-backed evidence by the evidence policy.
                                     Defaults to an empty bundle (no memory).
        skills:                      Skills available for routing. Pass ``None`` or ``[]``
                                     to skip skill selection and use the retrieval system prompt.
        system_tools:                Custom store-backed or service tools injected by the
                                     caller. Available on every turn alongside retrieval
                                     tools and (when skills are present) execution tools.
        max_calls:                   Maximum LLM completion rounds per run. Default 20.
                                     Skill-routing runs are agentic (fetch → read →
                                     apply → verify → save, with retries), so they need
                                     more rounds than a plain retrieval query.
        passthrough_token_threshold: Estimated token count of ``structured_lookup`` results
                                     above which records are returned directly without LLM
                                     synthesis. None means disabled. Defaults None.
        enable_memory_lookup:        Whether to expose the ``memory_lookup`` tool when a
                                     long-term memory tier is present. Defaults False; set
                                     True to opt into on-demand memory recall. Memory
                                     injected into context applies regardless.
    """

    def __init__(
        self,
        app_id: str,
        llm: LLMBase,
        resources: RetrievalResources,
        memory: MemoryTiers | None = None,
        *,
        skills: list | None = None,
        system_tools: list[SystemTool] | None = None,
        max_calls: int = 20,
        passthrough_token_threshold: int | None = None,
        context_token_budget: int | None = None,
        enable_memory_lookup: bool = False,
        work_root: str | None = None,
    ) -> None:
        self._app_id = app_id
        self._llm = llm
        self._work_root = work_root or _DEFAULT_WORK_ROOT

        # Unpack the bundles into flat attrs so the rest of the class reads
        # them directly; the bundles are only an assembly-boundary convenience.
        self._document_store = resources.document_store
        self._structured_store = resources.structured_store
        self._vector_store = resources.vector_store
        self._embedder = resources.embedder

        mem = memory or MemoryTiers()
        self._short_term = mem.short_term
        self._episodic = mem.episodic
        self._long_term = mem.long_term

        self._retrieval_system_prompt = _build_retrieval_prompt(
            resources.structured_schemas, resources.vector_schemas
        )
        self._skills: list = skills or []
        self._system_tools: dict[str, SystemTool] = {t.name: t for t in (system_tools or [])}
        self._max_calls = max_calls
        self._passthrough_token_threshold = passthrough_token_threshold
        # In-loop compaction stays opt-in: None leaves the guard off. When a
        # budget is wanted but unspecified, derive it from the model window.
        self._context_token_budget = context_token_budget

        # Retrieval tool definitions — exposed as _tool_defs for introspection.
        self._tool_defs: list[ToolDefinition] = []
        if resources.structured_schemas:
            self._tool_defs.append(_STRUCTURED_LOOKUP_DEF)
        if resources.vector_schemas:
            self._tool_defs.append(_VECTOR_SEARCH_DEF)
        if resources.document_store is not None and app_id is not None:
            self._tool_defs.append(_READ_DOCUMENT_DEF)
        if mem.long_term is not None and enable_memory_lookup:
            self._tool_defs.append(_MEMORY_LOOKUP_DEF)

    # ------------------------------------------------------------------
    # Direct collection access (bypasses the agent loop)
    # ------------------------------------------------------------------

    async def query_collection(
        self,
        collection: str,
        filters: list[Filter] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Query a structured collection directly, bypassing the agent loop.

        Backs the direct collection-query endpoint. Raises ``RuntimeError`` when
        no structured store is configured.
        """
        if self._structured_store is None:
            raise RuntimeError("structured store not configured")
        return await self._structured_store.query(collection, filters or None, fields or None)

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

    def build_system_prompt(self, base_prompt: str, skill=None, workdir: str | None = None) -> str:
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
            # Both dirs are also exported as env vars to shell/python (see _tool_env),
            # so scripts and outputs resolve deterministically without the model
            # hand-building paths — hence the terse pointers rather than prose rules.
            path_lines: list[str] = []
            if base_dir:
                path_lines.append(
                    f"- Skill directory `$COGBASE_SKILL_DIR` (`{base_dir}`): run bundled "
                    "scripts as `$COGBASE_SKILL_DIR/<script>`."
                )
            if workdir:
                path_lines.append(
                    f"- Working directory `$COGBASE_WORKDIR` (`{workdir}`): the cwd of every "
                    "shell/python command and where fetched files land. Write outputs here "
                    "(relative paths work); it persists across turns, so reuse files already present."
                )
            paths_block = ("Paths:\n" + "\n".join(path_lines) + "\n\n") if path_lines else ""
            paths_block += self._workdir_listing_block(workdir)
            skill_section = (
                f"## Active Skill: {skill.name}\n\n"
                + paths_block
                + metadata_block
                + "Follow the skill's instructions below to complete the user's request. "
                "Use the `shell` tool to run any commands it suggests.\n\n"
                + skill.raw_markdown
            )
            parts.append(skill_section)

        return "\n\n".join(parts)

    @staticmethod
    def _workdir_listing_block(workdir: str | None) -> str:
        """Render the files already present in *workdir* as a prompt block.

        The workdir persists across turns (per app+session), so a follow-up turn
        inherits the prior turn's materialized state — the fetched original, the
        review file, intermediate JSON, prior redline/final drafts. The model
        can't ``ls`` before it plans, so surface that inventory in the system
        prompt: a refine turn should reuse ``review.json`` and the fetched
        original in place rather than re-fetching and re-editing from scratch.
        Empty (fresh workdir) or on any I/O error, returns "".
        """
        if not workdir:
            return ""
        entries: list[str] = []
        try:
            for root, _dirs, files in os.walk(workdir):
                for fname in files:
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, workdir)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    entries.append(f"- `{rel}` ({size} bytes)")
        except OSError:
            return ""
        if not entries:
            return ""
        entries.sort()
        # Cap to keep the prompt bounded on long, file-heavy sessions.
        capped = entries[:50]
        if len(entries) > 50:
            capped.append(f"- …and {len(entries) - 50} more")
        return (
            "Files already in the working directory (persisted from earlier turns — "
            "reuse these in place; do not re-fetch or regenerate what is already here):\n"
            + "\n".join(capped)
            + "\n\n"
        )

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
                         it instead of ``history``.  Also keys the episodic log.
        """
        skills = self._skills
        memory_active = self._short_term is not None and session_id is not None
        episodic_on = self._episodic is not None and session_id is not None
        long_term_on = self._long_term is not None

        logger.info("[runner] start, session=%s question=%s", session_id, user_input)

        # Slot 0 is reserved for the system prompt; updated each iteration.
        messages: list[ChatMessage] = [{"role": "system", "content": ""}]
        context: list[ChatMessage] = []

        if episodic_on:
            # App attribution for every event this turn; idempotent per turn.
            # Recorded before build_context so the turn's user_message is in the
            # log family before the session's thread is (re)assembled.
            self._episodic.bind_app(session_id, app_id=self._app_id)
            await self._episodic_record_user_message(session_id, user_input)

        if memory_active:
            # Server-side session context — projected from the episodic log —
            # replaces caller-passed history.  The current input is threaded in
            # by build_context (its log record is still buffered, not yet flushed).
            context = await self._short_term.build_context(
                session_id=session_id,
                current_user_message=user_input,
            )
            messages.extend(context)
        else:
            messages.extend(history or [])
            messages.append({"role": "user", "content": user_input})

        all_records: list[dict] = []
        all_chunks: list[Chunk] = []
        all_slices: list[DocumentSlice] = []
        # Files persisted via save_artifact this turn; their download links are
        # appended to the final answer so a downloadable file always surfaces one.
        saved_artifacts: list[ArtifactRef] = []
        # Each injected memory block (the recall block below, plus any memory_lookup
        # results during the loop) is one citable passage keyed by its block id.
        # Only the blocks the answer cites are surfaced on the final QueryResult, so
        # the caller sees the memory the answer actually drew on — see
        # _cited_block_memories.
        memory_blocks: dict[str, list[LongTermRecord]] = {}

        # Long-term recall: relevant curated memories injected as a system
        # block marked memory-derived, so the evidence policy keeps them distinct
        # from document-backed claims.  Placed right after slot 0 (the system
        # prompt is written there below) and before the conversation turns.
        if long_term_on:
            # Recall against the conversation tail, not the bare input: a short
            # follow-up ("what about the second one?") carries no retrievable
            # signal on its own.
            prior = context if memory_active else (history or [])
            recall_query = self._compose_recall_query(user_input, prior)
            block_id = f"memory-{len(memory_blocks) + 1}"
            memory_block, recalled = await self._recall_memory_block(recall_query, block_id)
            if memory_block:
                messages.insert(1, {"role": "system", "content": memory_block})
                memory_blocks[block_id] = recalled

        total_input_tokens: int = 0
        total_output_tokens: int = 0

        # Select skill once before the loop. Support switching skill in the for loop when needed.
        # When memory is active the assembled session context is the source of truth,
        # so route against it rather than the (ignored) caller-passed history.
        current_skill = None
        if skills:
            skill_history = context if memory_active else history
            current_skill = await self.select(user_input, skill_history)
            if current_skill is not None:
                logger.info("[runner] active skill → '%s'", current_skill.name)
                yield f"Using skill: {current_skill.name}..."

        # Skills are the only tools that touch the filesystem (shell/python and the
        # fetch/save transports), so a scratch dir is only worth creating when one is
        # active — a pure-retrieval turn leaves no litter.
        workdir = self._session_workdir(session_id) if current_skill is not None else None
        if workdir:
            logger.info("[runner] session workdir=%s", workdir)

        messages[0] = {"role": "system", "content": self.build_system_prompt(base_prompt, current_skill, workdir)}

        for _ in range(self._max_calls):
            # --- Budget guard: compact the working list if it has outgrown the
            # context budget (tool outputs accumulated across rounds are not
            # bounded by build_context, which only assembles the initial turn).
            # Safe here: every assistant tool_call has been answered by its tool
            # message, so collapsing into a summary strands no dangling call.
            if self._context_token_budget and estimate_messages_tokens(messages) > self._context_token_budget:
                logger.info(
                    "[runner] in-loop compaction: working list exceeds budget %d",
                    self._context_token_budget,
                )
                messages = await self.compact_messages(messages[0]["content"], messages[1:])

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
                # Cited ids come from the model's own prose; extract them before
                # appending download links so a link's [filename] can't be mistaken
                # for an evidence citation.
                cited_ids = _extract_cited_ids(answer)
                if saved_artifacts:
                    answer = _append_download_links(answer, saved_artifacts)
                cited_chunks = _filter_cited_chunks(all_chunks, cited_ids)
                cited_slices = _filter_cited_slices(all_slices, cited_ids)
                cited_memories = _cited_block_memories(memory_blocks, cited_ids)
                if episodic_on:
                    await self._episodic_final_answer(
                        session_id,
                        answer,
                        _serialize_references(
                            all_records, cited_chunks, cited_slices, cited_memories
                        ),
                    )
                yield QueryResult(
                    answer=answer,
                    structured_records=all_records,
                    chunks=cited_chunks,
                    document_slices=cited_slices,
                    memories=cited_memories,
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

                call_ref: EventRef | None = None
                if episodic_on:
                    call_ref = await self._episodic_tool_call(session_id, tc["id"], name, inputs)
                t0 = time.monotonic()

                if name == "structured_lookup":
                    records, tool_output, passthrough = await self._run_structured_lookup(inputs)
                    all_records.extend(records)
                    if passthrough:
                        if episodic_on:
                            await self._episodic_tool_result(session_id, tc["id"], tool_output, t0, call_ref)
                            await self._episodic_final_answer(
                                session_id,
                                tool_output,
                                _serialize_references(all_records, all_chunks, all_slices, []),
                            )
                        yield QueryResult(
                            answer=tool_output,
                            structured_records=all_records,
                            chunks=all_chunks,
                            document_slices=all_slices,
                            # Passthrough returns the records directly with no LLM
                            # synthesis, so no memory was cited / used.
                            memories=[],
                            passthrough=True,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                        )
                        return
                elif name == "vector_search":
                    seen_chunk_ids = {c.chunk_id for c in all_chunks}
                    chunks, tool_output = await self._run_vector_search(inputs, exclude_ids=seen_chunk_ids, default_top_k=top_k)
                    all_chunks.extend(chunks)
                elif name == "read_document":
                    doc_slice, tool_output = await self._run_read_document(inputs)
                    if doc_slice is not None:
                        all_slices.append(doc_slice)
                elif name == "memory_lookup":
                    block_id = f"memory-{len(memory_blocks) + 1}"
                    tool_output, looked_up = await self._run_memory_lookup(inputs, block_id)
                    if looked_up:
                        memory_blocks[block_id] = looked_up
                elif name == "save_artifact":
                    artifact, tool_output = await self._run_save_artifact(inputs)
                    if artifact is not None:
                        saved_artifacts.append(artifact)
                else:
                    tool_output = await self._execute_tool(name, inputs, current_skill, workdir)
                    logger.info("[runner] execute_tool done: %s", tool_output[:300])

                if episodic_on:
                    await self._episodic_tool_result(session_id, tc["id"], tool_output, t0, call_ref)

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
        # limit the number of references to avoid large context; the canned answer
        # cites nothing, so no memory was used and none is returned.
        capped_records = all_records[:2]
        capped_chunks = all_chunks[:2]
        capped_slices = all_slices[:2]
        if episodic_on:
            await self._episodic_final_answer(
                session_id,
                answer,
                _serialize_references(capped_records, capped_chunks, capped_slices, []),
            )
        yield QueryResult(
            answer=answer,
            structured_records=capped_records,
            chunks=capped_chunks,
            document_slices=capped_slices,
            memories=[],
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

    async def _execute_tool(self, name: str, inputs: dict, skill=None, workdir: str | None = None) -> str:
        if name in self._system_tools:
            try:
                result = self._system_tools[name].handler(inputs)
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    return await result
                return result  # type: ignore[return-value]
            except Exception as e:
                return f"Tool error ({name}): {e}"

        if name == "fetch_document":
            return await self._run_fetch_document(inputs, workdir)
        if name == "fetch_artifact":
            return await self._run_fetch_artifact(inputs, workdir)
        if name == "delete_artifact":
            return await self._run_delete_artifact(inputs)

        env = self._tool_env(skill, workdir)
        if name == "python":
            return await self._run_python(inputs.get("code", ""), env, cwd=workdir)
        if name == "shell":
            return await self._run_shell(inputs.get("command", ""), env, cwd=workdir)
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
            estimated_tokens = estimate_tokens(json_str)
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
        if self._document_store is None or self._app_id is None:
            return None, "read_document is unavailable (no document store configured)"

        doc_id = str(inputs.get("doc_id", ""))
        if not doc_id:
            return None, "read_document error: doc_id is required"

        offset = max(0, int(inputs.get("offset") or 0))
        length = min(int(inputs.get("length") or 2000), 10000)

        try:
            text = await self._document_store.load(self._app_id, doc_id)
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

    def _session_workdir(self, session_id: str | None) -> str:
        """Return (creating if needed) this session's local scratch directory.

        Layout: ``<work_root>/<app_id>/<session_id>/``.  Deterministic per
        (app, session) so a retry or a follow-up turn lands in the same place —
        the model can reference known paths and never has to search the
        filesystem.  With no ``session_id`` (stateless query) a random bucket is
        used: it still keeps a turn's files together, it just isn't resumable.
        See ``_DEFAULT_WORK_ROOT`` for the cross-node contract.
        """
        app = re.sub(r"[^\w.-]", "_", self._app_id or "app")
        sess = (
            re.sub(r"[^\w.-]", "_", session_id)
            if session_id
            else f"nosession_{uuid.uuid4().hex[:8]}"
        )
        path = os.path.join(self._work_root, app, sess)
        os.makedirs(path, exist_ok=True)
        return path

    async def _run_fetch_document(self, inputs: dict, workdir: str | None = None) -> str:
        """Materialize a stored original file to a local path for skill scripts.

        General file-transport primitive (the inbound half; ``save_artifact`` is
        the outbound half). Uploads are stored at ``originals/{doc_id}{suffix}``;
        this assumes the ``.docx`` suffix (the docx-editing case) and falls back
        to a suffix-free key, so a skill can unpack/edit the raw file rather than
        only its extracted text.
        """
        if self._document_store is None or self._app_id is None:
            return "fetch_document is unavailable (no document store configured)"

        doc_id = str(inputs.get("doc_id", ""))
        if not doc_id:
            return "fetch_document error: doc_id is required"

        data: bytes | None = None
        suffix = ".docx"
        for key in (f"originals/{doc_id}.docx", f"originals/{doc_id}"):
            try:
                data = await self._document_store.load_bytes(self._app_id, key)
                suffix = os.path.splitext(key)[1]
                break
            except (KeyError, NotImplementedError):
                continue
        if data is None:
            return f"fetch_document error: no original file for '{doc_id}'"

        path = self._materialize(data, workdir, subdir="originals", name=f"{doc_id}{suffix}", suffix=suffix)
        logger.info("[runner] fetch_document doc_id=%s bytes=%d path=%s", doc_id, len(data), path)
        return f"Fetched document '{doc_id}' ({len(data)} bytes) to {path}"

    def _artifact_download_path(self, artifact_id: str) -> str:
        """App-scoped download path served by the generated-artifact endpoint.

        Mirrors ``GET /applications/{app_id}/documents/{artifact_id}/download``.
        Keyed by the stable ``app_id`` (not the mutable client-facing name), so a
        link handed to the user keeps resolving after the application is renamed.
        """
        return f"/applications/{self._app_id}/documents/{artifact_id}/download"

    async def _run_save_artifact(self, inputs: dict) -> tuple[ArtifactRef | None, str]:
        """Persist a skill-produced file to the document store for later download.

        Stores under ``generated/{artifact_id}`` (``artifact_id`` keeps the
        original extension), which the ``GET .../documents/{artifact_id}/download``
        endpoint serves verbatim.  Returns the ``ArtifactRef`` (so the runner can
        append its markdown link to the final answer) and the tool output, which
        carries that same ready-to-use link for the model to reproduce.  Returns
        ``(None, <error>)`` on any failure.
        """
        if self._document_store is None or self._app_id is None:
            return None, "save_artifact is unavailable (no document store configured)"

        path = str(inputs.get("path", ""))
        if not path or not os.path.exists(path):
            return None, f"save_artifact error: file not found at '{path}'"

        filename = str(inputs.get("filename") or os.path.basename(path))
        stem, ext = os.path.splitext(filename)
        safe_stem = re.sub(r"[^\w\-]", "_", stem) or "artifact"
        artifact_id = f"{safe_stem}__{uuid.uuid4().hex[:8]}{ext}"

        data = await asyncio.to_thread(lambda: open(path, "rb").read())
        try:
            await self._document_store.save_bytes(self._app_id, f"generated/{artifact_id}", data)
        except NotImplementedError:
            return None, "save_artifact error: the document store does not support binary artifacts"

        artifact = ArtifactRef(
            artifact_id=artifact_id,
            filename=filename,
            download_path=self._artifact_download_path(artifact_id),
        )
        logger.info("[runner] save_artifact artifact_id=%s bytes=%d", artifact_id, len(data))
        return artifact, (
            f"Saved artifact '{artifact_id}' ({len(data)} bytes). "
            f"Include this download link in your answer: {artifact.markdown_link}"
        )

    def _materialize(
        self, data: bytes, workdir: str | None, *, subdir: str | None, name: str, suffix: str
    ) -> str:
        """Write *data* to a deterministic path under *workdir* and return it.

        Files land at ``<workdir>[/<subdir>]/<name>`` (``name`` sanitized to a
        single path segment) so the location is predictable across turns. A
        ``workdir`` is guaranteed here: the fetch transports that reach this
        method are ``_BASE_TOOLS``, exposed only when a skill is active — the
        same condition under which the turn's workdir is created — so a missing
        one is a wiring bug, not a state to paper over with a scattered tempfile.
        """
        assert workdir, "_materialize requires a session workdir (fetch tools run only with a skill active)"
        safe_name = re.sub(r"[^\w.-]", "_", name) or f"file{suffix}"
        dest_dir = os.path.join(workdir, subdir) if subdir else workdir
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, safe_name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    async def _run_fetch_artifact(self, inputs: dict, workdir: str | None = None) -> str:
        """Materialize a previously saved artifact back to a local path.

        The inbound counterpart of ``save_artifact``: artifacts live at
        ``generated/{artifact_id}`` (the ``artifact_id`` keeps its extension), so a
        skill can reload working state produced in an earlier turn — patch it and
        ``save_artifact`` a fresh copy.
        """
        if self._document_store is None or self._app_id is None:
            return "fetch_artifact is unavailable (no document store configured)"

        artifact_id = str(inputs.get("artifact_id", ""))
        if not artifact_id:
            return "fetch_artifact error: artifact_id is required"

        try:
            data = await self._document_store.load_bytes(self._app_id, f"generated/{artifact_id}")
        except (KeyError, NotImplementedError):
            return f"fetch_artifact error: no artifact '{artifact_id}'"

        suffix = os.path.splitext(artifact_id)[1]
        path = self._materialize(data, workdir, subdir=None, name=artifact_id, suffix=suffix)
        logger.info("[runner] fetch_artifact artifact_id=%s bytes=%d path=%s", artifact_id, len(data), path)
        return f"Fetched artifact '{artifact_id}' ({len(data)} bytes) to {path}"

    async def _run_delete_artifact(self, inputs: dict) -> str:
        """Delete a previously saved artifact. Idempotent (missing == success)."""
        if self._document_store is None or self._app_id is None:
            return "delete_artifact is unavailable (no document store configured)"

        artifact_id = str(inputs.get("artifact_id", ""))
        if not artifact_id:
            return "delete_artifact error: artifact_id is required"

        try:
            await self._document_store.delete(self._app_id, f"generated/{artifact_id}")
        except KeyError:
            pass  # already gone — deletion is idempotent
        except NotImplementedError:
            return "delete_artifact error: the document store does not support deletion"
        logger.info("[runner] delete_artifact artifact_id=%s", artifact_id)
        return f"Deleted artifact '{artifact_id}'."

    async def _run_python(self, code: str, env: dict, cwd: str | None = None) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(_PY_PREAMBLE + code)
                tmp = f.name
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, tmp,
                    env=env,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                return await self._read_proc(proc)
            finally:
                os.unlink(tmp)
        except Exception as e:
            return f"Python error: {e}"

    async def _run_shell(self, command: str, env: dict, cwd: str | None = None) -> str:
        # A whole-filesystem scan only happens when the model has lost a path; it
        # has no legitimate use here and would burn the entire tool-timeout budget
        # before being killed. Refuse fast and point at the deterministic dirs.
        if _ROOT_FS_SCAN.search(command):
            return (
                "Refusing to scan the whole filesystem. The skill's bundled scripts are in "
                "$COGBASE_SKILL_DIR and your working files are in $COGBASE_WORKDIR — reference "
                "those (or a path already returned by a tool) instead of searching."
            )
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                env=env,
                cwd=cwd,
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
    def _tool_env(skill, workdir: str | None = None) -> dict:
        env = os.environ.copy()
        if skill and skill.site_packages:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{skill.site_packages}:{existing}" if existing else skill.site_packages
        # The skill's own directory, so the model can invoke bundled scripts as
        # `$COGBASE_SKILL_DIR/<script>` instead of hand-splicing an absolute path
        # from the prompt (which can drift or be hallucinated).
        if skill and skill.source_path:
            env["COGBASE_SKILL_DIR"] = str(skill.source_path.parent)
        if workdir:
            env["COGBASE_WORKDIR"] = workdir
        return env


    # ------------------------------------------------------------------
    # Long-term memory recall
    # ------------------------------------------------------------------

    async def _run_memory_lookup(
        self, inputs: dict, block_id: str = "memory-1"
    ) -> tuple[str, list[LongTermRecord]]:
        """Execute the memory_lookup tool against the long-term store.

        Returns the formatted tool output and the records it surfaced (empty on
        error or no match), so the caller can attach them to the QueryResult.
        The result is one citable block keyed by *block_id*; ``status=active``
        filtering is enforced inside ``LongTermMemory.lookup``, never from
        LLM-supplied arguments.
        """
        if self._long_term is None:
            return "memory_lookup is unavailable (no long-term memory configured)", []

        kind: MemoryKind | None = None
        raw_kind = inputs.get("kind")
        if raw_kind:
            try:
                kind = MemoryKind(raw_kind)
            except ValueError:
                return f"memory_lookup error: unknown kind '{raw_kind}'", []
        query = str(inputs.get("query") or "") or None
        entities = [str(e) for e in (inputs.get("entities") or [])]
        limit = min(int(inputs.get("limit") or 10), 20)
        if not query and not kind and not entities:
            return "memory_lookup error: provide at least one of query / kind / entities", []

        try:
            memories = await self._long_term.lookup(
                query=query, kind=kind, entities=entities, limit=limit
            )
        except Exception as exc:
            logger.exception("[runner] memory_lookup.error")
            return f"memory_lookup error: {exc}", []

        logger.info("[runner] memory_lookup.result memories=%d", len(memories))
        if not memories:
            return "(no matching memories)", []
        block = _format_memory_block(
            "Memories (recalled from long-term memory).", block_id, memories
        )
        return block, memories

    @staticmethod
    def _compose_recall_query(
        user_input: str, prior_messages: list[ChatMessage]
    ) -> str:
        """Build the recall query from the current input plus the last exchange.

        A short follow-up ("what about the second one?") embeds poorly on its
        own, so the previous user message and assistant answer are prepended,
        truncated so a long answer doesn't drown the follow-up's signal. The
        answer keeps its head, where the topic statement usually lives. With no
        prior exchange this is just ``user_input``.
        """
        # build_context threads the current input in as the trailing user
        # message; drop it so the walk below only sees completed turns.
        if (
            prior_messages
            and prior_messages[-1].get("role") == "user"
            and prior_messages[-1].get("content") == user_input
        ):
            prior_messages = prior_messages[:-1]

        prev_user: str | None = None
        prev_answer: str | None = None
        for msg in reversed(prior_messages):
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue  # tool results / tool-call-only assistant messages
            role = msg.get("role")
            if role == "assistant" and prev_answer is None:
                prev_answer = content
            elif role == "user" and prev_user is None:
                prev_user = content
            if prev_user is not None and prev_answer is not None:
                break

        parts: list[str] = []
        if prev_user:
            parts.append(prev_user[:300])
        if prev_answer:
            parts.append(prev_answer[:500])
        parts.append(user_input)
        return "\n".join(parts)

    async def _recall_memory_block(
        self, query: str, block_id: str = "memory-1"
    ) -> tuple[str | None, list[LongTermRecord]]:
        """Recall memories relevant to *query* and format them as a system block.

        Returns the block (or ``None`` to inject nothing) together with the
        recalled records, so the caller can surface them on the QueryResult.
        The block is one citable passage keyed by *block_id*.  Yields ``(None, [])``
        when nothing is recalled or recall fails — long-term recall is an
        enrichment, never allowed to break a turn.
        """
        try:
            memories = await self._long_term.recall(query=query)
        except Exception:
            logger.warning("[runner] long-term recall failed", exc_info=True)
            return None, []
        if not memories:
            return None, []
        block = _format_memory_block(
            "Relevant long-term memory about the user, topic, or entity (recalled).",
            block_id,
            memories,
        )
        logger.info("[runner] long-term recall injected %d memories", len(memories))
        return block, memories

    # ------------------------------------------------------------------
    # Episodic memory recording
    #
    # Recording buffers events in EpisodicMemory's per-session cache (cheap,
    # in-memory); the durable append happens at the turn boundary in
    # _episodic_final_answer's flush.  Every method is a no-op when episodic
    # memory is not wired.  Failure handling is *tiered* (see
    # docs/episodic-memory.md#episodicmemory-writer):
    #
    #   - Best-effort (tool_called / tool_result): swallow failures — losing one
    #     costs only analytics, never continuity, and must never break a turn.
    #   - Continuity-critical (user_message / final_answer + the turn flush):
    #     propagate failures.  These are what a later rehydrate reconstructs the
    #     thread from; if one is silently dropped, a follow-up that fails over to
    #     another node rehydrates a log missing a turn the user already saw.  So
    #     the turn must NOT be acknowledged complete unless they are durable.
    # ------------------------------------------------------------------

    async def _episodic_record_user_message(self, session_id: str, content: str) -> None:
        # Continuity-critical: buffer must not fail silently (it is flushed,
        # alongside final_answer, at the turn boundary).  Let exceptions propagate.
        if self._episodic is None:
            return
        await self._episodic.record_user_message(session_id=session_id, content=content)

    async def _episodic_tool_call(
        self, session_id: str, tool_call_id: str, name: str, inputs: dict
    ) -> EventRef | None:
        if self._episodic is None:
            return None
        try:
            return await self._episodic.record_tool_call(
                session_id=session_id, tool_call_id=tool_call_id, name=name, arguments=inputs
            )
        except Exception:
            logger.warning("[runner] episodic tool_called record failed", exc_info=True)
            return None

    async def _episodic_tool_result(
        self,
        session_id: str,
        tool_call_id: str,
        output: str,
        t0: float,
        call_ref: EventRef | None,
    ) -> None:
        if self._episodic is None:
            return
        try:
            await self._episodic.record_tool_result(
                session_id=session_id,
                tool_call_id=tool_call_id,
                result=output,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                parent_event_id=call_ref,
            )
        except Exception:
            logger.warning("[runner] episodic tool_result record failed", exc_info=True)

    async def _episodic_final_answer(
        self, session_id: str, answer: str, references: dict | None = None
    ) -> None:
        """Record the canonical assistant turn and durably flush the turn's events.

        ``final_answer`` is continuity-critical: rehydrate reconstructs the thread
        from it (and from the ``user_message`` flushed in the same batch), so the
        append must land before the turn is acknowledged complete.  The flush is
        retried a bounded number of times — the buffer is retained between
        attempts, so a retry re-appends the same batch — and a persistent failure
        is surfaced (raised), not swallowed: better to fail the turn than to
        acknowledge an answer that a future session cannot rehydrate.
        """
        if self._episodic is None:
            return
        await self._episodic.record_final_answer(
            session_id=session_id, answer=answer, references=references
        )
        last_exc: Exception | None = None
        for attempt in range(_CONTINUITY_FLUSH_RETRIES):
            try:
                await self._episodic.flush(session_id)
                return
            except LogFenced:
                # Fatal, not transient: another writer owns this session, so the
                # flush dropped our buffer.  Retrying would no-op the empty buffer
                # and falsely acknowledge a turn whose answer never landed — fail
                # the turn immediately instead.
                logger.error(
                    "[runner] episodic flush fenced for session=%s; another writer "
                    "owns it — failing the turn", session_id,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — retried, then re-raised below
                last_exc = exc
                logger.warning(
                    "[runner] episodic flush attempt %d/%d failed",
                    attempt + 1, _CONTINUITY_FLUSH_RETRIES, exc_info=True,
                )
                await asyncio.sleep(0.05 * (attempt + 1))
        logger.error(
            "[runner] episodic continuity flush failed after %d attempts; failing the turn",
            _CONTINUITY_FLUSH_RETRIES,
        )
        assert last_exc is not None
        raise last_exc

    async def compact_messages(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Summarise *messages* into a minimal list to recover from context overflow.

        Delegates to ``cogbase.llms.summarization.summarize_text``, which preserves
        the full transcript (no per-message truncation): long transcripts are
        split into budget-sized chunks, summarised, and merged recursively.
        """
        transcript = "\n".join(render_message(m) for m in messages)
        summary = await summarize_text(
            self._llm,
            transcript,
            chunk_tokens=summarise_chunk_tokens(self._llm),
            compress_prompt=WORKING_CONTEXT_PROMPT,
        ) or "(empty summary)"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compacted context:\n\n{summary}\n\nContinue from this point."},
        ]
