# Memory Layer

The memory layer gives CogBase continuity across a query, a session, and repeated use of an application. It should be implemented as a first-class service above the existing store abstractions, not as special logic buried inside the query runner.

```text
Query/API/Skills
    |
MemoryManager
    |
    +-- ShortTermMemory    session-local context assembly
    +-- EpisodicMemory     append-only event log of conversations, tools, and results
    +-- LongTermMemory     promoted, durable facts, preferences, and learned patterns
    |
Existing stores
    +-- StructuredStoreBase
    +-- VectorStoreBase
```

## Goals

- Preserve useful context within a session without overloading the LLM context window.
- Record query behavior in a structured form that can be inspected, replayed, and mined.
- Accumulate durable, provenance-backed knowledge across sessions.
- Feed adaptive evolution with evidence from real usage.
- Keep application evidence grounded in ingested documents and structured records.

## Memory tiers

| Tier | Scope | Persistence | Purpose |
|---|---|---|---|
| Short-term | Session | Runtime / expiring | Current transcript, retrieved chunks, tool outputs, compacted context, token-budget decisions |
| Episodic | User / session / app | Durable append-only log | Queries, answers, tool calls, retrieval results, feedback, and action traces |
| Long-term | User / project / org / app | Durable curated store | Confirmed facts, preferences, successful retrieval plans, common intent patterns |

## Short-term memory

Short-term memory owns the working context for an active query or session. It should not become a source of truth. Its job is to decide what belongs in the next LLM call.

Short-term memory should track:

- the active session identity and metadata
- recent user and assistant messages
- retrieved structured records, chunks, and document slices
- tool outputs that are still relevant to the current task
- compacted summaries when the raw transcript no longer fits
- token-budget and context-selection decisions

Short-term memory is an in-memory working cache; it needs no durable store of its
own — the episodic log is its persistence (see
[episodic-memory.md](episodic-memory.md#short-term-memory-rides-on-the-same-log)).
On a cold start or cache miss it rehydrates from the log tail. For multi-worker
deployments, route a session's requests to the same process by consistent-hashing
`session_id` so the cache is reused; affinity is an optimization, not a
correctness requirement, since any process can rehydrate from the log.

## Episodic memory

Episodic memory should be an append-only event log. It records what happened, not what the system believes forever. This makes it suitable for debugging, replay, analytics, and adaptive evolution.

The storage design — an event-sourced, per-session append-only NDJSON log in the
document store as the single source of truth, with the cross-session structured
projection deferred until a concrete consumer (the adaptive evolution engine) is
designed — is detailed in [episodic-memory.md](episodic-memory.md). Short-term
memory rides on the same log: it is an in-memory projection over the log tail and
needs no separate durable store.

Core event types:

- `session_started`
- `user_message`
- `assistant_message`
- `tool_called`
- `tool_result`
- `retrieval_result`
- `final_answer`
- `feedback`
- `session_compacted`

Events should include enough metadata to support filtering and attribution:

- `app_name`
- `user_id`
- `session_id`
- `event_id`
- `event_type`
- `created_at`
- `parent_event_id`
- `payload`
- `source`
- `latency_ms`
- `error`

Episodic memory feeds two kinds of consumer. Offline distillation reads each
session's log whole to extract durable facts and preferences into long-term memory
(the mem0-style path). Adaptive evolution mines *across* sessions for low vector
scores, repeated null answers, recurring tool chains, and query classes the current
app configuration does not cover; because it is the only cross-session consumer,
its supporting structured projection is deferred until the engine is designed (see
[episodic-memory.md](episodic-memory.md)).

## Long-term memory

Long-term memory should contain curated or promoted knowledge, not raw chat history. It stores durable information that is useful across sessions.

Examples:

- user preferences
- organization or project facts
- confirmed corrections
- successful retrieval plans
- common intent-to-tool-chain patterns
- domain-specific learned facts

Every long-term memory record should carry provenance:

- `memory_id`
- `scope`
- `kind`
- `content`
- `confidence`
- `status`
- `source_event_ids`
- `created_at`
- `updated_at`
- optional `expires_at`

Promotion from episodic to long-term memory should be confidence-aware. Some memories can be promoted automatically, but memories that affect app behavior or store user, project, or organization facts should carry source event IDs and a review status.

## Scoping

Memory must be scoped explicitly from the start. Recommended scopes:

- `session`
- `user`
- `app`
- `project`
- `organization`
- `global`

The memory manager should never return broader-scope memories unless the caller is authorized to see them. This is especially important for hosted or multi-tenant deployments.

## Storage strategy

Use the existing stores first.

Short-term memory:

- an in-memory working cache; durability comes from the episodic log, not a
  separate store
- rehydrate on cold start or cache miss by reading the log tail (from the last
  `session_compacted` marker onward)
- route a session's requests to the same process by consistent-hashing
  `session_id` so the cache is reused; affinity is a cache optimization, not a
  correctness requirement

Episodic memory (event-sourced — see [episodic-memory.md](episodic-memory.md)):

- canonical log: document store, one append-only NDJSON object per session — the
  single source of truth
- this requires `append` / `load_lines` support on the document store
- the cross-session structured projection (one lean row per event) is deferred
  until the adaptive evolution engine is designed and can specify what to index;
  every other consumer reads the log per-session

Long-term memory:

- structured store for canonical records
- vector store for semantic recall over memory text or summaries
- indexes on `scope`, `kind`, `status`, `confidence`, and timestamps

## Public interface

The memory layer should expose a small service interface. A representative shape:

```python
class MemoryManager:
    async def start_session(self, *, app_name: str, user_id: str | None = None, metadata: dict | None = None) -> str: ...
    async def record_user_message(self, *, session_id: str, content: str, metadata: dict | None = None) -> None: ...
    async def record_tool_call(self, *, session_id: str, name: str, arguments: dict, metadata: dict | None = None) -> str: ...
    async def record_tool_result(self, *, session_id: str, tool_call_id: str, result: dict | str, metadata: dict | None = None) -> None: ...
    async def record_final_answer(self, *, session_id: str, answer: str, metadata: dict | None = None) -> None: ...
    async def build_context(self, *, session_id: str, query: str, token_budget: int) -> list[dict]: ...
    async def recall(self, *, query: str, scope: dict, limit: int = 10) -> list[dict]: ...
    async def promote(self, *, source_event_ids: list[str], kind: str, content: str, scope: dict, confidence: float) -> str: ...
```

`QueryRunner` should accept `memory: MemoryManager | None`. When present, it should:

1. record the user message
2. ask memory for relevant context before the first LLM call
3. record each tool call and result
4. record retrieval results separately from generic tool output
5. record the final answer and cited evidence

## Evidence policy

Long-term memory must not silently override document evidence. In CogBase, ingested documents, extracted structured records, and workflow outputs remain the primary evidence layer.

Memory may help with continuity, routing, preferences, and learned retrieval paths. If a long-term memory contains a factual claim, answers should distinguish it from document-backed evidence unless the memory has provenance to source documents or accepted structured records.

## Integration points

Initial integration points:

- `cogbase/core/query_runner.py` records messages, tool calls, retrieval results, and final answers.
- API query requests accept optional `session_id`, `user_id`, and memory scope metadata.
- Adaptive evolution reads episodic memory rather than query runner internals.
- Skills receive session identity and can emit memory events through the manager.

## Build order

1. Add `append` / `load_lines` to the document store; add memory models and an
   in-memory `MemoryManager`.
2. Instrument `QueryRunner` to emit episodic events to the per-session log.
3. Add short-term context assembly and compaction over the log tail (rehydrate on
   cache miss).
4. Add long-term memory: distill durable facts and preferences from per-session
   logs (offline), with semantic recall.
5. Add promotion rules from episodic to long-term.
6. When the adaptive evolution engine is designed, add the cross-session
   structured projection it needs and feed episodic signals into it.

This sequence provides useful behavior early while keeping the event model stable before adding more complex recall and promotion behavior.
