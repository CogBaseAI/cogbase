# Episodic Memory — Storage Design

Episodic memory is the append-only event log of what happened during queries:
messages, tool calls, retrievals, answers, and feedback (see the event model in
[memory.md](memory.md#episodic-memory)). This document records the storage
design: a single per-session append-only NDJSON log in the document store is the
source of truth, short-term memory rides on that same log, and the cross-session
structured projection is deferred until a concrete consumer needs it.

## The write path is a red herring; the read path splits in two

Appending events is trivial in any backend, so the append-friendly write path
does not decide the storage. What decides it is how episodic memory is **read
back**, and the read workloads split along two axes — online vs. offline, and
per-session vs. cross-session:

| Workload | When | Access pattern | Wants |
|---|---|---|---|
| Short-term rehydrate | online (request path) | last *N* events of one session | tail of one log |
| Session replay / debug | offline | all events of *one* session, in order | one cheap whole-log fetch, ordering |
| Distillation → long-term | offline | whole log of *one* session | whole-log fetch |
| Adaptive evolution / gap detector | offline | *cross-session* scan filtered by `app_name`, `event_type`, `created_at`; aggregate (count null answers, group recurring tool chains, find low retrieval scores) | indexes, `WHERE` / `GROUP BY` |

The crucial observation: **every consumer except adaptive evolution reads the log
per-session** — a tail, or one whole log. A single append-only object per session
serves all of those ideally. Only the gap detector wants cross-session, filtered,
aggregate reads — the one workload a file-per-session serves *worst* (it would
scan every object in the store).

## Decision: the per-session log is the source of truth; defer the structured projection

We do **not** build a row-per-event structured table. A structured log table
would be abandoned in the long term: large payloads (full tool outputs, retrieved
chunks, answer text) bloat the table and its indexes, the table grows unbounded,
and replay degrades to an `ORDER BY` scan.

Instead, the document store holds the canonical log — one append-only NDJSON
object per session, one JSON event per line — and **that is the only episodic
storage we build now**. It is immutable, audit-friendly, literally an append-only
log; cheap to archive or expire (delete one object); payloads stay out of any
database. It serves every near-term consumer with a single per-session fetch.

The cross-session structured projection the read-path table calls for exists only
to serve adaptive evolution, and that engine is not yet designed. Building the
projection now would mean guessing its aggregation axes before its consumer
exists, then rebuilding when the guess is wrong. Because the projection is
*derived* from the log, it can be added — and shaped to exactly what the gap
detector needs — at any time with no data loss. So we defer it (see
[Deferred: the structured projection](#deferred-the-structured-projection)).

This keeps the near-term work small: add append/tail reads to the document store,
write events to per-session logs, and read them back per session.

## Document store additions

Episodic memory is the reason the document store grows append support. Today
`DocumentStoreBase.save` overwrites; the log needs ordered, durable appends and
tail reads:

```python
async def append(self, collection: str, doc_id: str, content: str) -> None:
    """Append content to doc_id, creating it if absent. Ordered and durable."""

async def load_lines(
    self, collection: str, doc_id: str, *, tail: int | None = None
) -> list[str]:
    """Read the log back, optionally just the last `tail` lines (for rehydrate)."""
```

Backend implementations:

- **local_fs** — `open(path, "a")` with `O_APPEND` (atomic for line-sized
  writes), via `run_in_executor`.
- **s3** — S3 Express One Zone **directory buckets** support native append
  (offset-based `PutObject`). For standard buckets, fall back to either a small
  append-buffer flushed periodically, or one object per event under a
  `session_id/` prefix (then "append" = new key and "replay" = list + sort the
  prefix).
- **memory** — string concatenation.

NDJSON is the format: `record()` writes `json.dumps(event) + "\n"`. It is
line-oriented (cheap `tail` reads), needs no partial-read parsing, and
interleaves cleanly under a single active writer — one session is effectively
one concurrent writer, so contention is a non-issue.

## Short-term memory rides on the same log

Short-term memory needs no durable store of its own; the episodic log is its
persistence. Short-term is an in-memory working cache during a live session, and
its durable record is the message/compaction events already in the log.

- **Cross-turn continuity needs only the conversational thread** —
  `user_message`, `assistant_message` / `final_answer` — plus compaction
  summaries. It does *not* need the tool-call chain. Tool outputs are intra-turn
  scratch held in RAM for the active agent loop and are never rehydrated: a dead
  turn is re-run, not resumed.
- **Compaction appends a `session_compacted` summary event** rather than
  rewriting history (the log is append-only). Rehydrate then reads from the last
  `session_compacted` marker onward — summary plus subsequent messages — so it is
  bounded, not a whole-log scan.
- On a cold start or cache miss, short-term rebuilds via `load_lines(tail=...)`
  filtered to the events it needs.

(If short-term ever needs a representation that cannot be expressed as appended
events — e.g. rewrite-in-place compaction — it would need its own mutable store.
Append-a-summary compaction avoids that, so it does not today.)

## Session affinity and routing

In production there are multiple processes per node and multiple nodes. To reuse
the in-memory short-term cache, requests for one session should usually reach the
process that holds it.

- **Route by consistent-hashing `session_id`** → node → process (your L7 load
  balancer's ring-hash / maglev mode, or a thin stateless gateway). The same
  session deterministically lands on the same process.
- **Affinity is a cache optimization, not a correctness requirement.** Because
  the log is the durable source of truth, any process can serve any session by
  rehydrating from the tail. Ring changes (scale, restart, failover) cost an
  affected session one cache miss, never an error.

Node-local shared cache:

- With **local_fs** as the document store, processes on a node need **no**
  node-local shared cache: the OS file-system page cache is already shared across
  processes on the host, so log tail reads after a process restart hit warm cache.
- With **s3** as the document store, there is no shared host cache, so a
  node-local shared store may later be worth adding to avoid re-fetching logs
  across processes. Defer it until measured.

Single-writer caveat:

- The NDJSON append assumes one writer per session at a time (one human session
  is effectively one serialized writer). Affinity-as-optimization is safe as long
  as two processes never serve the same session concurrently. If concurrent
  same-session traffic becomes real, prefer the s3 one-object-per-event layout
  (concurrent writers never collide; replay = list + sort the prefix) over a
  distributed lock.

## Deferred: the structured projection

When the adaptive evolution engine is designed, it will need cross-session,
filtered, aggregate reads that the per-session logs serve poorly. At that point
add a **derived** structured projection — rebuilt from the logs, payload-free —
shaped to what the gap detector actually mines. The likely shape, for reference:

| Field | Purpose |
|---|---|
| `event_id` (PK) | point lookup for `source_event_ids` during promotion |
| `app_name` (index) | scope filter |
| `user_id` (index) | scope / attribution |
| `session_id` (index) | locates the log object; replay drill-down |
| `event_type` (index) | gap-detector filter (null answers, tool chains, …) |
| `parent_event_id` | causal chain (tool_result → tool_called) |
| `source` | tool name, e.g. `vector_search` |
| `score` | retrieval score, for low-score mining |
| `latency_ms` | observability |
| `error` | failure mining |
| `created_at` (index) | time-window filter |

Two shape choices to settle then, not now:

- **per-event rows** (maximum query flexibility, point lookup by `event_id`) vs.
  **per-session signal rollups** (one row per session: did it null-answer?
  minimum retrieval score? tool-chain fingerprint? — far smaller, drills into the
  log by `session_id` on demand).
- **whether point lookup by `event_id` is needed at all**: if event ids are made
  self-locating (`session_id` + line), provenance resolution for long-term
  promotion is a log seek, not a structured lookup, and a per-event index may be
  unnecessary.

Because the projection is derived, either shape can be built or rebuilt from the
logs once its consumer is concrete.

## EpisodicMemory writer

A thin durable writer (`cogbase/memory/episodic.py`), async like the rest of the
memory layer. Unlike a cache it never compacts or evicts — it only appends. For
now each `record_*` simply appends an NDJSON line to the session log; the
projection write is added with the evolution engine. Best-effort: a logging
failure must never fail an in-flight query, matching the short-term tier's
philosophy.

```python
class EpisodicMemory:
    def __init__(self, document_store) -> None: ...

    async def record(self, event: MemoryEvent) -> str:
        # append NDJSON line to the session's log object (source of truth)
        ...

    # typed convenience wrappers (return event_id so callers can thread
    # parent_event_id into causal chains):
    async def record_user_message(self, *, session_id, content, **scope) -> str: ...
    async def record_tool_call(self, *, session_id, name, arguments, parent_event_id=None) -> str: ...
    async def record_tool_result(self, *, session_id, tool_call_id, result, latency_ms=None, error=None) -> str: ...
    async def record_final_answer(self, *, session_id, answer, cited_ids) -> str: ...

    async def replay(self, *, session_id) -> list[MemoryEvent]:
        # whole-log fetch from the document store (replay, debug, distillation)
        ...

    async def tail(self, *, session_id, limit) -> list[MemoryEvent]:
        # last N events, for short-term rehydrate
        ...
```

A cross-session `events(...)` query over a structured projection is added with
the evolution engine, not now.

## Integration points

The instrumentation sites in `cogbase/core/query_runner.py` are the same ones
already used by short-term memory:

- `run()` start → `user_message`
- tool dispatch loop → `tool_called` (capture `time.monotonic()` for `latency_ms`)
- after each tool → `tool_result`, plus `retrieval_result` for the
  `structured_lookup` / `vector_search` branches
- final answer and passthrough → `final_answer`

Wire `EpisodicMemory` in `api/factory.py` from the shared (system) document store
so events are cross-app. Short-term memory reads its tail from the same log. Pass
`app_name` / `user_id` / scope through `api/models.py` → `app.query_stream` →
`runner.run()` for attribution.

## Build order

1. Add `append` / `load_lines` to `DocumentStoreBase` and the local_fs, s3, and
   memory backends.
2. Add the `MemoryEvent` model.
3. Implement `EpisodicMemory` (append + replay + tail) over the document store.
4. Instrument `QueryRunner`; wire the factory.
5. Build short-term context assembly and compaction over the log tail; route by
   consistent-hashing `session_id`.
6. Plumb scope (`user_id`, scope metadata) through the API.
7. Distill long-term facts and preferences from per-session logs (offline).
8. When the adaptive evolution engine is designed, add the structured projection
   it needs and feed episodic signals into it.
