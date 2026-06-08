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

## A dedicated append-only log store, not an extension of the document store

Episodic memory needs ordered, durable appends and tail reads. Those do **not**
belong on `DocumentStoreBase`: that store is overwrite-oriented (`save` replaces),
and bolting an `append` next to it invites a caller to `save` over a log object
and silently truncate it. A log object must only ever grow. So the append-only
behavior lives in its own small contract, `LogStoreBase`
(`cogbase/stores/log/`), keeping the truncation-safety a type-level guarantee
rather than a convention:

```python
class LogStoreBase(abc.ABC):
    async def append(self, collection: str, log_id: str, lines: Sequence[str]) -> None:
        """Append lines to log_id, creating it if absent. Ordered and durable; the
        whole batch lands as one append. Never overwrites."""

    async def load_lines(
        self, collection: str, log_id: str, *, tail: int | None = None
    ) -> list[str]:
        """Read the log back, optionally just the last `tail` lines (for rehydrate)."""

    async def delete(self, collection: str, log_id: str) -> None:
        """Whole-object delete — the only supported mutation (retention, erasure)."""
```

The contract is **line-oriented**: callers append and read NDJSON *records*, so
the trailing `\n` framing is the store's job, not the caller's — another way the
log API resists misuse. `collection` namespaces log families (e.g. `episodic`);
`log_id` is one append-only stream (a `session_id`).

The log is one object per session, period — never one object per event. A
session's conversation is short, so its log object stays small; that keeps the
read path trivial across backends and is the simplifying assumption the rest of
this design leans on. If a pathological session ever grows large enough to make
whole-object reads expensive, the layout doesn't change — we just stop reading the
whole object. Two deferred S3 optimizations, neither worth building now:

- **Stash the last compaction marker's byte offset in the object metadata.** S3
  objects carry user metadata, so the writer can record the `session_compacted`
  offset there at compaction time; rehydrate then range-reads from that offset
  instead of fetching the whole object — directly bounding the rehydrate read.
- **Size-guided tail.** `HeadObject` gives the object size, so a `tail=N` can
  range-read a suffix (e.g. from the midpoint), check whether it already contains
  *N* lines, and widen only if it fell short.

Two backend implementations:

- **local_fs** (`LocalFSLogStore`) — `open(path, "a")` with `O_APPEND` (atomic
  for line-sized writes), via `run_in_executor`; a turn's batch is one `write()`
  call. `load_lines()` reads the whole file and `tail` slices the last *N* lines.
- **s3** (`S3LogStore`) — **directory buckets** (S3 Express One Zone), one object
  per session, using **native append**: each `PutObject` passes
  `WriteOffsetBytes` equal to the current object size, and S3 rejects a stale
  offset. That offset is a **fencing token** — a deposed or stalled old writer
  cannot append after a handoff (see [single-writer and append
  safety](#session-affinity-and-routing)) — which standard-bucket
  read-modify-write CAS could not provide. A read fetches the whole object and
  `tail` filters the last *N* lines in memory; no range reads, listing, or
  sorting at this size.

NDJSON is the format: each event serializes to `json.dumps(event) + "\n"`. It is
line-oriented (cheap `tail` reads), needs no partial-read parsing, and
interleaves cleanly under a single active writer — one session is effectively
one concurrent writer, so contention is a non-issue.

**Appends are batched per turn, not per event.** A turn's events accumulate in the
in-memory cache (each stamped with its `seq` as it occurs) and are flushed as one
multi-line append at the turn boundary — so `append` receives a block of NDJSON
lines, and a turn costs *one* store write rather than one per `tool_called` /
`tool_result`. This bounds request count and keeps a session well under the S3
Express One Zone per-object append-count ceiling; it is also the natural point to
enforce continuity durability (see
[EpisodicMemory writer](#episodicmemory-writer)).

## Event identity

Every event carries three identity fields, with distinct roles:

- **`session_id`** — *location*. It names the log object the event lives in, so a
  reference is self-locating: resolving one is a log seek, not an index lookup.
- **`seq`** — a per-session, monotonically increasing integer. It is the
  *authority* for ordering within a session and enables **gap detection** (a
  missing `seq` is a dropped event) and cheap resume (rehydrate reads the tail,
  learns the last `seq`, continues without duplicating). The single writer for a
  session assigns it from its in-memory cache.
- **`ulid`** — a globally-unique, time-sortable id generated once at `record()`
  time. It is the **idempotency/dedupe key** (stable across retries, so a retried
  append is recognized even when `seq` assignment is the contended step) and an
  independent **witness for `seq`**: if a bug or a single-writer slip ever reuses
  a `seq`, the two events collide on `(session_id, seq)` but keep distinct ULIDs,
  so the collision is *detectable* instead of silently overwriting one. Its
  embedded timestamp is a soft cross-check on ordering (wall-clock, skew-prone —
  `seq` stays authoritative).

References between events (`source_event_ids`, `feedback.target`,
`session_compacted.replaces_through`) store the **triplet**: locate by
`(session_id, seq)`, then verify the `ulid` matches the line found. A mismatch
catches a `seq` reuse at read time, for free.

Because `session_id` makes every reference self-locating, long-term promotion
resolves provenance by log seek — no per-event point-lookup index is needed.

## Event payloads

The shared envelope on every event is `session_id`, `seq`, `ulid`, `event_type`,
`created_at`, `app_name`, `user_id`, and optional `parent_event_id`. The per-type
`payload` contracts are minimal, with an open `metadata` dict for extension:

| Event | Payload | Tier |
|---|---|---|
| `session_started` | client / app-config-version `metadata` | continuity |
| `user_message` | `text`, optional `attachments` | continuity |
| `final_answer` | `text`, `cited_ids` (triplets) | continuity |
| `session_compacted` | `summary`, `replaces_through` (last `seq` covered), `token_stats` | continuity |
| `tool_called` | `tool_call_id`, `name`, `arguments` | best-effort |
| `tool_result` | `tool_call_id`, `ok`, `result` \| `error`, `latency_ms` | best-effort |
| `retrieval_result` | `collection`, `query`, `hits` (`[{id, score}]`), `top_k` | best-effort |
| `feedback` | `target` (event triplet), `rating`, optional `comment` | best-effort |

(The *tier* column maps to the failure-handling tiers in
[EpisodicMemory writer](#episodicmemory-writer).)

Three contract notes:

- **`final_answer` is the canonical assistant turn** — the text that ends the
  agent loop, and the only assistant output short-term rehydrate threads into the
  conversation. There is no `assistant_message` event in v1: intermediate model
  text (pre/between tool calls) is replay detail, not continuity, and would be
  added later as a separate *best-effort* event if a consumer needs it — never as a
  second continuity-critical turn output.
- **`retrieval_result` is a typed projection of `tool_result`** for the
  `vector_search` / `structured_lookup` branches, carrying `score` for low-score
  mining, linked to its `tool_called` via `parent_event_id`. Emit it *in addition
  to* `tool_result` only once the gap detector wants score-filtered reads; until
  then it can fold into `tool_result` to avoid double-logging every retrieval.
- **`feedback` is written to the *current* session's log**, carrying the target
  event's triplet — not appended into the targeted session's log. Feedback often
  arrives in a later session, and writing it into the target's log from another
  session's process would create a second writer for that log, breaking the
  single-writer invariant (see
  [Session affinity and routing](#session-affinity-and-routing)). The triplet keeps
  the reference resolvable by log seek; the gap detector joins on it later.

## Short-term memory rides on the same log

Short-term memory needs no durable store of its own; the episodic log is its
persistence. Short-term is an in-memory working cache during a live session, and
its durable record is the message/compaction events already in the log.

- **Cross-turn continuity needs only the conversational thread** —
  `user_message` and `final_answer` (the canonical assistant turn) — plus
  compaction summaries. It does *not* need the tool-call chain. Tool outputs are
  intra-turn scratch held in RAM for the active agent loop and are never
  rehydrated: a dead turn is re-run, not resumed.
- **Compaction appends a `session_compacted` summary event** rather than
  rewriting history (the log is append-only). The event records `replaces_through`
  (the last `seq` it covers), so rehydrate loads that summary plus every event
  after `replaces_through` — bounded, not a whole-log scan. Compaction triggers on
  *model-context* pressure — when the thread approaches a fixed fraction of the
  LLM window — not on a small per-turn budget, which keeps it rare and each summary
  worth persisting. The **durability invariant**: never drop a turn from the
  in-memory transcript unless either the turn itself, or a `session_compacted`
  covering it, is durable in the log. Because raw `user_message` / `final_answer`
  events are appended each turn, an in-memory summary lost to a crash before its
  `session_compacted` lands is self-healing — rehydrate re-derives it from the raw
  turns still in the log; it is never a *loss*, only a re-compaction.
- **`build_context` is then nearly a pass-through.** Because compaction already
  holds the rehydrated thread under the window, context assembly is "return the
  last `session_compacted` summary plus every event after `replaces_through`" — no
  per-turn newest-first budget walk. The only real work short-term does is
  rehydrate-or-cache-hit, append, and compact-when-near-window.
- On a cold start or cache miss, short-term rebuilds by reading the (small) whole
  log, taking the last `session_compacted` summary plus every event after its
  `replaces_through`, filtered to the continuity events it needs. Because the read
  is whole-object, the compaction marker is always in the result — there is no
  risk of a count-bounded `tail` window falling short of it. (`tail=N` stays a
  convenience for the no-compaction-yet case and for debugging.)

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

- With **local_fs** as the log store, processes on a node need **no** node-local
  shared cache: the OS file-system page cache is already shared across processes
  on the host, so log tail reads after a process restart hit warm cache.
- With **s3** as the log store, there is no shared host cache, so a node-local
  shared store may later be worth adding to avoid re-fetching logs across
  processes. Defer it until measured.

Single-writer and append safety:

- Single-writer-per-session is the **invariant** the design targets: consistent-
  hash affinity normally routes a session to one process, which assigns `seq` from
  its cache. But append *correctness* does not rely on affinity being perfect — a
  failover window can briefly produce two owners, and a paused old owner can wake
  after handoff (a lease alone does not prevent this; only a **fencing token** the
  store rejects when stale does).
- So the append leans on the backend's own serialization as the safety net, not on
  affinity: **local_fs** `O_APPEND` is atomic for line-sized writes; **s3
  directory buckets** use the **append offset** (`WriteOffsetBytes` must equal the
  current object size), itself a conditional on object length, and create-if-absent
  uses `If-None-Match: *`. (We target directory buckets specifically for native
  append; standard buckets, lacking it, would force a read-modify-write of the
  whole object under `If-Match` CAS and could not fence a stuck writer.)
- These guard *byte-level* integrity and *overwrite*, but the backends differ on
  the **stuck-writer** case. `O_APPEND` has **no fencing**: if a deposed or stalled
  old writer wakes and appends after the new owner has advanced, both appends
  succeed, so the log can hold an out-of-order, `seq`-colliding straggler (two
  distinct events both stamped `seq=6`). local_fs cannot *prevent* this — only
  *detect* it, which is exactly the `ulid` witness's job (see
  [Event identity](#event-identity)): the two records carry different ULIDs, so the
  straggler is identifiable instead of silently masking one. The s3 directory
  bucket, by contrast, can **reject** the stale writer — its `WriteOffsetBytes` no
  longer matches the object length, so the conditional append fails; the offset is
  the fencing token. So local_fs *detects*, s3 *prevents*.
- A retry (the same logical event again) is the simpler duplicate, caught by the
  `ulid` dedupe key regardless of backend — `O_APPEND` will happily write the same
  line twice, so identity, not the append primitive, makes recording idempotent. On
  read/rehydrate, prefer the contiguous monotonic `seq` run from the active writer
  and quarantine an out-of-order straggler (flag it for debugging, don't thread it
  into the conversation).

## Retention, deletion, and redaction

The log is the one place raw user text, tool arguments, retrieved chunks, and any
secrets that leak through them are stored verbatim. Scoping (see
[memory.md](memory.md#scoping)) controls *who reads* a memory; it says nothing
about *how long* the raw record lives or *how* it is erased. Two design choices
above make lifecycle non-trivial and force explicit answers.

**TTL and per-session deletion.** The one-object-per-session layout makes coarse
deletion cheap: expiring or erasing a session is a single object delete, and a
per-user erasure (GDPR-style) is a `list` by `user_id` prefix then delete. A TTL
sweep deletes session objects past a retention horizon. Long-term records carry
their own `expires_at` (see [memory.md](memory.md#long-term-memory)) and expire
independently — the log and the promoted knowledge have separate clocks.

**Deletion breaks log-seek provenance — promotion must snapshot.** Provenance
resolves by seeking into the per-session log via `(session_id, seq)` (see
[Event identity](#event-identity)), and a session object is cheap to delete — but
those two facts collide: **delete the log and every long-term memory promoted from
it has a dangling `source_event_ids` pointer.** So promotion must *copy the cited
evidence into the long-term record* (the `final_answer.cited_ids` text and the
deciding event payloads), not merely reference it. The triplet stays for audit and
drill-down *when the log still exists*; the long-term record stays valid and
self-contained when it does not. A retention sweep is then free to delete logs
without consulting long-term memory, and a provenance read treats a missing log as
"evidence archived," not corruption.

**Redaction fights append-only — pick the erasure primitive.** The log is
immutable NDJSON; even compaction appends a summary rather than rewriting history.
Redacting a single secret in place therefore contradicts the core invariant. The
design picks **whole-object delete as the only mutation** and layers two
non-mutating tools on top:

- **Redact-on-write.** Best-effort scrub of known-sensitive fields (auth headers,
  tokens in tool arguments) *before* the line is appended, so the verbatim secret
  never lands in the log. This is the primary defense; once written, the line is
  immutable.
- **Redact-on-read.** A filter applied when `replay` / `tail` surface payloads,
  for classes discovered after the fact. It hides, it does not erase.

True per-field erasure of an already-written record is explicitly **out of scope
for v1**: the supported erasure granularity is the session object. If
record-level cryptographic erasure (per-session key, shred the key to erase) is
ever required, it is added as a storage-layer option without changing the event
model — but it is not built now.

## Deferred: the structured projection

When the adaptive evolution engine is designed, it will need cross-session,
filtered, aggregate reads that the per-session logs serve poorly. At that point
add a **derived** structured projection — rebuilt from the logs, payload-free —
shaped to what the gap detector actually mines. The likely shape, for reference:

| Field | Purpose |
|---|---|
| `(session_id, seq)` (PK) | locates the event in the log; promotion references resolve by log seek |
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

One shape choice to settle then, not now: **per-event rows** (maximum query
flexibility) vs. **per-session signal rollups** (one row per session: did it
null-answer? minimum retrieval score? tool-chain fingerprint? — far smaller,
drills into the log by `session_id` on demand). Point lookup is *not* a factor:
references are already self-locating via `(session_id, seq)`, so promotion
resolves provenance by log seek with or without this projection.

Because the projection is derived, either shape can be built or rebuilt from the
logs once its consumer is concrete.

## EpisodicMemory writer

A thin durable writer (`cogbase/memory/episodic.py`), async like the rest of the
memory layer. Unlike a cache it never compacts or evicts — it only appends. For
now each `record_*` stamps and buffers an event in the session's in-memory cache;
the buffered events are flushed to the log as one multi-line append at the turn
boundary (see [Document store additions](#document-store-additions)). The
projection write is added with the evolution engine.

Failure handling is **tiered**, because the log is no longer write-only telemetry
— it is also the durable backing for short-term rehydrate:

- Best-effort appends must never **block or fail** the in-flight query. The
  current turn is served from short-term's in-memory cache, not from the log, so
  generating the answer never awaits their durability.
- But continuity-critical events — `user_message`, `final_answer`, and
  `session_compacted` — may no longer be **silently dropped**. They are what a
  later rehydrate reconstructs the thread from; dropping one silently corrupts a
  future session's context (a gap, or post-compaction reasoning over a summary that
  never persisted). These need at-least-once durability: the turn-boundary append
  must land **before the turn is acknowledged complete**, so a follow-up that fails
  over to another process never rehydrates a log missing a turn the user already
  saw. This is what makes "any process can rehydrate, so affinity is only an
  optimization" actually true — it holds for *flushed* events, and continuity
  events are flushed before the turn closes. The append is non-blocking with retry,
  buffered in the in-memory cache until it succeeds, and a persistent failure is
  surfaced/alerted rather than swallowed.
- Observability-only events — `tool_called`, `tool_result`, `retrieval_result`,
  latency — stay pure best-effort; losing one costs only analytics, never
  continuity.

The in-memory cache doubles as the retry buffer, so a transient append failure is
covered until a retry lands. The only true loss window is a process crash before a
turn's continuity events are first persisted; the turn-boundary flush keeps that
window to a single turn, and a lost in-memory `session_compacted` is self-healing
because rehydrate re-derives it from the raw turns still in the log.

```python
class EpisodicMemory:
    def __init__(self, log_store: LogStoreBase) -> None: ...

    async def record(self, event: MemoryEvent) -> str:
        # stamp seq + ulid, append the NDJSON line to the session's log (source of
        # truth) via the log store; dedupe by ulid so a retried append is idempotent
        ...

    # the wrappers return the event's (session_id, seq) reference so callers can
    # thread parent_event_id / references into causal chains:
    async def record_user_message(self, *, session_id, content, **scope) -> str: ...
    async def record_tool_call(self, *, session_id, name, arguments, parent_event_id=None) -> str: ...
    async def record_tool_result(self, *, session_id, tool_call_id, result, latency_ms=None, error=None) -> str: ...
    async def record_final_answer(self, *, session_id, answer, cited_ids) -> str: ...

    async def replay(self, *, session_id) -> list[MemoryEvent]:
        # whole-log fetch from the log store (replay, debug, distillation)
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

Wire `EpisodicMemory` in `api/factory.py` from a shared (system) log store so
events are cross-app. Short-term memory reads its tail from the same log. Pass
`app_name` / `user_id` / scope through `api/models.py` → `app.query_stream` →
`runner.run()` for attribution.

## Build order

1. Add the `LogStoreBase` contract (`append` / `load_lines` / `delete`) and its
   local_fs and s3 (directory-bucket) implementations. ✅
2. Add the `MemoryEvent` model with identity (`session_id` + `seq` + `ulid`) and
   the per-type payload contracts. ✅
3. Implement `EpisodicMemory` (append + replay + tail) over the log store. ✅
4. Instrument `QueryRunner`; wire the factory.
5. Build short-term context assembly and compaction over the log tail; route by
   consistent-hashing `session_id`.
6. Plumb scope (`user_id`, scope metadata) through the API.
7. Distill long-term facts and preferences from per-session logs (offline).
8. When the adaptive evolution engine is designed, add the structured projection
   it needs and feed episodic signals into it.
