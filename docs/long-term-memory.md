# Long-Term Memory — Distillation Design

Long-term memory is the curated, durable knowledge CogBase accumulates across
sessions: user preferences, project and organization facts, confirmed
corrections, and learned retrieval hints (see the tier overview in
[memory.md](memory.md#long-term-memory)). This document records how it is built —
**distillation**, the offline pipeline that reads per-session episodic logs and
promotes durable records — and where it stops, namely the seam against the
adaptive evolution engine.

Episodic memory already names two downstream consumers and cuts the seam between
them (see [episodic-memory.md](episodic-memory.md#the-write-path-is-a-red-herring-the-read-path-splits-in-two)):
per-session distillation reads each log whole, and cross-session gap mining reads
an aggregate projection. This document is the design of the *first* consumer. The
second is a separate engine; the boundary is stated below and is load-bearing.

## Two learning loops, not one

Distillation and adaptive evolution both read the same episodic log, but they
*learn different objects* and *write to different targets*. They are separate
subsystems, and conflating them would force one store to serve two unrelated read
patterns.

| | Distillation → long-term memory | Adaptive evolution |
|---|---|---|
| Learns about | the world / user (facts, preferences) | the app's own coverage gaps (meta) |
| Reads | one session's whole log | cross-session, filtered, aggregate |
| Produces | content records | config patches (new field / step / skill) |
| Consumed by | the query runner, online, via `recall()` | the config + re-ingest pipeline, offline |
| Cadence | per session, on settle | periodic batch over a corpus |
| Acceptance | confidence-aware auto / review | explicit human accept / reject |
| Storage | structured + vector (for recall) | the deferred structured projection |
| Scope | user / project / org | app |

The decisive test for which subsystem owns a behavior is **where its output is
consumed**: long-term memory writes *content* recalled into an LLM call at query
time; adaptive evolution writes *configuration* applied to the app structure
offline. This document owns the left column only.

What the two share is thinner than it looks, and all of it is upstream or
primitive — not a shared store or service:

- **The episodic log** as common source (already the design).
- **The deferred structured projection** — but *only* adaptive evolution needs
  it. Distillation reads per-session whole logs via `replay` and never touches the
  cross-session projection (see
  [episodic-memory.md](episodic-memory.md#deferred-the-structured-projection)).
- **Provenance snapshot, confidence, and review status** — shared *types and
  helpers*, not a shared engine. Both carry provenance and a human-acceptance
  gate; factor those into small reusable primitives, keep the engines apart.

Two loose feedback loops are worth naming but not wiring tightly: a distillation
that repeatedly fails to ground a fact in documents is itself a gap signal for
evolution, and a preference that recurs across many users is a candidate for an
app-level default. Let these flow through the episodic log / projection, never
through a direct dependency between the two engines.

So `cogbase/memory/` owns long-term memory; adaptive evolution is its own package
that consumes the same log. They sit side by side, not nested.

## Goals

- Promote durable, provenance-backed knowledge out of raw session logs.
- Reconcile new observations against accumulated belief — reinforce, update, or
  retract — so the store stays curated, not append-only.
- Recall relevant memories into the query path, scoped to the caller.
- Keep long-term records self-contained so a session log can expire without
  dangling their provenance.
- Distinguish memory-derived claims from document-backed evidence at answer time.

## Distillation is extract-structured over conversation logs

CogBase already ingests documents and extracts typed structured records (the
`extract-structured` step; see [pipeline.md](pipeline.md)). **Distillation is that
same pipeline pointed at session logs instead of documents.** A session log is the
"document"; memory records are the extracted "structured records." This reuses the
extraction machinery and mental model.

The *new* part is **reconciliation**: the document pipeline upserts by primary key,
but memory must merge a new observation against accumulated belief — including
contradiction. That step does not exist in the document pipeline and is the crux of
this design.

## Pipeline

Distillation runs offline and async, like ingestion — never on the request path.

1. **Trigger — on session settle.** A session is "settled" on explicit close or an
   idle TTL. Enqueue a background distillation task (mirrors the ingestion task model).
2. **Read — whole log.** `EpisodicMemory.replay(session_id)` returns the session's
   events in order. The session is short (the simplifying assumption episodic
   memory leans on), so the whole-log read is cheap and bounded.
3. **Extract candidates.** An LLM extraction prompt over the conversational thread
   produces candidate records: user preferences, stable user / project /
   organization facts, and confirmed corrections. Each candidate carries the
   `source_event_ids` (triplets) it was derived from.
4. **Reconcile — the crux.** For each candidate, vector-recall related existing
   records *in the same scope*, then an LLM emits one operation:
   - **ADD** — no related record; insert.
   - **UPDATE** — matches an existing record; reinforce (bump `confidence`,
     refresh `updated_at`, append `source_event_ids`) or revise its content.
   - **DELETE** — contradicts an existing record and supersedes it. Resolution is
     by recency and confidence; a *confirmed correction* outranks an inferred fact.
   - **NOOP** — already known, nothing to change.

   This is why long-term memory is **not** append-only — it is curated. It is also
   why it needs a vector index: you cannot reconcile what you cannot find.
5. **Write with provenance snapshot.** Promotion *copies the cited evidence into
   the record* (the `final_answer.cited_ids` text and the deciding event payloads),
   not merely a reference — so a later log deletion leaves the record valid and
   self-contained (see
   [episodic-memory.md](episodic-memory.md#retention-deletion-and-redaction)). The
   triplet stays for audit and drill-down while the log exists.

## Promotion: confidence and review

Promotion is confidence-aware (see [memory.md](memory.md#long-term-memory)).

- **Auto-promote** low-risk classes — e.g. interaction preferences ("prefers
  concise answers," "always wants citations") — at `status: active`.
- **Gate** behavior-affecting memories and user / project / organization facts
  behind `status: pending_review`, carrying `source_event_ids` so a reviewer can
  audit the evidence before the memory influences answers.

Records expire on their own `expires_at` clock, independent of the session logs
they were distilled from — the log and the promoted knowledge have separate
retention horizons.

## Record shape

Each long-term record carries provenance and lifecycle fields (the canonical list
is in [memory.md](memory.md#long-term-memory)):

| Field | Purpose |
|---|---|
| `memory_id` | stable identity |
| `app_name` | partition boundary — the outer RBAC fence; every record and every recall is filtered by it (see [The app is the partition boundary](#the-app-is-the-partition-boundary)) |
| `scope` | `user` / `project` / `organization` / `global` — the finer recall filter *within* an app |
| `kind` | `preference` / `fact` / `correction` / `retrieval_hint` |
| `content` | the durable claim, in natural language |
| `confidence` | reinforced on repeat observation, weighed in reconciliation |
| `status` | `active` / `pending_review` / `superseded` |
| `source_event_ids` | triplets into the episodic log — audit / drill-down |
| `evidence_snapshot` | copied cited text — keeps the record valid after log deletion |
| `created_at` / `updated_at` | lifecycle |
| `expires_at` (optional) | independent retention clock |

## Recall at query time

`recall(query, scope, limit)` runs a vector search over memory text, **filtered to
the calling app's partition and the caller's authorized scopes within it** — the
non-negotiable multi-tenant rule: the manager never crosses the app boundary (see
[The app is the partition boundary](#the-app-is-the-partition-boundary)) and never
returns broader-scope memories than the caller may see (see
[memory.md](memory.md#scoping)). Recalled records are injected into the query
runner's context **marked as memory-derived**, so the evidence policy can
distinguish them from document-backed claims.

This honors the evidence policy (see [memory.md](memory.md#evidence-policy)):
ingested documents, extracted records, and workflow outputs remain the primary
evidence layer. A memory's factual claim is surfaced as memory unless it has
provenance to source documents or accepted structured records.

## The app is the partition boundary

Recall filters by *scope* within an application; the application itself is the
*outer* fence. A customer runs several apps — HR, finance, engineering — that need
not know about one another, and pooling their memories would turn access control
(RBAC) into a query-time predicate a single bug could bypass. So `app_name` is a
**mandatory partition key** on every long-term record and every `recall`, not just
one of the `scope` values: `scope` (`user` / `project` / `organization` /
`global`) is the finer filter *inside* an app; `app_name` is the boundary *around*
it.

Physically the records still live in the shared **system** stores — like the
episodic log, long-term memory is system-level infrastructure, not data bundled
into one app. The separation is *logical*: each app addresses its own prefixed
collections over the system structured and vector stores — the same scoping
mechanism application data already uses — so isolation is a property of the store
layout, not of remembering to AND-in `app_name` on every query.

This is deliberately the *narrowest* useful boundary. Sharing a memory across apps
— a customer-wide preference that should hold everywhere — is a real future need,
but opt-in rather than default: it slots in as a **namespace / project scope above
the app**, a higher prefix a record is explicitly promoted to, with no change to
the record shape or the recall path. Until that exists, total per-app isolation is
the safe default.

This is the one place long-term memory's scoping diverges from episodic memory's.
The episodic log is deliberately *unscoped* and cross-app so the evolution engine
can mine across applications (see
[episodic-memory.md](episodic-memory.md#integration-points)); long-term memory is
recalled into a live answer for a specific app's user, so it carries the app fence
the log does not.

## The retrieval-hint boundary

[memory.md](memory.md#long-term-memory) lists "successful retrieval plans" and
"common intent-to-tool-chain patterns" under long-term memory — and that is the one
place this subsystem blurs into adaptive evolution. Resolve it with the same
consumption test:

- A **routing hint recalled into the query runner's context at query time** is
  long-term memory (`kind: retrieval_hint`).
- A **structural config change** — add a field, a step, a skill — applied offline
  is adaptive evolution.

The same raw observation ("queries of class X always chain tools A→B") yields two
different outputs depending on whether it is *recalled* online or *baked* into
config. Stating the test here keeps the two engines from fighting over a record.

## Storage strategy

Reuse the existing stores (see [memory.md](memory.md#storage-strategy)), over the
**system** stores partitioned per app (see
[The app is the partition boundary](#the-app-is-the-partition-boundary)):

- **Structured store** for canonical records, with indexes on `scope`, `kind`,
  `status`, `confidence`, and timestamps.
- **Vector store** for semantic recall over `content` — and for the reconcile
  step's "find related existing records" lookup.
- Each app addresses its own **prefixed collections** over the shared system
  stores — the same scoping mechanism application data already uses — so the app
  fence is enforced by the store layout, not by a query-time predicate.

Unlike the episodic log (one append-only object per session), long-term memory is
mutable: reconciliation updates and supersedes records in place.

## Public interface

Long-term memory is reached through `MemoryManager` (see
[memory.md](memory.md#public-interface)); the `recall` and `promote` methods there
are its surface. A representative service shape:

```python
class LongTermMemory:
    def __init__(self, structured_store, vector_store, llm, embeddings) -> None: ...

    async def recall(self, *, query: str, scope: dict, limit: int = 10) -> list[dict]:
        # scoped vector search over memory content; used online by the query runner
        ...

    async def reconcile(self, *, candidate: dict, scope: dict) -> str:
        # vector-recall related records, LLM decides ADD / UPDATE / DELETE / NOOP,
        # apply with provenance snapshot; returns the affected memory_id
        ...

    async def promote(
        self, *, source_event_ids: list[str], kind: str, content: str,
        scope: dict, confidence: float, evidence_snapshot: dict,
    ) -> str:
        # the ADD path of reconcile, exposed for direct promotion
        ...


class Distiller:
    def __init__(self, episodic: "EpisodicMemory", long_term: LongTermMemory, llm) -> None: ...

    async def distill_session(self, *, session_id: str, scope: dict) -> list[str]:
        # replay the log, extract candidates, reconcile each; returns memory_ids
        ...
```

Placement: `cogbase/memory/long_term.py` (the `LongTermMemory` service) and
`cogbase/memory/distill.py` (the offline `Distiller`). Adaptive evolution lives in
its own package and is out of scope here.

## Build order

1. Add the long-term record model and the `LongTermMemory` service over the
   structured and vector stores (`recall`, `promote`).
2. Add `reconcile`: vector-recall related records, LLM ADD / UPDATE / DELETE /
   NOOP, apply with provenance snapshot and confidence reinforcement.
3. Add the `Distiller`: replay a session log, extract candidates, reconcile each.
4. Trigger distillation on session settle (background task); plumb scope through.
5. Wire `recall` into the query runner's context assembly, marked as
   memory-derived, behind the evidence policy.
6. Add promotion review for gated kinds (`pending_review` → `active`).

This sequence delivers useful recall early while keeping reconciliation — the part
with no analog in the document pipeline — isolated and testable before it is wired
into the live query path.
