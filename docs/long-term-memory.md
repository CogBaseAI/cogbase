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
It is implemented in `cogbase/memory/distill.py` (`Distiller`) over the
`LongTermMemory` service in `cogbase/memory/long_term.py`.

1. **Trigger — on session settle.** A session is "settled" on explicit close (the
   `POST .../sessions/{id}/close` endpoint — `close_session` in
   `api/routers/applications.py`) or an idle TTL. The endpoint evicts the
   short-term cache and enqueues a background `distill_session` task (mirroring the
   ingestion task model), so close returns immediately.
2. **Read — only the un-distilled tail.** `EpisodicMemory.replay(session_id)`
   returns the session's events in order; the session is short, so the read is
   cheap and bounded. The distiller then projects only the turns *past* the last
   `session_distilled` watermark (`latest_distillation`), so a re-distill of a
   resumed / re-closed session re-examines only the new turns — see
   [Idempotency: the distillation watermark](#idempotency-the-distillation-watermark).
3. **Extract candidates (with front-loaded belief).** An LLM extraction prompt over
   the conversational thread produces candidate records: preferences, stable
   user / project / organization (or USER-asserted subject-matter) facts, confirmed
   corrections, and `retrieval_hint` routing patterns. Each candidate carries the
   `source_event_ids` it was derived from, the `entities` it is about, a
   `confidence` score, and the `linked_memory_ids` of related existing memories.
   Two things shape this single call:
   - **Existing memories are front-loaded.** Before extracting, the distiller
     vector-recalls the active memories most related to the transcript
     (`existing_memory_limit`, default 10) and injects them into the prompt as an
     `## Existing memories` block, each tagged with a *masked* integer id (`[id=0]`,
     `[id=1]`…) rather than its real UUID — an LLM asked to echo a 36-char UUID
     eventually invents one. This lets the same call both reconcile against and link
     to accumulated belief (see [Single-call reconciliation](#single-call-reconciliation)
     and [The memory graph](#the-memory-graph)).
   - **Relative time is anchored.** The prompt carries an *observation date* — when
     the distilled turns actually happened, not distill time — so "yesterday" /
     "last week" resolve correctly even for a session distilled days later. Each
     candidate's `observed_at` is then dated by its own latest source turn.
   - **A confidence floor drops weak candidates.** Each candidate is scored in
     [0, 1]; one below its kind's floor (`DEFAULT_MIN_CONFIDENCE` — correction/
     preference 0.7, fact/hint 0.6) or without a usable score is abandoned before
     reconcile. For the auto-promoting kinds the floor *is* the de facto auto-active
     bar (see [Promotion: confidence and review](#promotion-confidence-and-review)).
4. **Reconcile — the crux.** Merge each candidate against accumulated belief. The
   LLM emits one operation:
   - **ADD** — no related record; insert.
   - **UPDATE** — matches an existing record; reinforce (bump `confidence` by
     `CONFIDENCE_REINFORCE` = 0.1, refresh `updated_at`, merge `entities`,
     `linked_memory_ids`, and `source_event_ids`, advance `observed_at`) or revise
     its content.
   - **DELETE** — contradicts an existing record and supersedes it — *but only if
     the candidate outranks the target* (`_outranks`): a confirmed correction
     outranks an inferred fact, ties break on confidence, and the candidate is the
     more recent observation. A weaker contradiction is dropped and existing belief
     stands.
   - **NOOP** — already known, nothing to change.

   The related records the op is decided against are **vector hits ∪ entity
   overlap** (`_related_records`): vector similarity alone misses a paraphrased
   claim about the same entity, so active records sharing a normalized entity are
   unioned in. This is why long-term memory is **not** append-only — it is curated —
   and why it needs a vector index: you cannot reconcile what you cannot find. The
   op can be decided two ways (one shared apply tail); see
   [Single-call reconciliation](#single-call-reconciliation).
5. **Write with provenance snapshot.** Promotion *copies the cited evidence into
   the record* (the `final_answer.cited_ids` text and the deciding event payloads,
   each capped at `_MAX_CITED_PAYLOAD_BYTES`), not merely a reference — so a later
   log deletion leaves the record valid and self-contained (see
   [episodic-memory.md](episodic-memory.md#retention-deletion-and-redaction)). The
   `source_event_ids` triplets stay for audit and drill-down while the log exists.
6. **Advance the watermark.** Append a `session_distilled` event recording the last
   turn examined — even when the pass extracted nothing — so the next pass starts
   past it (see [Idempotency: the distillation watermark](#idempotency-the-distillation-watermark)).

## Single-call reconciliation

The pipeline above describes reconciliation as a per-candidate step, but there are
**two implementations** of it, sharing one apply tail (`_apply_op`), the same
anti-hallucination id masking, and the same promotion gate. They differ only in
*who decides the op*:

- **Two-phase** (`LongTermMemory.reconcile`, `single_call=False`) — the auditable
  path. The extractor produces N candidates blind; then *per candidate*
  `reconcile` re-recalls related records (`_related_records`) and spends one more
  LLM call (`_decide`) to choose ADD / UPDATE / DELETE / NOOP. **N+1 LLM calls per
  session.** Its upside is auditability: each `_decide` yields one op + reasoning
  for one (candidate, target) pair, and each candidate gets a fresh recall.
- **Single-call** (`LongTermMemory.reconcile_decided`, the distiller default) —
  collapses the above to **one LLM call per session.** Because the existing
  memories are already front-loaded into the extraction prompt (step 3), the
  extractor emits each fact *together with* its `operation`, `target_memory_id`
  (a masked id resolved back to a real UUID), and optional `revised_content`.
  Reconciliation then just applies the pre-made decision — no per-candidate recall,
  no second LLM call. It trades the per-candidate audit point and recall for O(1)
  cost. Robustness mirrors the two-phase path: an UPDATE / DELETE / NOOP whose
  target is missing, unresolvable, or no longer active **degrades to ADD** rather
  than touching the wrong record or dropping the candidate.

Toggle with the `Distiller(single_call=...)` flag.

### Domain-scoped prompts

Both the extraction prompt and the reconcile prompt accept an *additive* domain
slot, so one application can narrow judgement without being able to relax the core
rules or change the output schema:

- `Distiller(domain_fact_guidance=...)` narrows what subject matter counts as a
  durable `fact` / `correction` (e.g. "contract clauses and their effective
  dates"). It is placed *above* the provenance rule and framed as topic-scoping
  only, so it cannot relax the rule that a subject-matter fact is durable only when
  the USER is its source.
- `LongTermMemory(reconcile_guidance=...)` — the consolidation-side analog (and the
  analog of mem0's `custom_update_memory_prompt`) — injects domain judgement about
  *when* two observations are the same claim, a contradiction, or a mere refinement
  (e.g. "two clauses with different effective dates are distinct records, not a
  contradiction"), without changing the ADD / UPDATE / DELETE / NOOP set.

## The memory graph

Long-term records are not isolated rows: each carries `linked_memory_ids`, edges to
related memories, forming a graph that recall traverses. Edges are built two ways,
and recall reads them one hop out.

**LLM-emitted edges.** In the single-call extraction (step 3), the extractor sets
`linked_memory_ids` on each new memory to the ids of the front-loaded existing
memories it is specifically related to — same entity or topic, a follow-up event,
an update, or a contradiction. The prompt forbids linking on a vague shared theme.

**Deterministic auto-linking** (`Distiller._auto_link`). On top of the LLM's edges,
the distiller adds edges between a candidate and a recalled existing memory when
they **share a discriminative entity** — one present in no more than
`auto_link_max_entity_ratio` (default 0.1) of all active records, measured by
`LongTermMemory.active_entity_frequencies` (a cheap `entities`-only scan). The
ratio is the whole point: a *ubiquitous* entity (a recurring speaker tagging most
records) exceeds the threshold and is ignored, so the graph never collapses into a
same-subject clique; a *rare, specific* entity ("homeless shelter", "career fair")
is what earns an edge. Each candidate gains at most `_MAX_AUTO_LINKS` (3) such
edges, and the scan is best-effort — a failure leaves the LLM's edges untouched
rather than sinking the distillation. This is the deterministic complement to the
LLM's semantic linking: cheap, explainable, and recall-relevant.

**Recall traversal** (`LongTermMemory._neighbors`). When recall returns its
vector-relevance hits, it appends up to `recall_neighbors` (default 5) records one
hop out along the graph — context the vector query alone would miss (a follow-up
event, the other side of a contradiction). Edges are bidirectional in effect:
*forward* follows each hit's own `linked_memory_ids`; *reverse* finds active
records that link back *to* a hit (an `overlaps` scan pushed to the store), so a
freshly recalled older memory still surfaces the newer memories pointing at it.

## Idempotency: the distillation watermark

Sessions are resumable and re-closable, so distillation must be safe to re-run.
Each pass appends a `session_distilled` event recording `distilled_through` (the
last turn `seq` it examined); the next pass projects only turns past that watermark
(`latest_distillation` / `project_thread(since_seq=...)`). Without it, a re-distill
would re-extract the whole transcript and reinforce every already-captured record,
drifting confidence toward 1.0 with no new evidence.

The watermark is advanced even when a pass extracts *nothing* (those turns were
judged and produced no memory), but **only after a successful extraction** — an
unparseable / failed extraction returns early and leaves the turns un-watermarked
so they are retried. Writing it is best-effort and symmetric with compaction's
`replaces_through`: a failure risks a future re-examine, never a lost record, so it
is logged rather than raised.

## Promotion: confidence and review

Promotion is confidence-aware (see [memory.md](memory.md#long-term-memory)), gated
per kind by `AUTO_PROMOTE_CONFIDENCE`: a record auto-promotes to `status: active`
only when its `confidence` clears its kind's threshold, otherwise it lands at
`status: pending_review`.

| Kind | Auto-promote threshold | Effect |
|---|---|---|
| `preference` | 0.0 | always auto-active (low-risk interaction signal) |
| `retrieval_hint` | 0.0 | always auto-active |
| `fact` | 0.85 | auto-active only when strongly supported, else review |
| `correction` | 1.01 (unreachable) | **always** held for review — it overrides existing belief |

Because the auto-promote threshold for preferences / hints is 0.0, the distiller's
per-kind confidence *floor* (step 3) is what actually keeps weak ones out: a 0.5
preference is abandoned before reconcile rather than auto-promoted.

**Review gate.** Gated records wait at `pending_review`, invisible to `recall` and
`lookup` (both active-only). `LongTermMemory.list_pending` is the only reader of
them — FIFO, full record with `source_event_ids` / `evidence_snapshot` so a
reviewer can audit the evidence. `review_many` applies a batch of verdicts: accept
→ `active`, reject → `superseded` (a retraction marker consistent with reconcile's
DELETE, keeping the audit trail rather than hard-deleting). It is idempotent — a
record not currently `pending_review` is reported `skipped`, so a re-submitted
decision can't resurrect a superseded record.

Records expire on their own `expires_at` clock, independent of the session logs
they were distilled from — the log and the promoted knowledge have separate
retention horizons. (Retention sweeping of superseded / stale `pending_review` rows
is a known TODO; nothing GCs the structured store yet.)

## Record shape

Each long-term record carries provenance and lifecycle fields (the canonical list
is in [memory.md](memory.md#long-term-memory)):

| Field | Purpose |
|---|---|
| `memory_id` | stable identity (UUID) |
| `app_id` | partition boundary — the outer RBAC fence; every record and every recall is filtered by it (see [The app is the partition boundary](#the-app-is-the-partition-boundary)) |
| `kind` | `preference` / `fact` / `correction` / `retrieval_hint` |
| `content` | the durable claim, in natural language |
| `entities` | normalized named entities the claim is about — drives entity-overlap recall and `auto_link` |
| `confidence` | reinforced (+0.1, capped) on repeat observation, weighed in reconciliation |
| `status` | `active` / `pending_review` / `superseded` |
| `linked_memory_ids` | edges to related memories — [the memory graph](#the-memory-graph) recall traverses |
| `source_event_ids` | triplets into the episodic log — audit / drill-down |
| `evidence_snapshot` | copied cited text — keeps the record valid after log deletion |
| `observed_at` | when the claim was observed (latest source turn) — anchors recall dating, independent of `created_at` |
| `created_at` / `updated_at` | lifecycle |
| `expires_at` (optional) | independent retention clock |

> **Scope today is the app partition only.** The finer `scope`
> (`user` / `project` / `organization` / `global`) is in the design but not yet a
> record field: the stores are passed in already app-scoped, so `app_id` is the
> single partition until multi-user / RBAC is designed. The
> [partition-boundary](#the-app-is-the-partition-boundary) discussion below still
> holds — `scope` slots in as the finer filter when it lands.

## Recall at query time

There are two read paths into long-term memory, both **active-only** (the vector
index holds only `active` records, so `pending_review` / `superseded` never match)
and both **filtered to the calling app's partition** — the non-negotiable
multi-tenant rule: the manager never crosses the app boundary (see
[The app is the partition boundary](#the-app-is-the-partition-boundary)).

- **`recall(query, limit)` — the push path.** A vector search over memory `content`,
  run automatically before the first LLM call to inject relevant memory into the
  query runner's context. Beyond the `limit` relevance hits it appends up to
  `recall_neighbors` graph neighbors (see [recall traversal](#the-memory-graph)).
- **`lookup(query, kind, entities, limit)` — the pull path.** Backs the query
  runner's optional `memory_lookup` tool, for when the per-turn push isn't enough
  ("what do you know about project X?"). Combines an optional semantic `query` with
  `kind` / `entities` filters (entity match exact on the normalized form). With a
  query, results follow vector-relevance order; without one, it is a structured
  scan ordered most-recently-updated first.

Recalled records are injected **marked as memory-derived**, so the evidence policy
can distinguish them from document-backed claims.

This honors the evidence policy (see [memory.md](memory.md#evidence-policy)):
ingested documents, extracted records, and workflow outputs remain the primary
evidence layer. A memory's factual claim is surfaced as memory unless it has
provenance to source documents or accepted structured records.

## The app is the partition boundary

Recall filters by *scope* within an application; the application itself is the
*outer* fence. A customer runs several apps — HR, finance, engineering — that need
not know about one another, and pooling their memories would turn access control
(RBAC) into a query-time predicate a single bug could bypass. So `app_id` is the
**partition boundary** on every long-term record and every `recall`: the (designed)
`scope` (`user` / `project` / `organization` / `global`) is the finer filter
*inside* an app; `app_id` is the boundary *around* it.

In practice the boundary is enforced by the **store layout, not a query-time
predicate**: the factory hands `LongTermMemory` stores that are already prefixed to
the app, so isolation is a property of the collections it can address — not of
remembering to AND-in `app_id` on every query. The `app_id` stamped on each record
is for self-containment and audit. Physically the records still live in the shared
**system** stores — like the episodic log, long-term memory is system-level
infrastructure, not data bundled into one app.

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
[memory.md](memory.md#public-interface)). The two services below are the actual
shapes in the code (the stores arrive **already app-scoped**, so no `scope`
argument threads through — see the [scope note](#record-shape)):

```python
class LongTermMemory:  # cogbase/memory/long_term.py
    def __init__(self, structured_store, vector_store, llm, embedder, *,
                 app_id=None, recall_neighbors=5, reconcile_guidance=None) -> None: ...

    # --- recall (online, query-time) ---
    async def recall(self, *, query: str, limit: int = 5) -> list[LongTermRecord]:
        # vector hits + graph neighbors; the push path into the query runner
        ...
    async def lookup(self, *, query=None, kind=None, entities=None,
                     limit: int = 10) -> list[LongTermRecord]:
        # the pull path behind the memory_lookup tool
        ...

    # --- reconcile (the crux) ---
    async def reconcile(self, *, candidate, embeddings=None) -> str:
        # two-phase: recall related + LLM _decide ADD/UPDATE/DELETE/NOOP, apply
        ...
    async def reconcile_decided(self, *, candidate, embeddings=None) -> str:
        # single-call: apply the op the extractor already decided
        ...
    async def promote(self, *, candidate, status=None, embeddings=None) -> str:
        # the ADD path; status defaults to the kind+confidence gate
        ...
    async def embed_contents(self, candidates) -> dict[str, list[float]]:
        # batch-embed a session's candidates once, reused across reconcile
        ...

    # --- promotion review gate ---
    async def list_pending(self, *, kind=None, limit=50, offset=0) -> list[LongTermRecord]: ...
    async def review_many(self, *, decisions) -> list[ReviewResult]: ...

    # --- support for the distiller's auto-linking ---
    async def active_entity_frequencies(self) -> tuple[dict[str, int], int]: ...


class Distiller:  # cogbase/memory/distill.py
    def __init__(self, episodic, long_term, llm, *, min_confidence=None,
                 domain_fact_guidance=None, existing_memory_limit=10,
                 single_call=True, auto_link_max_entity_ratio=0.1) -> None: ...

    async def distill_session(self, *, session_id: str) -> list[str]:
        # replay past the watermark, extract candidates (front-loaded with existing
        # memories), auto-link, reconcile each, advance the watermark; returns ids
        ...
```

Placement: `cogbase/memory/long_term.py` (the `LongTermMemory` service),
`cogbase/memory/distill.py` (the offline `Distiller`), and
`cogbase/memory/projection.py` (the shared thread / watermark projection helpers,
also used by short-term memory). Adaptive evolution lives in its own package and is
out of scope here.

## Build order

The tier is implemented; this is the order it was built and where the open edges
remain.

1. ✅ Long-term record model and the `LongTermMemory` service over the structured
   and vector stores (`recall`, `promote`).
2. ✅ `reconcile`: related-records recall (vector ∪ entity overlap), LLM
   ADD / UPDATE / DELETE / NOOP, apply with provenance snapshot and confidence
   reinforcement. Later joined by the single-call `reconcile_decided`.
3. ✅ The `Distiller`: replay past the watermark, extract candidates (front-loaded
   with existing memories), auto-link, reconcile each.
4. ✅ The memory graph: LLM-emitted + deterministic `auto_link` edges, with
   neighbor traversal on recall.
5. ✅ Promotion review for gated kinds (`list_pending` / `review_many`).
6. ✅ The `lookup` pull path behind the `memory_lookup` tool.
7. ✅ Integration: distillation is triggered on session settle — the
   `POST .../sessions/{id}/close` endpoint (`close_session` in
   `api/routers/applications.py`) enqueues a background `distill_session` task — and
   `recall` is wired into the query runner's context assembly (`QueryRunner` calls
   `long_term.recall` per turn and exposes `memory_lookup` when
   `enable_memory_lookup` is set), behind the evidence policy. Both are wired by
   `api/factory.py`.
8. ⏳ Retention sweep for `superseded` / stale `pending_review` rows (nothing GCs
   the structured store yet).
9. ⏳ Finer `scope` (`user` / `project` / `organization` / `global`) within the app
   partition, when multi-user / RBAC is designed.

Reconciliation — the part with no analog in the document pipeline — was kept
isolated and testable before being wired toward the live query path.
