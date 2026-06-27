# Doc Registry

The doc registry is the canonical document catalog for an application. It answers "what documents does this app know about?"

## Three stores, three concerns

| Store | Answers |
|---|---|
| **Doc registry** (`DocRecord`) | What docs does this app know about? |
| **Ingest tasks** (`TaskRecord`, `task_type=ingest`) | What ingest operations ran, and did they succeed? |
| **Workflow tasks** (`TaskRecord`, `task_type=workflow`) | What workflow processing has been applied to each doc? |

Keeping these separate means ingest tasks become audit logs once complete — they can be pruned without affecting the document inventory or workflow history.

## Task durability

Background tasks (ingest, distill, workflow) are run in-process. The task record is the durable handle: it is created `PENDING`, flipped to `RUNNING` while executing, and settled to `DONE`/`FAILED`. If the node crashes, deploys, or OOMs mid-flight, the in-process coroutine is lost but the record survives. On startup, `recover_orphaned_tasks` (`api/task_runner.py`) sweeps every active app, resets `RUNNING` tasks back to `PENDING`, and re-dispatches all `PENDING` tasks through the same executors used by the live request path. Re-execution is safe because ingestion is idempotent (deterministic chunk ids; upsert-by-primary-key in both vector and structured stores).

This recovery is **single-node, at-least-once**. In a multi-node deployment two nodes could recover the same task concurrently; correctness then rests on idempotent re-execution. Exactly-once across nodes requires task leasing (an atomic `PENDING→RUNNING` claim carrying an owner id and lease expiry) and is not yet implemented.

## DocRecord

A `DocRecord` is written on successful ingest completion. Fields:

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Primary key within an app |
| `app_name` | string | Owning application |
| `status` | string | `active` \| `failed` \| `deleted` |
| `ingested_at` | string | ISO-8601 UTC timestamp |
| `metadata` | string | JSON blob — filename, source format, pipeline-routing metadata, etc. |

## Workflow tasks and the doc registry

Workflow tasks carry `doc_id` as a foreign key to `DocRecord`. This enables three operations that were previously awkward or impossible:

**Unprocessed docs query** — a server-side LEFT JOIN between the doc registry and workflow tasks returns docs that have no completed workflow task for a given workflow. No client-side diffing needed.

**Deletion anchor** — `DELETE /applications/{name}/docs/{doc_id}` deletes the `DocRecord`, cascades to workflow tasks, and drives cleanup of associated vector and structured store data.

**Re-runs** — a new workflow task can be created against any existing `DocRecord`, regardless of when the doc was originally ingested.

## API endpoints

```
GET    /applications/{name}/docs
       → list all docs; filterable by status

GET    /applications/{name}/docs/{doc_id}
       → single DocRecord

DELETE /applications/{name}/docs/{doc_id}
       → delete record, cascade workflow tasks, clean up stores

GET    /applications/{name}/workflows/{workflow}/docs?status=unprocessed
       → docs with no completed workflow task for this workflow
```

## Manual trigger workflows

`after_ingest` is sufficient for per-document workflows. Manual trigger remains necessary in three cases:

**Cross-document / aggregate workflows** — a portfolio summary after all company docs are loaded, for example. Running after each ingest produces partial results; the right moment to trigger is when the batch is complete.

**Workflow logic changes and re-runs** — when a prompt or schema is updated, already-ingested documents need to be reprocessed. `after_ingest` never fires again for those documents; the unprocessed-docs query provides a clean list to drive re-runs.

**Cost / rate control** — expensive multi-step LLM chains that should be batched manually or gated on human review.
