"""Durable execution of background tasks (ingest, distill, workflow).

Each ``run_*_task`` coroutine drives one task record from ``PENDING`` to a
terminal ``DONE``/``FAILED`` state.  The same coroutines back both the live
request path (upload / close-session / after-ingest) and the startup recovery
sweep, so the two can never diverge.

``recover_orphaned_tasks`` is what makes background work durable across
restarts: an in-process ``asyncio.create_task`` is lost if the node crashes,
deploys, or OOMs mid-flight, leaving its task record stuck in ``PENDING`` /
``RUNNING`` forever.  On startup we requeue those.  Re-execution is safe because
ingestion is idempotent — chunk ids are deterministic and both vector ``upsert``
and structured ``save`` are upsert-by-primary-key.

Concurrency model: this is **single-node, at-least-once** recovery. TODO In a
multi-node deployment two nodes could both recover the same task; correctness
then relies on idempotent re-execution.  Exactly-once across nodes needs task
leasing (an atomic ``PENDING→RUNNING`` claim carrying an owner id and lease
expiry); that is intentionally deferred.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from api.app_cache import AppCache, cache_key
from api.models import IngestResultSummary
from api.system_store import AppRecord, DocRecord, SystemStore, TaskStatus
from cogbase.core.models import Document, DocWorkflowStatus
from cogbase.pipeline.document_parser import parse_to_markdown

logger = logging.getLogger(__name__)

# Default fan-out for background task execution.
# TODO make configurable (also used by the upload endpoint).
DEFAULT_TASK_CONCURRENCY = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-task executors
# ---------------------------------------------------------------------------


async def run_ingest_task(
    task_id: str,
    *,
    app,
    app_name: str,
    app_cache: AppCache,
    app_id: str,
    system_store: SystemStore,
) -> None:
    """Load a document's bytes, parse, ingest, and record the outcome.

    Self-contained: everything needed is reconstructed from the task record
    (``doc_path`` + ``doc_metadata`` in ``params_json``) and the document store,
    which is what lets recovery re-run it after a crash.  ``app`` is the
    best-known instance; we re-resolve from the cache defensively in case the
    app was rebuilt (e.g. config update) since the task was queued.
    """
    task = await system_store.get_task(task_id)
    if task is None:
        return

    try:
        params = json.loads(task.params_json) if task.params_json else {}
    except Exception:
        params = {}

    doc_path = params.get("doc_path", "")
    doc_metadata = params.get("doc_metadata", {})
    doc_id = task.doc_id or ""
    filename = doc_metadata.get("source_filename", doc_id)

    key = cache_key(task.account_id, task.namespace_id, app_name)
    try:
        current_app = app_cache.get(key) or app
        content = await current_app.document_store.load_bytes(app_id, doc_path)
    except Exception as exc:
        await system_store.update_task(
            task_id, status=TaskStatus.FAILED, completed_at=_now(),
            error=f"Failed to load document bytes: {exc}",
        )
        return

    try:
        markdown_text = parse_to_markdown(content, filename)
    except Exception as exc:
        await system_store.update_task(
            task_id, status=TaskStatus.FAILED, completed_at=_now(),
            error=f"Failed to parse {filename!r}: {exc}",
        )
        return

    doc = Document(doc_id=doc_id, text=markdown_text, metadata=doc_metadata)
    await system_store.update_task(task_id, status=TaskStatus.RUNNING, started_at=_now())
    try:
        current_app = app_cache.get(key) or app
        results = await current_app.ingest_documents([doc])
        result = results[0]
        if result.success:
            # A successful ingest that lands nothing usually means the document
            # had no extractable text (scanned/image-only PDF) or no pipeline
            # step matched it. Surface that as a warning on the otherwise-'done'
            # task rather than reporting silent success.
            warning = None
            if result.ingested_nothing:
                warning = (
                    "no text could be extracted — the document may be a "
                    "scanned/image-only PDF or otherwise empty; nothing was ingested"
                    if not markdown_text.strip()
                    else "document parsed but produced no chunks or records; "
                    "check that a pipeline matches this document type"
                )
                logger.warning(
                    "ingest_task ingested nothing app=%s doc_id=%s: %s",
                    app_name, doc_id, warning,
                )
            summary = IngestResultSummary(
                chunks_written=result.chunks_written,
                records_extracted=result.records_extracted,
                warning=warning,
            )
            await system_store.update_task(
                task_id, status=TaskStatus.DONE, completed_at=_now(),
                result_json=summary.model_dump_json(),
            )
            await system_store.save_doc(DocRecord(
                account_id=task.account_id,
                namespace_id=task.namespace_id,
                app_id=app_id,
                doc_id=doc.doc_id,
                status="active",
                ingested_at=_now(),
                metadata=json.dumps(doc.metadata) if doc.metadata else None,
            ))
        else:
            await system_store.update_task(
                task_id, status=TaskStatus.FAILED, completed_at=_now(),
                error=str(result.error) if result.error else "ingest failed",
            )
    except Exception as exc:
        logger.exception("ingest_task failed app=%s doc_id=%s", app_name, doc_id)
        await system_store.update_task(task_id, status=TaskStatus.FAILED, completed_at=_now(), error=str(exc))


async def run_distill_task(task_id: str, *, app, system_store: SystemStore) -> None:
    """Run long-term distillation for the session carried in ``params_json``."""
    task = await system_store.get_task(task_id)
    if task is None:
        return

    distiller = app.distiller
    if distiller is None:
        await system_store.complete_distill_task(
            task_id, success=False,
            error="no distiller configured for this application",
        )
        return

    try:
        params = json.loads(task.params_json) if task.params_json else {}
    except Exception:
        params = {}
    session_id = params.get("session_id") or task.doc_id or ""

    await system_store.start_task(task_id)
    try:
        await distiller.distill_session(session_id=session_id)
        await system_store.complete_distill_task(task_id, success=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("distillation failed for session '%s'", session_id)
        await system_store.complete_distill_task(task_id, success=False, error=str(exc))


async def run_workflow_task(task_id: str, *, app, system_store: SystemStore) -> None:
    """Run one workflow param-set, then roll up the doc's workflow status."""
    task = await system_store.get_task(task_id)
    if task is None:
        return

    workflow_name = task.task_name
    try:
        wf_runner = app.get_workflow(workflow_name)
    except KeyError:
        await system_store.complete_workflow_task(
            task_id, success=False,
            error=f"workflow {workflow_name!r} not found",
        )
        return

    try:
        params = json.loads(task.params_json) if task.params_json else {}
    except Exception:
        params = {}

    await system_store.update_task(task_id, status=TaskStatus.RUNNING, started_at=_now())
    try:
        async for _ in wf_runner.run(params):
            pass
        await system_store.complete_workflow_task(task_id, success=True)
    except Exception as exc:
        logger.exception(
            "workflow_task failed app=%s workflow=%s task=%s",
            app.name, workflow_name, task_id,
        )
        await system_store.complete_workflow_task(task_id, success=False, error=str(exc))

    if task.doc_id:
        await _finalize_doc_workflow_if_settled(
            app=app, system_store=system_store,
            doc_id=task.doc_id, workflow_name=workflow_name,
        )


async def _finalize_doc_workflow_if_settled(
    *, app, system_store: SystemStore, doc_id: str, workflow_name: str,
) -> None:
    """Set DONE/FAILED on a doc's workflow once no tasks remain in flight.

    Best-effort: a doc may have several param-sets (tasks) for one workflow; we
    only resolve the rolled-up status when none are still PENDING/RUNNING.
    """
    try:
        tasks = await system_store.list_tasks(
            app.app_id, task_type="workflow", task_name=workflow_name, doc_id=doc_id,
        )
    except Exception:
        logger.exception(
            "doc_workflow rollup query failed workflow=%s doc_id=%s", workflow_name, doc_id
        )
        return

    if any(t.status in (TaskStatus.PENDING, TaskStatus.RUNNING) for t in tasks):
        return

    all_ok = not any(t.status == TaskStatus.FAILED for t in tasks)
    try:
        await system_store.upsert_doc_workflow_status(
            app.account_id, app.namespace_id, app.app_id, doc_id, workflow_name,
            DocWorkflowStatus.DONE if all_ok else DocWorkflowStatus.FAILED,
        )
    except Exception:
        logger.exception(
            "doc_workflow rollup upsert failed workflow=%s doc_id=%s", workflow_name, doc_id
        )


# ---------------------------------------------------------------------------
# Startup recovery sweep
# ---------------------------------------------------------------------------


async def recover_orphaned_tasks(
    system_store: SystemStore,
    resolve_app: Callable[[AppRecord], Awaitable[object | None]],
    app_cache: AppCache,
    *,
    concurrency: int = DEFAULT_TASK_CONCURRENCY,
) -> int:
    """Requeue tasks left unfinished by a previous process.

    Sweeps every active application in every account/namespace. RUNNING tasks
    (interrupted mid-flight) are reset to PENDING, then all PENDING tasks are
    dispatched to their executor.  ``resolve_app`` maps an ``AppRecord`` to a live
    instance (cache hit or rebuild), returning ``None`` if it cannot be resolved.
    Returns the number of tasks requeued.
    """
    apps = await system_store.list_apps()  # all accounts/namespaces
    semaphore = asyncio.Semaphore(concurrency)
    coros: list[Awaitable[None]] = []

    for record in apps:
        if record.status != "active":
            continue
        try:
            app = await resolve_app(record)
        except Exception:
            logger.exception("recover_orphaned_tasks: failed to resolve app=%s", record.name)
            app = None
        if app is None:
            logger.warning("recover_orphaned_tasks: skipping unresolved app=%s", record.name)
            continue

        running = await system_store.list_tasks(record.app_id, status=TaskStatus.RUNNING)
        for task in running:
            logger.info(
                "recovered interrupted task id=%s type=%s app=%s",
                task.task_id, task.task_type, record.name,
            )
            await system_store.update_task(task.task_id, status=TaskStatus.PENDING)

        pending = await system_store.list_tasks(record.app_id, status=TaskStatus.PENDING)
        for task in pending:
            coro = _dispatch_task(
                task=task, app=app, app_name=record.name, app_id=record.app_id,
                app_cache=app_cache, system_store=system_store, semaphore=semaphore,
            )
            if coro is not None:
                coros.append(coro)

    if not coros:
        logger.info("recover_orphaned_tasks: no orphaned tasks found")
        return 0

    logger.info("recover_orphaned_tasks: requeuing %d task(s)", len(coros))
    await asyncio.gather(*coros)
    return len(coros)


def _dispatch_task(
    *, task, app, app_name, app_id, app_cache, system_store, semaphore,
) -> Awaitable[None] | None:
    """Return a semaphore-bounded coroutine running ``task`` via its executor."""

    async def _run() -> None:
        async with semaphore:
            if task.task_type == "ingest":
                await run_ingest_task(
                    task.task_id, app=app, app_name=app_name,
                    app_cache=app_cache, app_id=app_id, system_store=system_store,
                )
            elif task.task_type == "distill":
                await run_distill_task(task.task_id, app=app, system_store=system_store)
            elif task.task_type == "workflow":
                await run_workflow_task(task.task_id, app=app, system_store=system_store)
            else:
                logger.warning(
                    "recover_orphaned_tasks: unknown task_type=%s id=%s — skipping",
                    task.task_type, task.task_id,
                )

    return _run()
