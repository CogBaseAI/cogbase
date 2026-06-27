"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer), a
``QueryRunner`` (query layer), and optional ``WorkflowRunner`` instances behind
a small interface: ``ingest_documents`` → ``query_stream`` / ``run_workflow``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Sequence, TYPE_CHECKING
from uuid import uuid4

from cogbase.config.config import RoutingStrategy
from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import DocWorkflowStatus, Document
from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.stores import DocumentStoreBase, StructuredStoreBase
from cogbase.stores.filters import Col
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
    from cogbase.memory import (
        Distiller,
        EpisodicMemory,
        LongTermMemory,
        LongTermRecord,
        MemoryKind,
        MemoryStatus,
        ReviewDecision,
        ReviewResult,
        ShortTermMemory,
    )
    from cogbase.workflows.runner import WorkflowRunner

logger = logging.getLogger(__name__)

_ROUTING_SYSTEM_PROMPT = (
    "You are a document router. Classify the document excerpt below into exactly one of the "
    "following pipelines by responding with the pipeline name only — no explanation, no quotes, "
    "no other text.\n\nPipelines:\n{pipelines}"
)


class CogBaseApp:
    """CogBase application: ingestion pipelines + query runner + workflows.

    Args:
        name:             Logical name for the application.
        pipelines:        List of configured ``IngestionPipeline`` objects. Each
                          document is routed to the first pipeline whose ``match``
                          condition is satisfied; a pipeline with ``match=None``
                          accepts all documents.
        runner:           Pre-built ``QueryRunner`` (query layer).
        document_store:   Document store for raw document persistence.
        workflow_runners: Named ``WorkflowRunner`` instances keyed by workflow name.
    """

    def __init__(
        self,
        name: str,
        pipelines: list[IngestionPipeline],
        runner: QueryRunner,
        *,
        app_id: str,
        document_store: DocumentStoreBase,
        structured_store: StructuredStoreBase,
        workflow_runners: dict[str, "WorkflowRunner"],
        llm: LLMBase,
        routing_strategy: RoutingStrategy = RoutingStrategy.AUTO,
        task_store: Any,
        query_prompt: str | None = None,
        short_term: "ShortTermMemory | None" = None,
        episodic: "EpisodicMemory | None" = None,
        long_term: "LongTermMemory | None" = None,
        distiller: "Distiller | None" = None,
    ) -> None:
        self.name = name
        # Stable internal id — the per-app document-store collection key and the
        # storage identity that survives a rename of ``name``.
        self.app_id = app_id
        self._pipelines = pipelines
        self._runner = runner
        self._document_store = document_store
        self._structured_store = structured_store
        self._workflows: dict[str, "WorkflowRunner"] = workflow_runners
        self._llm = llm
        self._routing_strategy = routing_strategy
        self._task_store = task_store
        self._query_prompt = query_prompt
        # Memory tiers (all optional — wired only when the system stores back
        # them).  The runner already holds short/long-term for the query path;
        # the app holds them too so the session-lifecycle API can start/close
        # sessions and trigger distillation.
        self._short_term = short_term
        self._episodic = episodic
        self._long_term = long_term
        self._distiller = distiller

    def _find_pipeline_by_metadata(self, doc: Document) -> IngestionPipeline | None:
        for p in self._pipelines:
            if p.match is not None and all(doc.metadata.get(k) == v for k, v in p.match.items()):
                return p
        return None

    async def _find_pipeline_by_llm(self, doc: Document) -> IngestionPipeline | None:
        if not self._pipelines:
            return None
        pipeline_list = "\n".join(f"- {p.name}: {p.description}" for p in self._pipelines)
        messages: list[ChatMessage] = [
            {"role": "system", "content": _ROUTING_SYSTEM_PROMPT.format(pipelines=pipeline_list)},
            {"role": "user", "content": (doc.text or "")[:2000]},
        ]
        try:
            result = await self._llm.complete(messages, temperature=0.0)
            chosen = (result.get("content") or "").strip()
        except Exception:
            logger.exception("app.routing.llm_failed doc_id=%s", doc.doc_id)
            return None
        matched = next((p for p in self._pipelines if p.name == chosen), None)
        if matched is None:
            logger.warning("app.routing.llm_no_match doc_id=%s response=%r", doc.doc_id, chosen)
        else:
            logger.info("app.routing.llm_matched doc_id=%s pipeline=%s", doc.doc_id, matched.name)
        return matched

    async def _find_pipeline(self, doc: Document) -> IngestionPipeline | None:
        if len(self._pipelines) == 1:
            return self._pipelines[0]
        if self._routing_strategy == RoutingStrategy.LLM:
            return await self._find_pipeline_by_llm(doc)
        if self._routing_strategy == RoutingStrategy.AUTO:
            matched = self._find_pipeline_by_metadata(doc)
            if matched is not None:
                return matched
            logger.info("app.routing.metadata_miss doc_id=%s falling_back_to=llm", doc.doc_id)
            matched = await self._find_pipeline_by_llm(doc)
            if matched is not None and matched.match:
                for k, v in matched.match.items():
                    doc.metadata.setdefault(k, v)
            return matched
        return self._find_pipeline_by_metadata(doc)

    async def ingest_documents(
        self,
        documents: Sequence[Document],
    ) -> list[IngestResult]:
        """Ingest a batch of documents.

        When a document store is configured, each document is saved there first.
        A store save failure is captured as a failed ``IngestResult`` and that
        document is skipped by the pipeline.  A pipeline failure on one document
        does not abort the others.  Results are returned in the same order as
        *documents*.
        """
        logger.info(
            "app.ingest_documents.start app=%s documents=%d",
            self.name,
            len(documents),
        )

        store_failures: dict[str, Exception] = {}
        docs_to_process: list[Document] = []
        for doc in documents:
            try:
                await self._document_store.save(self.app_id, doc.doc_id, doc.text)
                docs_to_process.append(doc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("app.ingest_documents.store_save_failed doc_id=%s", doc.doc_id)
                store_failures[doc.doc_id] = exc

        pipeline_groups: dict[int, tuple[IngestionPipeline, list[Document]]] = {}
        unmatched: list[Document] = []
        for doc in docs_to_process:
            matched = await self._find_pipeline(doc)
            if matched is None:
                unmatched.append(doc)
            else:
                pid = id(matched)
                if pid not in pipeline_groups:
                    pipeline_groups[pid] = (matched, [])
                pipeline_groups[pid][1].append(doc)

        group_results_lists = await asyncio.gather(
            *(p.ingest_documents(docs) for p, docs in pipeline_groups.values())
        )
        results_by_id: dict[str, IngestResult] = {}
        for group_results in group_results_lists:
            for r in group_results:
                results_by_id[r.doc_id] = r
        pipeline_names = ", ".join(p.name for p in self._pipelines)
        for doc in unmatched:
            results_by_id[doc.doc_id] = IngestResult(
                doc_id=doc.doc_id,
                success=False,
                error=ValueError(
                    f"no pipeline matched doc_id={doc.doc_id!r} "
                    f"(tried: {pipeline_names}) — adjust routing_description on an existing "
                    f"pipeline or add a new pipeline for this document type"
                ),
            )

        results = [
            IngestResult(doc_id=doc.doc_id, success=False, error=store_failures[doc.doc_id])
            if doc.doc_id in store_failures
            else results_by_id[doc.doc_id]
            for doc in documents
        ]
        failures = sum(1 for r in results if not r.success)
        logger.info("app.ingest_documents.done documents=%d failures=%d", len(results), failures)

        import json as _json

        # For each successfully ingested document, determine which workflows apply
        # and mark them pending. Fire after_ingest workflows immediately.
        for result in results:
            if not result.success:
                continue
            doc = next((d for d in documents if d.doc_id == result.doc_id), None)
            if doc is None:
                continue
            for wf_runner in self._workflows.values():
                trigger = wf_runner.workflow.trigger
                when_meta = trigger.when.metadata if trigger.when else {}
                if not all(doc.metadata.get(k) == v for k, v in when_meta.items()):
                    continue
                try:
                    workflow_params = await self.resolve_workflow_params(wf_runner, doc.doc_id)
                except Exception:
                    logger.exception(
                        "app.workflow.params_failed workflow=%s doc_id=%s",
                        wf_runner.workflow.name, doc.doc_id,
                    )
                    continue
                if not workflow_params:
                    continue
                initial_status = (
                    DocWorkflowStatus.PENDING if trigger.type == "after_ingest"
                    else DocWorkflowStatus.READY
                )
                try:
                    await self._task_store.upsert_doc_workflow_status(
                        self.app_id, doc.doc_id, wf_runner.workflow.name, initial_status
                    )
                except Exception:
                    logger.exception(
                        "app.doc_workflow.upsert_failed workflow=%s doc_id=%s",
                        wf_runner.workflow.name, doc.doc_id,
                    )
                if trigger.type != "after_ingest":
                    continue

                # TODO create_workflow_task in batch
                task_params: list[tuple[dict, str | None]] = []
                for params in workflow_params:
                    task_id: str | None = None
                    try:
                        task_id = await self._task_store.create_workflow_task(
                            self.app_id, wf_runner.workflow.name, doc.doc_id, _json.dumps(params)
                        )
                    except Exception:
                        logger.exception(
                            "app.task_store.create_workflow_task.failed workflow=%s doc_id=%s",
                            wf_runner.workflow.name, doc.doc_id,
                        )
                    task_params.append((params, task_id))
                asyncio.create_task(
                    self._run_workflow_tasks_bg(wf_runner, doc.doc_id, task_params)
                )

        return results

    async def delete_document(self, doc_id: str) -> None:
        """Purge a document's ingested data from every pipeline and the doc store.

        Removes the document's vector chunks and structured records from every
        pipeline's collections, then deletes the parsed text the app persisted at
        ingest time.  A document is not tagged with the pipeline that ingested it,
        so every pipeline is purged; a ``doc_id`` absent from a collection is a
        no-op.  Does not touch the task/document registry — the API layer owns
        that, along with the raw uploaded file.
        """
        logger.info("app.delete_document.start app=%s doc_id=%s", self.name, doc_id)
        for pipeline in self._pipelines:
            await pipeline.purge_document(doc_id)
        try:
            await self._document_store.delete(self.app_id, doc_id)
        except Exception:
            # The derived stores are already purged; a stale parsed-text blob is a
            # storage leak, not a correctness problem, so don't fail the delete.
            logger.exception("app.delete_document.doc_store_delete_failed doc_id=%s", doc_id)

    async def resolve_workflow_params(
        self,
        wf_runner: "WorkflowRunner",
        doc_id: str,
    ) -> list[dict[str, Any]]:
        source = wf_runner.workflow.params_from_collection
        ctx = {"doc": {"doc_id": doc_id}}
        filter_values = render_value(source.filters, ctx)
        filters = [Col(field) == value for field, value in filter_values.items()]
        records = await self._structured_store.query(source.collection, filters)

        params_list: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for record in records:
            params = render_value(source.params, {**ctx, "record": record})
            if not isinstance(params, dict):
                raise ValueError(
                    f"params_from_collection for workflow "
                    f"{wf_runner.workflow.name!r} resolved to "
                    f"{type(params).__name__}, expected dict"
                )
            if source.distinct:
                key = tuple(sorted((str(k), repr(v)) for k, v in params.items()))
                if key in seen:
                    continue
                seen.add(key)
            params_list.append(params)
        return params_list

    async def _run_workflow_tasks_bg(
        self,
        wf_runner: "WorkflowRunner",
        doc_id: str,
        task_params: list[tuple[dict[str, Any], str | None]],
    ) -> None:
        """Run all param sets for a doc+workflow; update DocWorkflowRecord when all finish."""
        wf_name = wf_runner.workflow.name
        all_ok = True
        for params, task_id in task_params:
            try:
                async for _ in wf_runner.run(params):
                    pass
                if task_id is not None:
                    await self._task_store.complete_workflow_task(task_id, success=True)
            except Exception as exc:
                all_ok = False
                logger.exception(
                    "app.workflow.task_failed workflow=%s doc_id=%s", wf_name, doc_id
                )
                if task_id is not None:
                    await self._task_store.complete_workflow_task(task_id, success=False, error=str(exc))
        try:
            # TODO if failed, some items such as some clauses in a contract may be successfully processed,
            #      need to clean up the partial results.
            await self._task_store.upsert_doc_workflow_status(
                self.app_id, doc_id, wf_name,
                DocWorkflowStatus.DONE if all_ok else DocWorkflowStatus.FAILED,
            )
        except Exception:
            logger.exception(
                "app.doc_workflow.upsert_failed workflow=%s doc_id=%s", wf_name, doc_id
            )

    async def query_stream(
        self,
        text: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
        top_k: int = 10,
        session_id: str | None = None,
    ):
        """Stream the answer token-by-token, then yield a final QueryResult.

        The retrieval loop runs until the LLM has enough evidence to answer or
        ``query_max_rounds`` is exhausted.  Large structured result sets are
        returned directly as formatted text (passthrough rule).

        Args:
            system_prompt: When set, overrides the app-level ``query_prompt``
                           from the application config for this request only.
            top_k:         Default number of chunks per vector_search call.
            session_id:    When set and the runner has short-term memory wired,
                           the turn is recorded into that session and its
                           assembled context replaces ``history``.
        """
        logger.info("app.query_stream.start query=%s session=%s", text[:200], session_id)
        effective_prompt = system_prompt or self._query_prompt
        kwargs = {"history": history, "top_k": top_k, "session_id": session_id}
        if effective_prompt:
            kwargs["base_prompt"] = effective_prompt
        async for chunk in self._runner.run(text, **kwargs):
            yield chunk

    # ------------------------------------------------------------------
    # Session lifecycle (short-term + long-term memory)
    # ------------------------------------------------------------------

    @property
    def distiller(self) -> "Distiller | None":
        """The offline distiller, if long-term memory is wired."""
        return self._distiller

    async def pending_memories(
        self,
        *,
        kind: "MemoryKind | None" = None,
        limit: int = 50,
        offset: int = 0,
    ) -> "list[LongTermRecord]":
        """The gated long-term records awaiting review (oldest first).

        Raises if long-term memory is not configured — the review surface is
        meaningless without the store the gated records live in.
        """
        if self._long_term is None:
            raise RuntimeError("long-term memory is not configured")
        return await self._long_term.list_pending(kind=kind, limit=limit, offset=offset)

    async def memories(
        self,
        *,
        status: "MemoryStatus | None" = None,
        kind: "MemoryKind | None" = None,
        limit: int = 50,
        offset: int = 0,
    ) -> "list[LongTermRecord]":
        """Browse stored long-term records (most-recently-observed first).

        The inspection counterpart to :meth:`pending_memories`: ``status``/``kind``
        are optional filters; omit ``status`` to span every lifecycle state.
        Raises if long-term memory is not configured.
        """
        if self._long_term is None:
            raise RuntimeError("long-term memory is not configured")
        return await self._long_term.list_records(
            status=status, kind=kind, limit=limit, offset=offset
        )

    async def review_memories(
        self, *, decisions: "list[ReviewDecision]"
    ) -> "list[ReviewResult]":
        """Apply a batch of accept/reject verdicts to gated records.

        Raises ``RuntimeError`` when long-term memory is unconfigured and
        ``ValueError`` when the batch exceeds the service's cap.
        """
        if self._long_term is None:
            raise RuntimeError("long-term memory is not configured")
        return await self._long_term.review_many(decisions=decisions)

    async def add_memory(
        self,
        *,
        messages: "list[dict]",
        session_id: str | None = None,
        metadata: dict | None = None,
        observation_date: datetime | None = None,
    ) -> "tuple[str, list[LongTermRecord]]":
        """Add a batch of conversation messages to long-term memory.

        One self-contained "add memory" operation (mem0's ``add`` shape): append
        the messages to a session's episodic log, distill durable facts from it,
        and activate everything distilled so it is immediately recallable —
        bypassing the pending-review gate (an external conversation has no
        reviewer in the loop).  No session bookkeeping for the caller: when
        ``session_id`` is omitted a fresh one is generated and returned, so each
        call is an isolated, independently-distilled session.

        ``messages`` are ``{"role": "user"|"assistant", "content": str}`` dicts;
        roles map to the episodic continuity thread (user message / final answer)
        the distiller reads.  ``observation_date`` pins when the conversation took
        place so relative time references resolve correctly at distill time.

        Returns ``(session_id, records)`` where ``records`` are the long-term
        memories this call created or reinforced.  Raises if long-term memory is
        not configured.
        """
        if self._episodic is None or self._distiller is None or self._long_term is None:
            raise RuntimeError("long-term memory is not configured")

        from cogbase.memory import ReviewDecision

        sid = session_id or f"add-{uuid4().hex}"
        self._episodic.bind_app(sid, app_id=self.app_id)
        await self._episodic.record_session_started(
            session_id=sid,
            app_id=self.app_id,
            metadata=metadata,
            observation_date=observation_date,
        )
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            if msg.get("role") == "assistant":
                await self._episodic.record_final_answer(
                    session_id=sid, answer=content, observation_date=observation_date
                )
            else:
                await self._episodic.record_user_message(
                    session_id=sid, content=content, observation_date=observation_date
                )
        await self._episodic.flush(sid)

        memory_ids = await self._distiller.distill_session(session_id=sid)
        # Activate everything distilled: facts below the auto-promote threshold (and
        # all corrections) land in pending_review and stay out of recall otherwise.
        # accept() is a no-op ('skipped') for ids already active, so this is safe.
        if memory_ids:
            await self._long_term.review_many(
                decisions=[ReviewDecision(memory_id=mid, accept=True) for mid in memory_ids]
            )
        records = await self._long_term.get_records(memory_ids)
        logger.info(
            "app.add_memory session=%s messages=%d -> %d memory record(s)",
            sid, len(messages), len(records),
        )
        return sid, records

    async def start_session(
        self,
        *,
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Open (or resume) a session and return its id.

        Seeds the short-term metadata cache; the conversational thread lives in
        the episodic log and is materialised on the first query.  Raises if no
        short-term memory is configured (sessions are meaningless without it).
        """
        if self._short_term is None:
            raise RuntimeError("session lifecycle requires short-term memory to be configured")
        return await self._short_term.start_session(
            app_id=self.app_id,
            metadata=metadata,
            session_id=session_id,
        )

    async def close_session(self, session_id: str) -> bool:
        """Settle a session: evict the short-term cache and distill it.

        Evicts only the in-memory short-term cache (the durable episodic log is
        untouched), then — when a distiller is wired — runs distillation inline
        and returns whether it ran.  Callers wanting non-blocking settle should
        enqueue :meth:`Distiller.distill_session` as a background task instead and
        call :meth:`end_session`.
        """
        await self.end_session(session_id)
        if self._distiller is None:
            return False
        await self._distiller.distill_session(session_id=session_id)
        return True

    async def end_session(self, session_id: str) -> None:
        """Evict a session's short-term cache without distilling."""
        if self._short_term is not None:
            await self._short_term.end_session(session_id)

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    @property
    def workflows(self) -> list[str]:
        """Names of all registered workflows."""
        return list(self._workflows.keys())

    def get_workflow(self, name: str) -> "WorkflowRunner":
        """Return the named ``WorkflowRunner``, raising ``KeyError`` if absent."""
        try:
            return self._workflows[name]
        except KeyError:
            raise KeyError(f"Workflow '{name}' not found in app '{self.name}'")

    # ------------------------------------------------------------------
    # Accessors (advanced use)
    # ------------------------------------------------------------------

    @property
    def ingestion_pipelines(self) -> list[IngestionPipeline]:
        """The ingestion pipelines (ingestion layer)."""
        return list(self._pipelines)

    @property
    def query_runner(self) -> QueryRunner:
        """The underlying ``QueryRunner`` (query layer)."""
        return self._runner

    @property
    def document_store(self) -> DocumentStoreBase:
        return self._document_store
