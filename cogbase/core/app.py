"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer), a
``QueryRunner`` (query layer), and optional ``WorkflowRunner`` instances behind
a small interface: ``ingest_documents`` → ``query_stream`` / ``run_workflow``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence, TYPE_CHECKING

from cogbase.config.config import RoutingStrategy
from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import DocWorkflowStatus, Document
from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.stores import DocumentStoreBase, StructuredStoreBase
from cogbase.stores.filters import Col
from cogbase.workflows.context import render_value

if TYPE_CHECKING:
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
        document_store: DocumentStoreBase,
        structured_store: StructuredStoreBase,
        workflow_runners: dict[str, "WorkflowRunner"],
        llm: LLMBase,
        routing_strategy: RoutingStrategy = RoutingStrategy.AUTO,
        task_store: Any,
        query_prompt: str | None = None,
    ) -> None:
        self.name = name
        self._pipelines = pipelines
        self._runner = runner
        self._document_store = document_store
        self._structured_store = structured_store
        self._workflows: dict[str, "WorkflowRunner"] = workflow_runners
        self._llm = llm
        self._routing_strategy = routing_strategy
        self._task_store = task_store
        self._query_prompt = query_prompt

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
                await self._document_store.save(self.name, doc.doc_id, doc.text)
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
                        self.name, doc.doc_id, wf_runner.workflow.name, initial_status
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
                            self.name, wf_runner.workflow.name, doc.doc_id, _json.dumps(params)
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
                self.name, doc_id, wf_name,
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
        user_id: str | None = None,
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
            user_id:       Attribution carried onto the session's episodic events.
        """
        logger.info("app.query_stream.start query=%s session=%s", text[:200], session_id)
        effective_prompt = system_prompt or self._query_prompt
        kwargs = {"history": history, "top_k": top_k, "session_id": session_id, "user_id": user_id}
        if effective_prompt:
            kwargs["base_prompt"] = effective_prompt
        async for chunk in self._runner.run(text, **kwargs):
            yield chunk

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
