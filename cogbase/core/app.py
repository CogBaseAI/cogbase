"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer), a
``QueryRunner`` (query layer), and optional ``WorkflowRunner`` instances behind
a small interface: ``ingest_documents`` → ``query_stream`` / ``run_workflow``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence, TYPE_CHECKING

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import Document
from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.stores import DocumentStoreBase

if TYPE_CHECKING:
    from cogbase.workflows.runner import WorkflowRunner

logger = logging.getLogger(__name__)


class CogBaseApp:
    """CogBase application: ingestion pipeline + query runner + workflows.

    Args:
        name:             Logical name for the application.
        pipeline:         Fully configured ``IngestionPipeline`` (ingestion layer).
        runner:           Pre-built ``QueryRunner`` (query layer).
        document_store:   Optional document store for raw document persistence.
        workflow_runners: Named ``WorkflowRunner`` instances keyed by workflow name.
    """

    def __init__(
        self,
        name: str,
        pipeline: IngestionPipeline,
        runner: QueryRunner,
        *,
        document_store: DocumentStoreBase | None = None,
        workflow_runners: dict[str, "WorkflowRunner"] | None = None,
    ) -> None:
        self.name = name
        self._ingest_pipeline = pipeline
        self._runner = runner
        self._document_store = document_store
        self._workflows: dict[str, "WorkflowRunner"] = workflow_runners or {}

    async def ingest_documents(
        self,
        documents: Sequence[Document],
        *,
        concurrency: int = 5,
    ) -> list[IngestResult]:
        """Ingest a batch of documents, running up to *concurrency* at a time.

        When a document store is configured, each document is saved there first.
        A store save failure is captured as a failed ``IngestResult`` and that
        document is skipped by the pipeline.  A pipeline failure on one document
        does not abort the others.  Results are returned in the same order as
        *documents*.
        """
        logger.info("app.ingest_documents.start documents=%d concurrency=%d", len(documents), concurrency)

        store_failures: dict[str, Exception] = {}
        docs_to_process: list[Document] = list(documents)

        if self._document_store is not None:
            docs_to_process = []
            for doc in documents:
                try:
                    await self._document_store.save(self.name, doc.doc_id, doc.text)
                    docs_to_process.append(doc)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("app.ingest_documents.store_save_failed doc_id=%s", doc.doc_id)
                    store_failures[doc.doc_id] = exc

        pipeline_results = await self._ingest_pipeline.ingest_documents(docs_to_process, concurrency=concurrency)
        pipeline_by_id = {r.doc_id: r for r in pipeline_results}

        results = [
            IngestResult(doc_id=doc.doc_id, success=False, error=store_failures[doc.doc_id])
            if doc.doc_id in store_failures
            else pipeline_by_id[doc.doc_id]
            for doc in documents
        ]
        failures = sum(1 for r in results if not r.success)
        logger.info("app.ingest_documents.done documents=%d failures=%d", len(results), failures)

        # Fire after_ingest workflows for successfully ingested documents.
        for result in results:
            if not result.success:
                continue
            doc = next((d for d in documents if d.doc_id == result.doc_id), None)
            if doc is None:
                continue
            for wf_runner in self._workflows.values():
                trigger = wf_runner.workflow.trigger
                if trigger.type != "after_ingest":
                    continue
                when_meta = trigger.when.metadata if trigger.when else {}
                if not all(doc.metadata.get(k) == v for k, v in when_meta.items()):
                    continue
                # TODO the workflow task may fail, for example, node crashes, need to ensure the state is tracked
                asyncio.create_task(self._run_workflow_bg(wf_runner, {"doc_id": doc.doc_id}))

        return results

    async def _run_workflow_bg(self, wf_runner: "WorkflowRunner", params: dict[str, Any]) -> None:
        try:
            async for _ in wf_runner.run(params):
                pass
            logger.info(
                "app.after_ingest_workflow.done workflow=%s doc_id=%s",
                wf_runner.workflow.name, params.get("doc_id"),
            )
        except Exception:
            # TODO update task state
            logger.exception(
                "app.after_ingest_workflow.failed workflow=%s doc_id=%s",
                wf_runner.workflow.name, params.get("doc_id"),
            )

    async def query_stream(self, text: str):
        """Stream the answer token-by-token, then yield a final QueryResult.

        The retrieval loop runs until the LLM has enough evidence to answer or
        ``query_max_rounds`` is exhausted.  Large structured result sets are
        returned directly as formatted text (passthrough rule).
        """
        logger.info("app.query_stream.start query=%s", text[:200])
        async for chunk in self._runner.run(text):
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
    def ingestion_pipeline(self) -> IngestionPipeline:
        """The underlying ``IngestionPipeline`` (ingestion layer)."""
        return self._ingest_pipeline

    @property
    def query_runner(self) -> QueryRunner:
        """The underlying ``QueryRunner`` (query layer)."""
        return self._runner

    @property
    def document_store(self) -> DocumentStoreBase | None:
        """The document store, if configured."""
        return self._document_store
