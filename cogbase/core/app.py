"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer) and a
``QueryRunner`` (query layer) behind a small interface: ``ingest_documents`` →
``query_stream``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import Document
from cogbase.core.query_runner import QueryResult, QueryRunner
from cogbase.stores import DocumentStoreBase

logger = logging.getLogger(__name__)


class CogBaseApp:
    """CogBase application: an ingestion pipeline + query runner under one object.

    Args:
        name:      Logical name for the application.
        pipeline:  Fully configured ``IngestionPipeline`` (ingestion layer).
        runner:    Pre-built ``QueryRunner`` (query layer).
    """

    def __init__(
        self,
        name: str,
        pipeline: IngestionPipeline,
        runner: QueryRunner,
        *,
        document_store: DocumentStoreBase | None = None,
    ) -> None:
        self.name = name
        self._ingest_pipeline = pipeline
        self._runner = runner
        self._document_store = document_store

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
        return results

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
