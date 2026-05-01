"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer) and a
``Runner`` (query layer) behind a small interface: ``ingest_documents`` →
``query_stream``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import Document
from cogbase.core.runner import RunResult, Runner
from cogbase.stores import DocumentStoreBase

logger = logging.getLogger(__name__)


class CogBaseApp:
    """CogBase application: an ingestion pipeline + query runner under one object.

    Args:
        name:      Logical name for the application.
        pipeline:  Fully configured ``IngestionPipeline`` (ingestion layer).
        runner:    Pre-built ``Runner`` (query layer).
    """

    def __init__(
        self,
        name: str,
        pipeline: IngestionPipeline,
        runner: Runner,
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

        A failure on one document does not abort the others — the error is
        captured in the corresponding ``IngestResult``.  Results are returned
        in the same order as *documents*.
        """
        logger.info("app.ingest_documents.start documents=%d concurrency=%d", len(documents), concurrency)
        if self._document_store is not None:
            import asyncio
            await asyncio.gather(*(
                self._document_store.save(self.name, doc.doc_id, doc.text) for doc in documents
            ))
        results = await self._ingest_pipeline.ingest_documents(documents, concurrency=concurrency)
        failures = sum(1 for r in results if not r.success)
        logger.info("app.ingest_documents.done documents=%d failures=%d", len(results), failures)
        return results

    async def query_stream(self, text: str):
        """Stream the answer token-by-token, then yield a final RunResult.

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
    def query_runner(self) -> Runner:
        """The underlying ``Runner`` (query layer)."""
        return self._runner

    @property
    def document_store(self) -> DocumentStoreBase | None:
        """The document store, if configured."""
        return self._document_store
