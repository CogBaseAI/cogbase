"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer) and a
``Runner`` (query layer) behind a small interface: ``setup`` → ``ingest`` /
``ingest_documents`` → ``query_stream``.

Typical usage::

    from cogbase.core.app import CogBaseApp
    from cogbase.core.models import Document
    from cogbase.pipeline.ingestion_pipeline import (
        IngestionPipeline, VectorCollection, StructuredCollection, SummarizeCollection,
    )

    pipeline = IngestionPipeline(
        name="legal",
        steps=[
            ("chunk-embed-upsert",     "document_chunks"),
            ("extract-structured",     "contracts"),
            ("summarize-embed-upsert", "document_summary"),
        ],
        vector_collections=[VectorCollection(schema=VectorCollectionSchema(name="document_chunks", dimensions=1536), ...)],
        structured_collections=[StructuredCollection(schema=..., ...)],
        summarize_collections=[SummarizeCollection(schema=VectorCollectionSchema(name="document_summary", dimensions=1536), ...)],
    )
    app = CogBaseApp("legal", llm, pipeline)
    await app.setup()
    await app.ingest_documents([Document(doc_id="c-001", text=contract_text)])
    async for item in app.query_stream("which contracts expire before 2026?"):
        ...
"""

from __future__ import annotations

import logging
from typing import Sequence

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult
from cogbase.core.models import Document
from cogbase.core.runner import RunResult, Runner
from cogbase.llms import LLMBase
from cogbase.stores import CollectionSchema, DocumentStoreBase

logger = logging.getLogger(__name__)


class CogBaseApp:
    """CogBase application: an ingestion pipeline + query runner under one object.

    Args:
        name:                        Logical name for the application.
        llm:                         LLM for query reasoning.
        pipeline:                    Fully configured ``IngestionPipeline`` carrying
                                     all vector, structured, and summarize collections.
        skills:                      Optional skills exposed to the query runner.
        passthrough_token_threshold: Estimated token count of structured lookup results
                                     above which records are returned directly without
                                     LLM synthesis.  Defaults to 2000.
        query_max_rounds:            Maximum LLM reasoning rounds per query.  Defaults to 5.
    """

    def __init__(
        self,
        name: str,
        llm: LLMBase,
        pipeline: IngestionPipeline,
        *,
        document_store: DocumentStoreBase | None = None,
        skills: list | None = None,
        passthrough_token_threshold: int = 2000,
        query_max_rounds: int = 5,
    ) -> None:
        self.name = name
        self._ingest_pipeline = pipeline
        self._document_store = document_store

        structured_store, vector_store, embedder, default_vc = pipeline.runner_resources()

        self._runner = Runner(
            llm=llm,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            default_vector_collection=default_vc,
            vector_collections=pipeline.vector_collection_infos or None,
            structured_schemas=pipeline.structured_schemas or None,
            passthrough_token_threshold=passthrough_token_threshold,
            max_calls=query_max_rounds,
            skills=skills,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all structured collections in their respective stores. Idempotent."""
        await self._ingest_pipeline.setup()

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

    @property
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections (convenience proxy)."""
        return self._ingest_pipeline.structured_schemas
