"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``Application`` (ingestion layer) and an
``Engine`` (query layer) behind a small interface: ``setup`` → ``ingest`` /
``ingest_many`` → ``query``.

Typical usage::

    import openai
    from cogbase.core.app import CogBaseApp
    from cogbase.core.models import Document
    from cogbase.pipeline.extraction.llm import LLMExtractor
    from cogbase.stores.structured.sqlite import SQLiteStructuredStore
    from cogbase.stores.vector.faiss_store import FAISSVectorStore
    from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
    from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
    from examples.contract_analyst_demo.schema import ContractExtraction, CONTRACTS_COLLECTION

    client = openai.AsyncOpenAI(api_key="...")
    extractor = LLMExtractor(
        client=client,
        model="gpt-4o-mini",
        extraction_model=ContractExtraction,
        collection_name=CONTRACTS_COLLECTION,
        id_field="contract_id",
    )
    app = CogBaseApp(
        client=client,
        model="gpt-4o-mini",
        extractors=[extractor],
        structured_store=SQLiteStructuredStore("contracts.db"),
        vector_store=FAISSVectorStore(dim=384),
        embedder=SentenceTransformersEmbedding(),
        chunker=FixedSizeChunker(chunk_size=512, overlap=64),
    )
    await app.setup()
    results = await app.ingest_many([Document(doc_id="c-001", text=contract_text)])
    result = await app.query("which contracts expire before 2026-01-01?")
    print(result.answer)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from cogbase.core.application import Application, IngestResult, StructuredCollection, VectorCollection
from cogbase.core.models import Document
from cogbase.engine.engine import Engine
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.generation.llm import LLMGenerator
from cogbase.engine.retrieval.hybrid import HybridRetriever
from cogbase.engine.router import LLMRouter, QueryPattern
from cogbase.embeddings import EmbeddingBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema

logger = logging.getLogger(__name__)

# Patterns available when no vector store is configured (B and C require one).
_STRUCTURED_ONLY_PATTERNS = [QueryPattern.A, QueryPattern.D]


class CogBaseApp:
    """Generic CogBase application wiring ingestion and query together.

    Args:
        client:               Async OpenAI-compatible client for the router and
                              generator.
        model:                Model name forwarded to the router and generator.
        extractors:           One or more ``ExtractorBase`` instances.  Each
                              extractor writes to its own structured collection.
        structured_store:     Persistent store for extracted records.
        vector_store:         Vector store for raw text chunks.  Must be provided
                              together with *embedder* and *chunker*.  When
                              ``None`` the app runs in structured-only mode.
        embedder:             Embedder for chunked text.  Required with *vector_store*.
        chunker:              Chunker for splitting text.  Required with *vector_store*.
        name:                 Logical name for the application.
        generator_max_tokens: Max tokens for the ``LLMGenerator`` LLM call.
        retriever_top_k:      Nearest-neighbour chunks returned per semantic query.

    Raises:
        ValueError: If only some of *vector_store*, *embedder*, *chunker* are
                    supplied — all three must be present or all absent.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        extractors: list[ExtractorBase],
        structured_store: StructuredStoreBase,
        *,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        chunker: ChunkerBase | None = None,
        name: str = "app",
        generator_max_tokens: int = 4096,
        retriever_top_k: int = 10,
    ) -> None:
        vector_params = (vector_store, embedder, chunker)
        n_provided = sum(p is not None for p in vector_params)
        if 0 < n_provided < 3:
            raise ValueError(
                "vector_store, embedder, and chunker must all be provided together "
                "or all omitted. Received a partial set."
            )

        structured_collections = [
            StructuredCollection(
                schema=extractor.schema,
                store=structured_store,
                extractor=extractor,
            )
            for extractor in extractors
        ]

        vector_collections: list[VectorCollection] = []
        if vector_store is not None:
            assert embedder is not None and chunker is not None  # validated above
            vector_collections.append(
                VectorCollection(
                    name="documents",
                    store=vector_store,
                    embedder=embedder,
                    chunker=chunker,
                )
            )

        self._app = Application(
            name=name,
            vector_collections=vector_collections,
            structured_collections=structured_collections,
        )

        self._engine = Engine(
            router=LLMRouter(
                client,
                model,
                schema=self._app.structured_schemas,
                available_patterns=None if vector_store else _STRUCTURED_ONLY_PATTERNS,
            ),
            retriever=HybridRetriever(
                structured_store=structured_store,
                vector_store=vector_store,
                embedder=embedder,
                top_k=retriever_top_k,
            ),
            generator=LLMGenerator(client, model, max_tokens=generator_max_tokens),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all collections in their respective stores. Idempotent."""
        logger.info("app.setup.start name=%s", self._app.name)
        await self._app.setup()
        logger.info("app.setup.done name=%s", self._app.name)

    async def ingest(self, doc: Document) -> None:
        """Ingest a single document.

        Chunks and embeds the text into the vector store (if configured) and
        runs structured extraction into the structured store.
        """
        logger.info("app.ingest.start doc_id=%s", doc.doc_id)
        await self._app.ingest(doc)
        logger.info("app.ingest.done doc_id=%s", doc.doc_id)

    async def ingest_many(
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
        logger.info("app.ingest_many.start documents=%d concurrency=%d", len(documents), concurrency)
        results = await self._app.ingest_many(documents, concurrency=concurrency)
        failures = sum(1 for r in results if not r.success)
        logger.info("app.ingest_many.done documents=%d failures=%d", len(results), failures)
        return results

    async def query(self, text: str) -> GenerationResult:
        """Answer a natural-language query over ingested documents.

        Automatically routes to the correct retrieval pattern:

        - Pattern A — structured lookup (no LLM call needed)
        - Pattern B — semantic search over raw text
        - Pattern C — hybrid reasoning across structured records and text
        - Pattern D — grounded report with ``[FINDINGS]`` / ``[SUPPORTING_QUOTES]``
        """
        logger.info("app.query.start query=%s", text[:200])
        result = await self._engine.query(text)
        logger.info("app.query.done answer=%s", result.answer[:200])
        return result

    # ------------------------------------------------------------------
    # Accessors (advanced use)
    # ------------------------------------------------------------------

    @property
    def application(self) -> Application:
        """The underlying ``Application`` (ingestion layer)."""
        return self._app

    @property
    def engine(self) -> Engine:
        """The underlying ``Engine`` (query layer)."""
        return self._engine

    @property
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections (convenience proxy)."""
        return self._app.structured_schemas
