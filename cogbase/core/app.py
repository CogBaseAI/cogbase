"""Generic CogBase application — bundles ingestion and query under one object.

``CogBaseApp`` wires together an ``IngestionPipeline`` (ingestion layer) and an
``Runner`` (query layer) behind a small interface: ``setup`` → ``ingest`` /
``ingest_documents`` → ``query``.

Typical usage::

    import openai
    from cogbase.llms import OpenAILLM
    from cogbase.core.app import CogBaseApp
    from cogbase.core.models import Document
    from cogbase.pipeline.extraction.llm import LLMExtractor
    from cogbase.stores.structured.sqlite import SQLiteStructuredStore
    from cogbase.stores.vector.faiss_store import FAISSVectorStore
    from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
    from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
    import json
    from cogbase.core.json_schema_to_basemodel import build_model_from_json_schema

    extraction_model = build_model_from_json_schema(extraction_json_schema_str)
    client = openai.AsyncOpenAI(api_key="...")
    llm = OpenAILLM(client, model="gpt-5.4")
    extractor = LLMExtractor(
        llm=llm,
        model="gpt-4o-mini",
        extraction_model=extraction_model,
        collection_name='your_collection_name',
    )
    app = CogBaseApp(
        llm=llm,
        model="gpt-4o-mini",
        extractor=extractor,
        structured_store=SQLiteStructuredStore("contracts.db"),
        vector_store=FAISSVectorStore(dim=384),
        embedder=SentenceTransformersEmbedding(),
        chunker=FixedSizeChunker(chunk_size=512, overlap=64),
    )
    await app.setup()
    results = await app.ingest_documents([Document(doc_id="c-001", text=contract_text)])
    async for item in app.query_stream("which contracts expire before 2026-01-01?"):
        if isinstance(item, str):
            print(item, end="", flush=True)
        else:
            print()
            print("answer:", item.answer)
            print("passthrough:", item.passthrough)
"""

from __future__ import annotations

import logging
from typing import Sequence

from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, IngestResult, StructuredCollection, VectorCollection
from cogbase.core.models import Document
from cogbase.core.runner import RunResult, Runner
from cogbase.embeddings import EmbeddingBase
from cogbase.llms import LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema

logger = logging.getLogger(__name__)


class CogBaseApp:
    """Generic CogBase application wiring ingestion and query together.

    Args:
        name:                        Logical name for the application.
        llm:                         LLM for query reasoning.
        extractor:                   Optional ``ExtractorBase`` for structured extraction.
        structured_store:            Persistent store for extracted records.
        vector_store:                Vector store for raw text chunks.  Must be provided
                                     together with *embedder* and *chunker*.  When
                                     ``None`` the app runs in structured-only mode.
        embedder:                    Embedder for chunked text.  Required with *vector_store*.
        chunker:                     Chunker for splitting text.  Required with *vector_store*.
        vector_collection_name:      Name used for the vector collection.
                                     Defaults to *name* when ``None``.
        passthrough_token_threshold: Estimated token count of structured results above
                                     which records are returned directly without LLM
                                     synthesis.  Defaults to 2000.
        query_max_rounds:            Maximum retrieval rounds per query.  Defaults to 5.

    Raises:
        ValueError: If only some of *vector_store*, *embedder*, *chunker* are
                    supplied — all three must be present or all absent.
    """

    def __init__(
        self,
        name: str,
        llm: LLMBase,
        extractor: ExtractorBase | None,
        structured_store: StructuredStoreBase | None,
        *,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbeddingBase | None = None,
        chunker: ChunkerBase | None = None,
        vector_collection_name: str | None = None,
        passthrough_token_threshold: int = 2000,
        query_max_rounds: int = 5,
    ) -> None:
        vector_params = (vector_store, embedder, chunker)
        n_provided = sum(p is not None for p in vector_params)
        if 0 < n_provided < 3:
            raise ValueError(
                "vector_store, embedder, and chunker must all be provided together "
                "or all omitted. Received a partial set."
            )

        structured_collection: StructuredCollection | None = None
        if structured_store is not None and extractor is not None:
            structured_collection = StructuredCollection(
                schema=extractor.schema,
                store=structured_store,
                extractor=extractor,
            )

        _vc_name = vector_collection_name or name
        vector_collection: VectorCollection | None = None
        if vector_store is not None:
            assert embedder is not None and chunker is not None  # validated above
            vector_collection = VectorCollection(
                name=_vc_name,
                store=vector_store,
                embedder=embedder,
                chunker=chunker,
            )

        self.name = name

        self._ingest_pipeline = IngestionPipeline(
            name=name,
            vector_collection=vector_collection,
            structured_collection=structured_collection,
        )

        self._runner = Runner(
            llm=llm,
            structured_store=structured_store,
            vector_store=vector_store,
            embedder=embedder,
            default_vector_collection=_vc_name,
            structured_schemas=self._ingest_pipeline.structured_schemas or None,
            passthrough_token_threshold=passthrough_token_threshold,
            max_calls=query_max_rounds,
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
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections (convenience proxy)."""
        return self._ingest_pipeline.structured_schemas
