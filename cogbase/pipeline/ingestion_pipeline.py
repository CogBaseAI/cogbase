"""IngestionPipeline — top-level grouping of vector and structured collections.

An ``IngestionPipeline`` is the primary entry point for configuring CogBase
ingestion.  It bundles one or more collections (vector and/or structured) under
a single name and exposes ``setup`` and ``ingest`` as the two lifecycle methods.

Typical usage::

    from cogbase.pipeline.ingestion_pipeline import IngestionPipeline, VectorCollection, StructuredCollection

    pipeline = IngestionPipeline(
        name="legal",
        vector_collections=[
            VectorCollection(
                name="documents",
                store=FAISSVectorStore(dim=384),
                embedder=SentenceTransformersEmbedding(),
                chunker=FixedSizeChunker(chunk_size=512, overlap=64),
            )
        ],
        structured_collections=[
            StructuredCollection(
                schema=clause_schema,
                store=SQLiteStructuredStore("data.db"),
                extractor=ClauseExtractor(),
            )
        ],
    )

    await pipeline._ingest(Document(doc_id="contract-001", text=contract_text))

    # Pass schemas to the router so it can target the right collections:
    router = LLMRouter(client, model="...", schema=pipeline.structured_schemas)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of ingesting a single document.

    Args:
        doc_id:            Identifier of the document that was processed.
        success:           ``True`` when ingestion completed without error.
        records_extracted: Total number of records written across all structured
                           collections (0 when no structured collections are
                           configured or the extractor produced no output).
        error:             The exception raised, when *success* is ``False``.
    """

    doc_id: str
    success: bool
    records_extracted: int = 0
    error: Exception | None = field(default=None, repr=False)


@dataclass
class VectorCollection:
    """A named vector collection backed by a store, embedder, and chunker.

    Args:
        name:     Logical name for this collection (used for lookup and logging).
        store:    ``VectorStoreBase`` implementation that persists chunks.
        embedder: ``EmbeddingBase`` implementation that produces dense vectors.
        chunker:  ``ChunkerBase`` implementation that splits document text.
    """

    name: str
    store: VectorStoreBase
    embedder: EmbeddingBase
    chunker: ChunkerBase


@dataclass
class StructuredCollection:
    """A structured collection backed by a store, schema, and extractor.

    The collection name is taken from ``schema.name`` — no separate ``name``
    field is needed.

    Args:
        schema:    ``CollectionSchema`` describing the table and its fields.
        store:     ``StructuredStoreBase`` implementation that persists records.
        extractor: ``ExtractorBase`` implementation that extracts records from
                   document text.  ``extractor.collection`` must match
                   ``schema.name``; this is validated at construction time.
    """

    schema: CollectionSchema
    store: StructuredStoreBase
    extractor: ExtractorBase

    def __post_init__(self) -> None:
        if self.extractor.collection != self.schema.name:
            raise ValueError(
                f"StructuredCollection extractor.collection '{self.extractor.collection}' "
                f"does not match schema.name '{self.schema.name}'"
            )

    @property
    def name(self) -> str:
        """Collection name, taken from the schema."""
        return self.schema.name


class IngestionPipeline:
    """Top-level entry point: a named set of vector and structured collections.

    An ingestion pipeline groups all the stores, schemas, extractors, embedders,
    and chunkers needed for a single deployment under one object.  It exposes one
    lifecycle method:

    - ``ingest(doc)`` — chunks, embeds, and extracts a document into every
      collection.

    The ``structured_schemas`` property returns the ``CollectionSchema`` list
    needed by ``LLMRouter`` so the router can reference the correct collections
    and fields when building query filters.

    Args:
        name:                   Logical name for the pipeline.
        vector_collections:     Vector collections to manage.  Defaults to
                                an empty list (structured-only pipelines are
                                valid).
        structured_collections: Structured collections to manage.  Defaults to
                                an empty list (vector-only pipelines are
                                valid).
    """

    def __init__(
        self,
        name: str,
        vector_collections: list[VectorCollection] | None = None,
        structured_collections: list[StructuredCollection] | None = None,
    ) -> None:
        self.name = name
        self._vector_collections: list[VectorCollection] = vector_collections or []
        self._structured_collections: list[StructuredCollection] = structured_collections or []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def vector_collections(self) -> list[VectorCollection]:
        """Read-only view of the registered vector collections."""
        return list(self._vector_collections)

    @property
    def structured_collections(self) -> list[StructuredCollection]:
        """Read-only view of the registered structured collections."""
        return list(self._structured_collections)

    @property
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections.

        Pass this to ``LLMRouter(schema=pipeline.structured_schemas)`` so the
        router knows which collection names and field types are available.
        """
        return [sc.schema for sc in self._structured_collections]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all structured collections in their respective stores. Idempotent."""
        logger.info(
            "ingestion_pipeline.setup.start name=%s structured_collections=%d",
            self.name,
            len(self._structured_collections),
        )
        for sc in self._structured_collections:
            logger.debug("ingestion_pipeline.setup.create_collection name=%s collection=%s", self.name, sc.name)
            await sc.store.create_collection(sc.schema)
        logger.info("ingestion_pipeline.setup.done name=%s", self.name)

    async def _ingest(self, doc: Document) -> int:
        """Ingest a document into all collections.

        For each vector collection: chunk → embed → upsert.
        For each structured collection: extract → save.

        Empty text is a no-op for vector collections (no chunks produced).
        Extractors may still return records for structured collections even when
        the text is short, depending on the extractor's implementation.

        Args:
            doc: Document to ingest.

        Returns:
            Total number of records written across all structured collections.
        """
        logger.info("ingestion_pipeline.ingest.start name=%s doc_id=%s", self.name, doc.doc_id)
        for vc in self._vector_collections:
            chunks = vc.chunker.chunk(doc)
            logger.debug(
                "ingestion_pipeline.ingest.vector_chunked name=%s doc_id=%s collection=%s chunks=%d",
                self.name,
                doc.doc_id,
                vc.name,
                len(chunks),
            )
            if chunks:
                embedded = await vc.embedder.embed(chunks)
                await vc.store.upsert(embedded)
                logger.debug(
                    "ingestion_pipeline.ingest.vector_upserted name=%s doc_id=%s collection=%s embedded=%d",
                    self.name,
                    doc.doc_id,
                    vc.name,
                    len(embedded),
                )

        total_records = 0
        for sc in self._structured_collections:
            record = await sc.extractor.extract(doc)
            if record is not None:
                await sc.store.save(sc.schema.name, [record])
                total_records += 1
                logger.debug(
                    "ingestion_pipeline.ingest.structured_saved name=%s doc_id=%s collection=%s",
                    self.name,
                    doc.doc_id,
                    sc.name,
                )
        logger.info(
            "ingestion_pipeline.ingest.done name=%s doc_id=%s records_extracted=%d",
            self.name,
            doc.doc_id,
            total_records,
        )
        return total_records

    async def ingest_many(
        self,
        documents: Sequence[Document],
        *,
        concurrency: int = 5,
    ) -> list[IngestResult]:
        """Ingest a sequence of documents, running up to *concurrency* at a time.

        Each document is processed independently.  A failure on one document does
        not abort the others — the error is captured in the corresponding
        ``IngestResult`` and ingestion continues for the remaining documents.
        Results are returned in the same order as *documents*.

        Args:
            documents:   Sequence of ``Document`` objects to ingest.
            concurrency: Maximum number of documents ingested simultaneously.
                         Defaults to ``5`` — a safe limit for LLM API rate caps.
                         Set to ``1`` for strictly sequential ingestion.

        Returns:
            ``list[IngestResult]`` in input order, one entry per document.

        Raises:
            ValueError: If *concurrency* is less than 1.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)

        async def _ingest_one(doc: Document) -> IngestResult:
            async with semaphore:
                try:
                    records_extracted = await self._ingest(doc)
                    return IngestResult(
                        doc_id=doc.doc_id,
                        success=True,
                        records_extracted=records_extracted,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "ingestion_pipeline.ingest_many.failed name=%s doc_id=%s",
                        self.name,
                        doc.doc_id,
                    )
                    return IngestResult(doc_id=doc.doc_id, success=False, error=exc)

        return list(await asyncio.gather(*(_ingest_one(d) for d in documents)))
