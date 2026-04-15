"""Application — top-level grouping of vector and structured collections.

An ``Application`` is the primary entry point for configuring CogBase.  It
bundles one or more collections (vector and/or structured) under a single name
and exposes ``setup`` and ``ingest`` as the two lifecycle methods.

Typical usage::

    from cogbase.core.application import Application, VectorCollection, StructuredCollection

    app = Application(
        name="legal",
        vector_collections=[
            VectorCollection(
                name="documents",
                store=FAISSVectorStore(dim=384),
                embedder=SentenceTransformersEmbedder(),
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

    await app.setup()          # idempotent — safe on every restart
    await app.ingest(text, doc_id="contract-001")

    # Pass schemas to the router so it can target the right collections:
    router = LLMRouter(client, model="...", schema=app.structured_schemas)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

from cogbase.core.models import Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase
from cogbase.stores.schema import CollectionSchema


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
        embedder: ``EmbedderBase`` implementation that produces dense vectors.
        chunker:  ``ChunkerBase`` implementation that splits document text.
    """

    name: str
    store: VectorStoreBase
    embedder: EmbedderBase
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


class Application:
    """Top-level entry point: a named set of vector and structured collections.

    An application groups all the stores, schemas, extractors, embedders, and
    chunkers needed for a single deployment under one object.  It exposes two
    lifecycle methods:

    - ``setup()`` — creates all structured collections (idempotent).
    - ``ingest(text, doc_id)`` — chunks, embeds, and extracts a document into
      every collection.

    The ``structured_schemas`` property returns the ``CollectionSchema`` list
    needed by ``LLMRouter`` so the router can reference the correct collections
    and fields when building query filters.

    Args:
        name:                   Logical name for the application.
        vector_collections:     Vector collections to manage.  Defaults to
                                an empty list (structured-only applications are
                                valid).
        structured_collections: Structured collections to manage.  Defaults to
                                an empty list (vector-only applications are
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

        Pass this to ``LLMRouter(schema=app.structured_schemas)`` so the router
        knows which collection names and field types are available.
        """
        return [sc.schema for sc in self._structured_collections]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all structured collections in their respective stores.

        This is idempotent — safe to call on every application startup.
        ``create_collection`` on each store will not overwrite existing data.
        """
        for sc in self._structured_collections:
            await sc.store.create_collection(sc.schema)

    async def ingest(self, text: str, doc_id: str) -> int:
        """Ingest a document into all collections.

        For each vector collection: chunk → embed → upsert.
        For each structured collection: extract → save.

        Empty text is a no-op for vector collections (no chunks produced).
        Extractors may still return records for structured collections even when
        the text is short, depending on the extractor's implementation.

        Args:
            text:   Full document text to ingest.
            doc_id: Stable identifier for the source document.

        Returns:
            Total number of records written across all structured collections.
        """
        for vc in self._vector_collections:
            chunks = vc.chunker.chunk(text, doc_id)
            if chunks:
                embedded = await vc.embedder.embed(chunks)
                await vc.store.upsert(embedded)

        total_records = 0
        for sc in self._structured_collections:
            record = await sc.extractor.extract(text, doc_id)
            if record is not None:
                await sc.store.save(sc.schema.name, [record])
                total_records += 1
        return total_records

    async def ingest_many(
        self,
        documents: Sequence[Document | tuple[str, str]],
        *,
        concurrency: int = 5,
    ) -> list[IngestResult]:
        """Ingest a sequence of documents, running up to *concurrency* at a time.

        Each document is processed independently.  A failure on one document does
        not abort the others — the error is captured in the corresponding
        ``IngestResult`` and ingestion continues for the remaining documents.
        Results are returned in the same order as *documents*.

        Args:
            documents:   Sequence of ``Document`` objects **or**
                         ``(text, doc_id)`` tuples (both forms are accepted).
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

        async def _ingest_one(doc: Document | tuple[str, str]) -> IngestResult:
            if isinstance(doc, tuple):
                text, doc_id = doc
            else:
                text, doc_id = doc.text, doc.doc_id

            async with semaphore:
                try:
                    records_extracted = await self.ingest(text, doc_id)
                    return IngestResult(
                        doc_id=doc_id,
                        success=True,
                        records_extracted=records_extracted,
                    )
                except Exception as exc:  # noqa: BLE001
                    return IngestResult(doc_id=doc_id, success=False, error=exc)

        return list(await asyncio.gather(*(_ingest_one(d) for d in documents)))
