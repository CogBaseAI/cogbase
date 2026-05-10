"""IngestionPipeline — ordered steps over multiple vector and structured collections.

Supports three step types:

- ``chunk-embed-upsert``    — chunk document text, embed, upsert to a vector collection
- ``extract-structured``    — LLM extraction → save to a structured collection
- ``document-embed-upsert`` — one vector record per document; embeds an LLM-generated
                              summary (when ``llm`` is configured on the step) or the
                              raw document text

Steps run in declaration order.  For config-driven construction see ``api/factory.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

from cogbase.core.models import Chunk, Document
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import ChatMessage, LLMBase
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.stores import CollectionSchema, StructuredStoreBase, VectorCollectionSchema, VectorStoreBase

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
    """A vector collection backed by a store and embedder.

    Args:
        schema:   ``VectorCollectionSchema`` carrying the collection name,
                  dimensions, description, metadata_fields, and optional metadata.
        store:    ``VectorStoreBase`` implementation that persists chunks.
        embedder: ``EmbeddingBase`` implementation that produces dense vectors.
    """

    schema: VectorCollectionSchema
    store: VectorStoreBase
    embedder: EmbeddingBase

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description


@dataclass
class StructuredCollection:
    """A structured collection backed by a store and schema.

    The collection name is taken from ``schema.name`` — no separate ``name``
    field is needed.

    Args:
        schema: ``CollectionSchema`` describing the table and its fields.
        store:  ``StructuredStoreBase`` implementation that persists records.
    """

    schema: CollectionSchema
    store: StructuredStoreBase

    @property
    def name(self) -> str:
        """Collection name, taken from the schema."""
        return self.schema.name


@dataclass
class PipelineStep:
    """One step in the ingestion pipeline.

    Args:
        tool:       One of ``"chunk-embed-upsert"``, ``"extract-structured"``,
                    or ``"document-embed-upsert"``.
        collection: Name of the target collection for this step.
        chunker:    Chunker for ``chunk-embed-upsert`` steps.
        extractor:  Extractor for ``extract-structured`` steps.
        llm:        Optional LLM for ``document-embed-upsert`` steps.  When
                    ``None`` the raw document text is embedded directly.
        doc_prompt: System prompt for the document summarization call.
    """

    tool: str
    collection: str
    chunker: ChunkerBase | None = None
    extractor: ExtractorBase | None = None
    llm: LLMBase | None = None
    doc_prompt: str = "Summarize this document in a few sentences."


class IngestionPipeline:
    """Ordered ingestion pipeline supporting multiple collections.

    Each step maps a tool name to a named collection:

    - ``"chunk-embed-upsert"``    → :class:`VectorCollection` (requires ``step.chunker``)
    - ``"extract-structured"``    → :class:`StructuredCollection`
    - ``"document-embed-upsert"`` → :class:`VectorCollection` (optional ``step.llm``)

    Args:
        name:                   Logical name for this pipeline.
        steps:                  Ordered list of :class:`PipelineStep` objects.
        vector_collections:     Vector collections available to steps.
        structured_collections: Structured collections available to steps.
        match:                  Metadata filter — this pipeline only processes
                                documents whose metadata contains all specified
                                key/value pairs.  ``None`` matches all documents.
        parallel:               When ``True``, all steps run concurrently via
                                ``asyncio.gather`` instead of sequentially.
    """

    match: dict[str, str] | None = None

    def __init__(
        self,
        name: str,
        steps: list[PipelineStep] | None = None,
        vector_collections: list[VectorCollection] | None = None,
        structured_collections: list[StructuredCollection] | None = None,
        match: dict[str, str] | None = None,
        parallel: bool = False,
    ) -> None:
        self.name = name
        self.match = match
        self.parallel = parallel

        _vcs: list[VectorCollection] = list(vector_collections or [])
        _scs: list[StructuredCollection] = list(structured_collections or [])

        self._vector_by_name: dict[str, VectorCollection] = {vc.name: vc for vc in _vcs}
        self._structured_by_name: dict[str, StructuredCollection] = {sc.name: sc for sc in _scs}

        self._steps = list(steps or [])

    async def _run_step(self, doc: Document, step: PipelineStep) -> int:
        """Dispatch one step and return the number of records extracted (0 for non-structured steps)."""
        if step.tool == "chunk-embed-upsert":
            return await self._run_chunk_embed_upsert(doc, step)
        if step.tool == "extract-structured":
            return await self._run_extract_structured(doc, step)
        if step.tool == "document-embed-upsert":
            await self._run_document_embed_upsert(doc, step)
            return 0
        logger.warning(
            "ingestion_pipeline.ingest.unknown_tool name=%s tool=%s", self.name, step.tool
        )
        return 0

    async def _ingest(self, doc: Document) -> int:
        """Ingest a document by executing each step, sequentially or in parallel.

        Returns:
            Number of structured records saved (sum across all structured steps).
        """
        logger.info("ingestion_pipeline.ingest.start name=%s doc_id=%s", self.name, doc.doc_id)

        if self.parallel:
            counts = await asyncio.gather(*[self._run_step(doc, step) for step in self._steps])
            records_extracted = sum(counts)
        else:
            records_extracted = 0
            for step in self._steps:
                records_extracted += await self._run_step(doc, step)

        logger.info(
            "ingestion_pipeline.ingest.done name=%s doc_id=%s records_extracted=%d",
            self.name, doc.doc_id, records_extracted,
        )
        return records_extracted

    async def _run_chunk_embed_upsert(self, doc: Document, step: PipelineStep) -> int:
        vc = self._vector_by_name.get(step.collection)
        if vc is None:
            logger.warning(
                "ingestion_pipeline.chunk_embed_upsert.unknown_collection name=%s collection=%s",
                self.name, step.collection,
            )
            return 0
        if step.chunker is None:
            logger.warning(
                "ingestion_pipeline.chunk_embed_upsert.no_chunker name=%s collection=%s",
                self.name, step.collection,
            )
            return 0

        chunks = step.chunker.chunk(doc)
        logger.info(
            "ingestion_pipeline.chunk_embed_upsert.chunked name=%s doc_id=%s collection=%s chunks=%d",
            self.name, doc.doc_id, step.collection, len(chunks),
        )
        if not chunks:
            return 0

        embeddings = await vc.embedder.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks."
            )
        doc_meta = {k: v for k, v in doc.metadata.items() if k in vc.schema.metadata_fields}
        embedded = [
            chunk.model_copy(update={"embedding": emb, "metadata": {**chunk.metadata, **doc_meta}})
            for chunk, emb in zip(chunks, embeddings)
        ]
        await vc.store.upsert(vc.name, embedded)
        logger.info(
            "ingestion_pipeline.chunk_embed_upsert.upserted name=%s doc_id=%s collection=%s count=%d",
            self.name, doc.doc_id, step.collection, len(embedded),
        )
        return 0

    async def _run_extract_structured(self, doc: Document, step: PipelineStep) -> int:
        sc = self._structured_by_name.get(step.collection)
        if sc is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.unknown_collection name=%s collection=%s",
                self.name, step.collection,
            )
            return 0
        if step.extractor is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.no_extractor name=%s collection=%s",
                self.name, step.collection,
            )
            return 0

        records = await step.extractor.extract(doc)
        if not records:
            return 0

        await sc.store.save(sc.schema.name, records)
        logger.info(
            "ingestion_pipeline.extract_structured.saved name=%s doc_id=%s collection=%s count=%d",
            self.name, doc.doc_id, step.collection, len(records),
        )
        return len(records)

    async def _run_document_embed_upsert(self, doc: Document, step: PipelineStep) -> None:
        vc = self._vector_by_name.get(step.collection)
        if vc is None:
            logger.warning(
                "ingestion_pipeline.document_embed_upsert.unknown_collection name=%s collection=%s",
                self.name, step.collection,
            )
            return

        text = await self._get_document_text(doc, step)
        if not text:
            logger.info(
                "ingestion_pipeline.document_embed_upsert.empty_text name=%s doc_id=%s",
                self.name, doc.doc_id,
            )
            return

        (embedding,) = await vc.embedder.embed([text])
        metadata = {k: v for k, v in doc.metadata.items() if k in vc.schema.metadata_fields}
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}__document",
            doc_id=doc.doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        await vc.store.upsert(vc.name, [chunk])
        logger.info(
            "ingestion_pipeline.document_embed_upsert.upserted name=%s doc_id=%s collection=%s",
            self.name, doc.doc_id, step.collection,
        )

    async def _get_document_text(self, doc: Document, step: PipelineStep) -> str | None:
        if step.llm is None:
            return doc.text or None
        messages: list[ChatMessage] = [
            {"role": "system", "content": step.doc_prompt},
            {"role": "user", "content": doc.text},
        ]
        try:
            result = await step.llm.complete(messages)
            return result.get("content") or None
        except Exception:
            logger.exception(
                "ingestion_pipeline.get_document_text.failed name=%s doc_id=%s collection=%s",
                self.name, doc.doc_id, step.collection,
            )
            return None

    async def ingest_documents(
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

        async def _ingest_one(doc: Document) -> IngestResult:
            try:
                records_extracted = await self._ingest(doc)
                return IngestResult(doc_id=doc.doc_id, success=True, records_extracted=records_extracted)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "ingestion_pipeline.ingest_documents.failed name=%s doc_id=%s",
                    self.name,
                    doc.doc_id,
                )
                return IngestResult(doc_id=doc.doc_id, success=False, error=exc)

        if len(documents) == 1:
            return [await _ingest_one(documents[0])]

        semaphore = asyncio.Semaphore(concurrency)

        async def _ingest_one_gated(doc: Document) -> IngestResult:
            async with semaphore:
                return await _ingest_one(doc)

        return list(await asyncio.gather(*(_ingest_one_gated(d) for d in documents)))
