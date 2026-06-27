"""IngestionPipeline — ordered steps over multiple vector and structured collections.

Supports three step types:

- ``chunk-embed-upsert``    — chunk document text, embed, upsert to a vector collection
- ``extract-structured``    — LLM extraction → save to a structured collection
- ``document-embed-upsert`` — one vector record per document; embeds an LLM-generated
                              summary (when ``llm`` is configured on the step) or the
                              raw document text. Summaries of documents that overflow the
                              llm context window are produced map-reduce; with no ``llm``,
                              a document that overflows the embedding context window raises
                              (it cannot be reduced to a single vector without one)

Steps run in declaration order.  For config-driven construction see ``api/factory.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

from cogbase.core.models import Chunk, Document
from cogbase.embeddings import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.llms.summarization import estimate_tokens, summarise_chunk_tokens, summarize_text
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.chunking.base import ChunkerBase
from cogbase.stores import CollectionSchema, StructuredStoreBase, VectorCollectionSchema, VectorStoreBase
from cogbase.stores.filters import Col

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
        chunks_written:    Total number of vector chunks upserted across all
                           ``chunk-embed-upsert`` and ``document-embed-upsert``
                           steps (0 when no vector collections are configured).
        extraction_failed: ``True`` when at least one ``extract-structured`` step
                           returned ``None`` after all retries (parse failure or
                           blank document).  The document may still be partially
                           ingested (e.g. vector chunks were written); this flag
                           distinguishes that from a hard ingest failure.
        error:             The exception raised, when *success* is ``False``.
    """

    doc_id: str
    success: bool
    records_extracted: int = 0
    chunks_written: int = 0
    extraction_failed: bool = False
    error: Exception | None = field(default=None, repr=False)

    @property
    def ingested_nothing(self) -> bool:
        """``True`` when the document succeeded but wrote no chunks and no records.

        A successful ingest that lands nothing in any store usually means the
        document carried no extractable text (e.g. a scanned/image-only PDF) or
        the pipeline does not match its content — worth surfacing rather than
        reporting as a silent success.
        """
        return self.success and self.chunks_written == 0 and self.records_extracted == 0


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
        app_id:                 Stable internal id of the owning application.
                                Threaded into every log line so ingestion
                                activity can be attributed to an app; ``""`` when
                                the pipeline is used standalone (e.g. in tests).
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
        description: str = "",
        app_id: str = "",
    ) -> None:
        self.name = name
        self.description = description or name
        self.match = match
        self.parallel = parallel
        self.app_id = app_id

        _vcs: list[VectorCollection] = list(vector_collections or [])
        _scs: list[StructuredCollection] = list(structured_collections or [])

        self._vector_by_name: dict[str, VectorCollection] = {vc.name: vc for vc in _vcs}
        self._structured_by_name: dict[str, StructuredCollection] = {sc.name: sc for sc in _scs}

        self._steps = list(steps or [])

    async def _run_step(self, doc: Document, step: PipelineStep) -> tuple[int, int, bool]:
        """Dispatch one step.

        Returns:
            ``(records_extracted, chunks_written, extraction_failed)`` —
            ``extraction_failed`` is ``True`` only when an ``extract-structured``
            step's extractor returned ``None`` after all retries.
        """
        if step.tool == "chunk-embed-upsert":
            return 0, await self._run_chunk_embed_upsert(doc, step), False
        if step.tool == "extract-structured":
            records, failed = await self._run_extract_structured(doc, step)
            return records, 0, failed
        if step.tool == "document-embed-upsert":
            return 0, await self._run_document_embed_upsert(doc, step), False
        logger.warning(
            "ingestion_pipeline.ingest.unknown_tool app_id=%s name=%s tool=%s",
            self.app_id, self.name, step.tool,
        )
        return 0, 0, False

    async def purge_document(self, doc_id: str) -> None:
        """Remove ``doc_id``'s prior data from every collection this pipeline writes.

        Makes re-ingestion idempotent. Vector collections are purged by
        ``delete_doc`` — this drops chunks a shorter re-extraction would otherwise
        orphan (chunk ids are positional, so a re-ingest yielding fewer chunks
        leaves the trailing old ones behind without this). Structured collections
        are purged by deleting every record carrying this ``doc_id``, so a
        re-ingest replaces the document's rows instead of appending duplicates
        (``save`` upserts only on the primary key, which for list-mode extraction
        is a positional item id that shifts when extraction output changes).

        Deleting a ``doc_id`` absent from a collection is a no-op, so this is safe
        on first ingest. Each collection is purged once even when several steps
        target it.
        """
        vector_names: set[str] = set()
        structured_names: set[str] = set()
        for step in self._steps:
            if step.tool in ("chunk-embed-upsert", "document-embed-upsert"):
                vector_names.add(step.collection)
            elif step.tool == "extract-structured":
                structured_names.add(step.collection)

        for name in vector_names:
            vc = self._vector_by_name.get(name)
            if vc is not None:
                await vc.store.delete_doc(vc.name, doc_id)
        for name in structured_names:
            sc = self._structured_by_name.get(name)
            if sc is not None:
                await sc.store.delete_records(sc.schema.name, [Col("doc_id") == doc_id])

    async def _ingest(self, doc: Document) -> tuple[int, int, bool]:
        """Ingest a document by executing each step, sequentially or in parallel.

        Returns:
            ``(records_extracted, chunks_written, extraction_failed)`` —
            ``extraction_failed`` is ``True`` if any ``extract-structured`` step
            failed after all retries.
        """
        logger.info(
            "ingestion_pipeline.ingest.start app_id=%s name=%s doc_id=%s steps=%d parallel=%s",
            self.app_id, self.name, doc.doc_id, len(self._steps), self.parallel,
        )

        # Purge any prior ingest of this doc_id before (re)writing, so a re-ingest
        # replaces the document's data rather than duplicating structured rows or
        # leaving orphaned vector chunks. Must complete before any step runs —
        # including the parallel path, where steps would otherwise race the delete.
        await self.purge_document(doc.doc_id)

        if self.parallel:
            results = await asyncio.gather(*[self._run_step(doc, step) for step in self._steps])
            records_extracted = sum(r[0] for r in results)
            chunks_written = sum(r[1] for r in results)
            extraction_failed = any(r[2] for r in results)
        else:
            records_extracted = 0
            chunks_written = 0
            extraction_failed = False
            for step in self._steps:
                records, chunks, failed = await self._run_step(doc, step)
                records_extracted += records
                chunks_written += chunks
                extraction_failed = extraction_failed or failed

        logger.info(
            "ingestion_pipeline.ingest.done app_id=%s name=%s doc_id=%s records_extracted=%d chunks_written=%d extraction_failed=%s",
            self.app_id, self.name, doc.doc_id, records_extracted, chunks_written, extraction_failed,
        )
        return records_extracted, chunks_written, extraction_failed

    async def _run_chunk_embed_upsert(self, doc: Document, step: PipelineStep) -> int:
        vc = self._vector_by_name.get(step.collection)
        if vc is None:
            logger.warning(
                "ingestion_pipeline.chunk_embed_upsert.unknown_collection app_id=%s name=%s collection=%s",
                self.app_id, self.name, step.collection,
            )
            return 0
        if step.chunker is None:
            logger.warning(
                "ingestion_pipeline.chunk_embed_upsert.no_chunker app_id=%s name=%s collection=%s",
                self.app_id, self.name, step.collection,
            )
            return 0

        chunks = step.chunker.chunk(doc)
        logger.info(
            "ingestion_pipeline.chunk_embed_upsert.chunked app_id=%s name=%s doc_id=%s collection=%s chunks=%d",
            self.app_id, self.name, doc.doc_id, step.collection, len(chunks),
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
            "ingestion_pipeline.chunk_embed_upsert.upserted app_id=%s name=%s doc_id=%s collection=%s count=%d",
            self.app_id, self.name, doc.doc_id, step.collection, len(embedded),
        )
        return len(embedded)

    async def _run_extract_structured(self, doc: Document, step: PipelineStep) -> tuple[int, bool]:
        sc = self._structured_by_name.get(step.collection)
        if sc is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.unknown_collection app_id=%s name=%s collection=%s",
                self.app_id, self.name, step.collection,
            )
            return 0, False
        if step.extractor is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.no_extractor app_id=%s name=%s collection=%s",
                self.app_id, self.name, step.collection,
            )
            return 0, False

        records = await step.extractor.extract(doc)
        if records is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.failed app_id=%s name=%s doc_id=%s collection=%s",
                self.app_id, self.name, doc.doc_id, step.collection,
            )
            return 0, True
        if not records:
            logger.debug(
                "ingestion_pipeline.extract_structured.no_records app_id=%s name=%s doc_id=%s collection=%s",
                self.app_id, self.name, doc.doc_id, step.collection,
            )
            return 0, False

        await sc.store.save(sc.schema.name, records)
        logger.info(
            "ingestion_pipeline.extract_structured.saved app_id=%s name=%s doc_id=%s collection=%s count=%d",
            self.app_id, self.name, doc.doc_id, step.collection, len(records),
        )
        return len(records), False

    async def _run_document_embed_upsert(self, doc: Document, step: PipelineStep) -> int:
        vc = self._vector_by_name.get(step.collection)
        if vc is None:
            logger.warning(
                "ingestion_pipeline.document_embed_upsert.unknown_collection app_id=%s name=%s collection=%s",
                self.app_id, self.name, step.collection,
            )
            return 0

        text = await self._get_document_text(doc, vc, step)
        if not text:
            logger.info(
                "ingestion_pipeline.document_embed_upsert.empty_text app_id=%s name=%s doc_id=%s",
                self.app_id, self.name, doc.doc_id,
            )
            return 0

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
            "ingestion_pipeline.document_embed_upsert.upserted app_id=%s name=%s doc_id=%s collection=%s",
            self.app_id, self.name, doc.doc_id, step.collection,
        )
        return 1

    async def _get_document_text(
        self, doc: Document, vc: VectorCollection, step: PipelineStep
    ) -> str | None:
        """Produce the text to embed as this document's single vector record.

        With no ``step.llm`` the raw document text is embedded directly, so it
        must fit the embedding context window — otherwise the document-level
        vector would be silently truncated by the backend. That is a step
        misconfiguration (a document-level embed with no summarizer over an
        oversized document), so it raises rather than corrupting the vector.

        With ``step.llm`` configured the text is summarized through
        ``step.doc_prompt`` regardless of length, since the summary is the point
        of the step. Documents that overflow the model's context window are
        summarized map-reduce (see :meth:`_summarize`).
        """
        text = doc.text or None
        if text is None:
            return None

        if step.llm is None:
            if estimate_tokens(text) > vc.embedder.context_window:
                raise ValueError(
                    f"document-embed-upsert: document {doc.doc_id!r} is too large to "
                    f"embed as a single vector — it exceeds the embedding context "
                    f"window of {vc.embedder.context_window} tokens and the step has "
                    f"no llm to summarize it. Configure an llm/doc_prompt on this step."
                )
            return text

        return await self._summarize(doc, step)

    async def _summarize(self, doc: Document, step: PipelineStep) -> str | None:
        """Summarize ``doc.text`` through ``step.doc_prompt``; ``None`` on failure.

        Delegates to :func:`cogbase.llms.summarization.summarize_text`, the
        shared map-reduce summariser: a document that fits the summariser's
        context window is summarized in one call, while a larger one is split,
        summarized per chunk, and recursively merged into a single bounded
        summary. Runs on the cheaper ``"mini"`` model. Transient LLM failures are
        a step-local concern — they are logged and degraded to ``None`` (the step
        skips the upsert) rather than aborting the document's other steps.
        """
        try:
            summary = await summarize_text(
                step.llm,
                doc.text,
                chunk_tokens=summarise_chunk_tokens(step.llm, "mini"),
                compress_prompt=step.doc_prompt,
                model="mini",
            )
            return summary or None
        except Exception:
            logger.exception(
                "ingestion_pipeline.summarize.failed app_id=%s name=%s doc_id=%s collection=%s",
                self.app_id, self.name, doc.doc_id, step.collection,
            )
            return None

    async def ingest_documents(
        self,
        documents: Sequence[Document],
    ) -> list[IngestResult]:
        """Ingest a sequence of documents concurrently.

        Each document is processed independently.  A failure on one document does
        not abort the others — the error is captured in the corresponding
        ``IngestResult`` and ingestion continues for the remaining documents.
        Results are returned in the same order as *documents*.
        """
        async def _ingest_one(doc: Document) -> IngestResult:
            try:
                records_extracted, chunks_written, extraction_failed = await self._ingest(doc)
                return IngestResult(
                    doc_id=doc.doc_id,
                    success=True,
                    records_extracted=records_extracted,
                    chunks_written=chunks_written,
                    extraction_failed=extraction_failed,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "ingestion_pipeline.ingest_documents.failed app_id=%s name=%s doc_id=%s",
                    self.app_id,
                    self.name,
                    doc.doc_id,
                )
                return IngestResult(doc_id=doc.doc_id, success=False, error=exc)

        if not documents:
            return []
        logger.info(
            "ingestion_pipeline.ingest_documents.start app_id=%s name=%s documents=%d",
            self.app_id, self.name, len(documents),
        )
        if len(documents) == 1:
            results = [await _ingest_one(documents[0])]
        else:
            results = list(await asyncio.gather(*(_ingest_one(d) for d in documents)))
        failures = sum(1 for r in results if not r.success)
        logger.info(
            "ingestion_pipeline.ingest_documents.done app_id=%s name=%s documents=%d failures=%d",
            self.app_id, self.name, len(results), failures,
        )
        return results
