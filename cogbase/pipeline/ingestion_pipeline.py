"""IngestionPipeline — ordered steps over multiple vector and structured collections.

Supports three step types:

- ``chunk-embed-upsert``    — chunk document text, embed, upsert to a vector collection
- ``extract-structured``    — LLM extraction → save to a structured collection
- ``document-embed-upsert`` — one vector record per document; embeds an LLM-generated
                              summary (when ``llm`` is configured) or the raw document
                              text; carries optional metadata fields for search filtering

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


_DEFAULT_VC_DESCRIPTIONS: dict[str, str] = {
    "document_chunks": "Full-text passage chunks; use for detailed or specific questions about document content.",
    "document_summary": "One-per-document summaries; use for topic-level or high-level questions about what documents cover.",
}


@dataclass
class ChunkCollection:
    """A vector collection backed by a store, embedder, and chunker.

    Args:
        schema:   ``VectorCollectionSchema`` carrying the collection name,
                  dimensions, description, and optional metadata.
        store:    ``VectorStoreBase`` implementation that persists chunks.
        embedder: ``EmbeddingBase`` implementation that produces dense vectors.
        chunker:  ``ChunkerBase`` implementation that splits document text.
    """

    schema: VectorCollectionSchema
    store: VectorStoreBase
    embedder: EmbeddingBase
    chunker: ChunkerBase

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description


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


@dataclass
class DocumentCollection:
    """A vector collection with exactly one record per document.

    Each ingested document produces one chunk whose text is either an
    LLM-generated summary (when ``llm`` is supplied) or the raw document text
    (when ``llm`` is ``None``).  Useful for document-level semantic search and
    cross-document similarity queries.

    ``metadata_fields`` lists keys to copy from ``Document.metadata`` into the
    stored ``Chunk.metadata``, making them available as filter predicates at
    search time (e.g. ``customer_id``, ``deal_stage``).  Only keys present in
    the document metadata are copied; missing keys are silently skipped.

    Args:
        schema:         ``VectorCollectionSchema`` carrying the collection name,
                        dimensions, description, and optional metadata.
        store:          ``VectorStoreBase`` that persists the chunks.
        embedder:       ``EmbeddingBase`` that produces the embedding.
        llm:            Optional ``LLMBase`` used to generate the summary text.
                        When ``None`` the raw document text is embedded directly.
        prompt:         System prompt for the LLM summarisation call.
        max_tokens:     Maximum tokens for the generated summary.
        metadata_fields: Document metadata keys to project into chunk metadata.
    """

    schema: VectorCollectionSchema
    store: VectorStoreBase
    embedder: EmbeddingBase
    llm: LLMBase | None = None
    prompt: str = "Summarize this document in a few sentences."
    max_tokens: int = 1024
    metadata_fields: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description


class IngestionPipeline:
    """Ordered ingestion pipeline supporting multiple collections.

    Each step maps a tool name to a named collection:

    - ``"chunk-embed-upsert"``    → :class:`ChunkCollection`
    - ``"extract-structured"``    → :class:`StructuredCollection`
    - ``"document-embed-upsert"`` → :class:`DocumentCollection`

    Steps run in declaration order.

    Args:
        name:                   Logical name for this pipeline.
        steps:                  Ordered list of ``(tool, collection_name)`` tuples.
                                Auto-generated from provided collections when omitted.
        chunk_collections:      Chunk collections available to steps.
        structured_collections: Structured collections available to steps.
        document_collections:   Document-level vector collections available to steps.
    """

    def __init__(
        self,
        name: str,
        steps: list[tuple[str, str]] | None = None,
        chunk_collections: list[ChunkCollection] | None = None,
        structured_collections: list[StructuredCollection] | None = None,
        document_collections: list[DocumentCollection] | None = None,
    ) -> None:
        self.name = name

        _vcs: list[ChunkCollection] = list(chunk_collections or [])
        _scs: list[StructuredCollection] = list(structured_collections or [])
        _dcs: list[DocumentCollection] = list(document_collections or [])

        self._chunk_by_name: dict[str, ChunkCollection] = {vc.name: vc for vc in _vcs}
        self._structured_by_name: dict[str, StructuredCollection] = {sc.name: sc for sc in _scs}
        self._document_by_name: dict[str, DocumentCollection] = {dc.name: dc for dc in _dcs}

        # Auto-generate steps from collection order when not explicitly provided
        if steps is None:
            _steps: list[tuple[str, str]] = []
            for vc in _vcs:
                _steps.append(("chunk-embed-upsert", vc.name))
            for sc in _scs:
                _steps.append(("extract-structured", sc.name))
            for dc in _dcs:
                _steps.append(("document-embed-upsert", dc.name))
            self._steps = _steps
        else:
            self._steps = list(steps)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def structured_schemas(self) -> list[CollectionSchema]:
        """Schemas for all structured collections.

        Pass to ``Runner(structured_schemas=...)`` so the LLM knows available
        collection names and field types.
        """
        return [sc.schema for sc in self._structured_by_name.values()]

    @property
    def vector_collection_names(self) -> list[str]:
        """Names of all vector collections (chunk-embed and document-embed), in step order."""
        seen: list[str] = []
        for tool, name in self._steps:
            if tool in ("chunk-embed-upsert", "document-embed-upsert") and name not in seen:
                seen.append(name)
        return seen

    @property
    def vector_collection_infos(self) -> list[tuple[str, str]]:
        """``(name, description)`` pairs for all vector collections, in step order.

        The description is taken from the collection's ``description`` field, falling
        back to ``_DEFAULT_VC_DESCRIPTIONS`` for well-known names, then the bare name.
        Pass to ``Runner(vector_collections=...)`` so the LLM can pick the right one.
        """
        seen: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for tool, name in self._steps:
            if tool not in ("chunk-embed-upsert", "document-embed-upsert"):
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            col = self._chunk_by_name.get(name) or self._document_by_name.get(name)
            explicit = col.description if col is not None else ""
            desc = explicit or _DEFAULT_VC_DESCRIPTIONS.get(name, name)
            seen.append((name, desc))
        return seen

    def runner_resources(
        self,
    ) -> tuple[StructuredStoreBase | None, VectorStoreBase | None, EmbeddingBase | None, str | None]:
        """Return ``(structured_store, vector_store, embedder, default_vector_collection)`` for Runner.

        Selects the first chunk-embed-upsert collection's store/embedder as the
        default vector resources (falls back to document-embed-upsert if none).
        """
        vector_store: VectorStoreBase | None = None
        embedder: EmbeddingBase | None = None
        default_vc: str | None = None

        for tool, name in self._steps:
            if tool == "chunk-embed-upsert" and name in self._chunk_by_name:
                vc = self._chunk_by_name[name]
                vector_store, embedder, default_vc = vc.store, vc.embedder, vc.name
                break

        if vector_store is None:
            for tool, name in self._steps:
                if tool == "document-embed-upsert" and name in self._document_by_name:
                    dc = self._document_by_name[name]
                    vector_store, embedder, default_vc = dc.store, dc.embedder, dc.name
                    break

        structured_store: StructuredStoreBase | None = None
        for sc in self._structured_by_name.values():
            structured_store = sc.store
            break

        return structured_store, vector_store, embedder, default_vc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create all structured collections in their stores. Idempotent."""
        logger.info("ingestion_pipeline.setup.start name=%s", self.name)
        for sc in self._structured_by_name.values():
            logger.debug(
                "ingestion_pipeline.setup.create_collection name=%s collection=%s",
                self.name, sc.name,
            )
            await sc.store.create_collection(sc.schema)
        logger.info("ingestion_pipeline.setup.done name=%s", self.name)

    async def _ingest(self, doc: Document) -> int:
        """Ingest a document by executing each step in declaration order.

        Returns:
            Number of structured records saved (sum across all structured steps).
        """
        logger.info("ingestion_pipeline.ingest.start name=%s doc_id=%s", self.name, doc.doc_id)
        records_extracted = 0

        for tool, collection_name in self._steps:
            if tool == "chunk-embed-upsert":
                records_extracted += await self._run_chunk_embed_upsert(doc, collection_name)
            elif tool == "extract-structured":
                records_extracted += await self._run_extract_structured(doc, collection_name)
            elif tool == "document-embed-upsert":
                await self._run_document_embed_upsert(doc, collection_name)
            else:
                logger.warning(
                    "ingestion_pipeline.ingest.unknown_tool name=%s tool=%s", self.name, tool
                )

        logger.info(
            "ingestion_pipeline.ingest.done name=%s doc_id=%s records_extracted=%d",
            self.name, doc.doc_id, records_extracted,
        )
        return records_extracted

    async def _run_chunk_embed_upsert(self, doc: Document, collection_name: str) -> int:
        vc = self._chunk_by_name.get(collection_name)
        if vc is None:
            logger.warning(
                "ingestion_pipeline.chunk_embed_upsert.unknown_collection name=%s collection=%s",
                self.name, collection_name,
            )
            return 0

        chunks = vc.chunker.chunk(doc)
        logger.debug(
            "ingestion_pipeline.chunk_embed_upsert.chunked name=%s doc_id=%s collection=%s chunks=%d",
            self.name, doc.doc_id, collection_name, len(chunks),
        )
        if not chunks:
            return 0

        embeddings = await vc.embedder.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks."
            )
        embedded = [
            chunk.model_copy(update={"embedding": emb})
            for chunk, emb in zip(chunks, embeddings)
        ]
        await vc.store.upsert(vc.name, embedded)
        logger.debug(
            "ingestion_pipeline.chunk_embed_upsert.upserted name=%s doc_id=%s collection=%s count=%d",
            self.name, doc.doc_id, collection_name, len(embedded),
        )
        return 0

    async def _run_extract_structured(self, doc: Document, collection_name: str) -> int:
        sc = self._structured_by_name.get(collection_name)
        if sc is None:
            logger.warning(
                "ingestion_pipeline.extract_structured.unknown_collection name=%s collection=%s",
                self.name, collection_name,
            )
            return 0

        record = await sc.extractor.extract(doc)
        if record is None:
            return 0

        await sc.store.save(sc.schema.name, [record])
        logger.debug(
            "ingestion_pipeline.extract_structured.saved name=%s doc_id=%s collection=%s",
            self.name, doc.doc_id, collection_name,
        )
        return 1

    async def _run_document_embed_upsert(self, doc: Document, collection_name: str) -> None:
        dc = self._document_by_name.get(collection_name)
        if dc is None:
            logger.warning(
                "ingestion_pipeline.document_embed_upsert.unknown_collection name=%s collection=%s",
                self.name, collection_name,
            )
            return

        text = await self._get_document_text(doc, dc)
        if not text:
            logger.debug(
                "ingestion_pipeline.document_embed_upsert.empty_text name=%s doc_id=%s",
                self.name, doc.doc_id,
            )
            return

        (embedding,) = await dc.embedder.embed([text])
        metadata = {k: v for k, v in doc.metadata.items() if k in dc.metadata_fields}
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}__document",
            doc_id=doc.doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        await dc.store.upsert(dc.name, [chunk])
        logger.debug(
            "ingestion_pipeline.document_embed_upsert.upserted name=%s doc_id=%s collection=%s",
            self.name, doc.doc_id, collection_name,
        )

    async def _get_document_text(self, doc: Document, dc: DocumentCollection) -> str | None:
        if dc.llm is None:
            return doc.text or None
        messages: list[ChatMessage] = [
            {"role": "system", "content": dc.prompt},
            {"role": "user", "content": doc.text},
        ]
        try:
            result = await dc.llm.complete(messages, max_tokens=dc.max_tokens)
            return result.get("content") or None
        except Exception:
            logger.exception(
                "ingestion_pipeline.get_document_text.failed name=%s doc_id=%s collection=%s",
                self.name, doc.doc_id, dc.name,
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
