"""IngestionPipeline — ordered steps over multiple vector and structured collections.

An ``IngestionPipeline`` is the primary entry point for configuring CogBase
ingestion.  It supports three step types:

- ``chunk-embed-upsert``   — chunk document text, embed, upsert to a vector collection
- ``extract-structured``   — LLM extraction → save to a structured collection
- ``summarize-embed-upsert`` — LLM summary of the full document → embed → upsert to
                               a vector collection (one chunk per document)

Steps run in declaration order.  Multiple vector collections and structured
collections may be used in the same pipeline.

Typical usage (multi-collection)::

    from cogbase.pipeline.ingestion_pipeline import (
        IngestionPipeline, VectorCollection, StructuredCollection, SummarizeCollection,
    )

    pipeline = IngestionPipeline(
        name="legal",
        steps=[
            ("chunk-embed-upsert",     "document_chunks"),
            ("extract-structured",     "contract_extraction"),
            ("summarize-embed-upsert", "document_summary"),
        ],
        vector_collections=[
            VectorCollection(
                name="document_chunks",
                store=FAISSVectorStore(dim=1536),
                embedder=OpenAIEmbedding(...),
                chunker=FixedSizeChunker(chunk_size=512, overlap=64),
            ),
        ],
        structured_collections=[
            StructuredCollection(
                schema=contracts_schema,
                store=SQLiteStructuredStore("data.db"),
                extractor=ContractExtractor(),
            ),
        ],
        summarize_collections=[
            SummarizeCollection(
                name="document_summary",
                store=FAISSVectorStore(dim=1536),  # typically the same store instance
                embedder=OpenAIEmbedding(...),
                llm=llm,
            ),
        ],
    )

Backward-compatible single-collection usage (existing callers unaffected)::

    pipeline = IngestionPipeline(
        name="legal",
        vector_collection=VectorCollection(...),
        structured_collection=StructuredCollection(...),
    )
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


_DEFAULT_VC_DESCRIPTIONS: dict[str, str] = {
    "document_chunks": "Full-text passage chunks; use for detailed or specific questions about document content.",
    "document_summary": "One-per-document summaries; use for topic-level or high-level questions about what documents cover.",
}


@dataclass
class VectorCollection:
    """A named vector collection backed by a store, embedder, and chunker.

    Args:
        name:        Logical name for this collection (used for lookup and logging).
        store:       ``VectorStoreBase`` implementation that persists chunks.
        embedder:    ``EmbeddingBase`` implementation that produces dense vectors.
        chunker:     ``ChunkerBase`` implementation that splits document text.
        description: Short description shown to the LLM to help it choose the right
                     collection.  Falls back to ``_DEFAULT_VC_DESCRIPTIONS[name]``
                     when empty, or the bare name if no default exists.
    """

    name: str
    store: VectorStoreBase
    embedder: EmbeddingBase
    chunker: ChunkerBase
    description: str = ""


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
class SummarizeCollection:
    """A vector collection populated with LLM-generated per-document summaries.

    Each ingested document produces exactly one chunk: the LLM summary of its
    full text.  This is useful for high-level semantic search over document
    topics rather than specific passages.

    Args:
        name:        Logical name for this collection.
        store:       ``VectorStoreBase`` that persists the summary chunks.
        embedder:    ``EmbeddingBase`` that produces the summary embedding.
        llm:         ``LLMBase`` used to generate the summary.
        prompt:      System prompt for the summarisation call.
        max_tokens:  Maximum tokens for the generated summary.
        description: Short description shown to the LLM to help it choose the right
                     collection.  Falls back to ``_DEFAULT_VC_DESCRIPTIONS[name]``
                     when empty, or the bare name if no default exists.
    """

    name: str
    store: VectorStoreBase
    embedder: EmbeddingBase
    llm: LLMBase
    prompt: str = "Summarize this document in a few sentences."
    max_tokens: int = 1024
    description: str = ""


class IngestionPipeline:
    """Ordered ingestion pipeline supporting multiple collections.

    Each step maps a tool name to a named collection:

    - ``"chunk-embed-upsert"``     → :class:`VectorCollection`
    - ``"extract-structured"``     → :class:`StructuredCollection`
    - ``"summarize-embed-upsert"`` → :class:`SummarizeCollection`

    Steps run in declaration order.

    Args:
        name:                  Logical name for this pipeline.
        steps:                 Ordered list of ``(tool, collection_name)`` tuples.
                               Auto-generated from provided collections when omitted.
        vector_collections:    Vector collections available to steps.
        structured_collections: Structured collections available to steps.
        summarize_collections: Summarize collections available to steps.
        vector_collection:     Backward-compat alias for a single VectorCollection.
        structured_collection: Backward-compat alias for a single StructuredCollection.
    """

    def __init__(
        self,
        name: str,
        steps: list[tuple[str, str]] | None = None,
        vector_collections: list[VectorCollection] | None = None,
        structured_collections: list[StructuredCollection] | None = None,
        summarize_collections: list[SummarizeCollection] | None = None,
        # Backward-compat single-item kwargs:
        vector_collection: VectorCollection | None = None,
        structured_collection: StructuredCollection | None = None,
    ) -> None:
        self.name = name

        # Merge backward-compat single items (put them first to preserve step order)
        _vcs: list[VectorCollection] = []
        _scs: list[StructuredCollection] = []
        _smcs: list[SummarizeCollection] = list(summarize_collections or [])

        if vector_collection is not None:
            _vcs.append(vector_collection)
        _vcs.extend(vc for vc in (vector_collections or []) if vc not in _vcs)

        if structured_collection is not None:
            _scs.append(structured_collection)
        _scs.extend(sc for sc in (structured_collections or []) if sc not in _scs)

        self._vector_by_name: dict[str, VectorCollection] = {vc.name: vc for vc in _vcs}
        self._structured_by_name: dict[str, StructuredCollection] = {sc.name: sc for sc in _scs}
        self._summarize_by_name: dict[str, SummarizeCollection] = {smc.name: smc for smc in _smcs}

        # Auto-generate steps from collection order when not explicitly provided
        if steps is None:
            _steps: list[tuple[str, str]] = []
            for vc in _vcs:
                _steps.append(("chunk-embed-upsert", vc.name))
            for sc in _scs:
                _steps.append(("extract-structured", sc.name))
            for smc in _smcs:
                _steps.append(("summarize-embed-upsert", smc.name))
            self._steps = _steps
        else:
            self._steps = list(steps)

    # ------------------------------------------------------------------
    # Backward-compat single-item accessors
    # ------------------------------------------------------------------

    @property
    def _vector_collection(self) -> VectorCollection | None:
        """First registered VectorCollection, or ``None``."""
        return next(iter(self._vector_by_name.values()), None)

    @property
    def _structured_collection(self) -> StructuredCollection | None:
        """First registered StructuredCollection, or ``None``."""
        return next(iter(self._structured_by_name.values()), None)

    @property
    def vector_collection(self) -> VectorCollection | None:
        """First registered VectorCollection, or ``None`` (backward compat)."""
        return self._vector_collection

    @property
    def structured_collection(self) -> StructuredCollection | None:
        """First registered StructuredCollection, or ``None`` (backward compat)."""
        return self._structured_collection

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
        """Names of all vector collections (chunk-embed and summarize), in step order."""
        seen: list[str] = []
        for tool, name in self._steps:
            if tool in ("chunk-embed-upsert", "summarize-embed-upsert") and name not in seen:
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
            if tool not in ("chunk-embed-upsert", "summarize-embed-upsert"):
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            col = self._vector_by_name.get(name) or self._summarize_by_name.get(name)
            explicit = col.description if col is not None else ""
            desc = explicit or _DEFAULT_VC_DESCRIPTIONS.get(name, name)
            seen.append((name, desc))
        return seen

    def runner_resources(
        self,
    ) -> tuple[StructuredStoreBase | None, VectorStoreBase | None, EmbeddingBase | None, str | None]:
        """Return ``(structured_store, vector_store, embedder, default_vector_collection)`` for Runner.

        Selects the first chunk-embed-upsert collection's store/embedder as the
        default vector resources (falls back to summarize-embed-upsert if none).
        """
        vector_store: VectorStoreBase | None = None
        embedder: EmbeddingBase | None = None
        default_vc: str | None = None

        for tool, name in self._steps:
            if tool == "chunk-embed-upsert" and name in self._vector_by_name:
                vc = self._vector_by_name[name]
                vector_store, embedder, default_vc = vc.store, vc.embedder, vc.name
                break

        if vector_store is None:
            for tool, name in self._steps:
                if tool == "summarize-embed-upsert" and name in self._summarize_by_name:
                    smc = self._summarize_by_name[name]
                    vector_store, embedder, default_vc = smc.store, smc.embedder, smc.name
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
            elif tool == "summarize-embed-upsert":
                await self._run_summarize_embed_upsert(doc, collection_name)
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
        vc = self._vector_by_name.get(collection_name)
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

    async def _run_summarize_embed_upsert(self, doc: Document, collection_name: str) -> None:
        smc = self._summarize_by_name.get(collection_name)
        if smc is None:
            logger.warning(
                "ingestion_pipeline.summarize_embed_upsert.unknown_collection name=%s collection=%s",
                self.name, collection_name,
            )
            return

        summary = await self._generate_summary(doc, smc)
        if not summary:
            logger.debug(
                "ingestion_pipeline.summarize_embed_upsert.empty_summary name=%s doc_id=%s",
                self.name, doc.doc_id,
            )
            return

        (embedding,) = await smc.embedder.embed([summary])
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}__summary",
            doc_id=doc.doc_id,
            text=summary,
            embedding=embedding,
        )
        await smc.store.upsert(smc.name, [chunk])
        logger.debug(
            "ingestion_pipeline.summarize_embed_upsert.upserted name=%s doc_id=%s collection=%s",
            self.name, doc.doc_id, collection_name,
        )

    async def _generate_summary(self, doc: Document, smc: SummarizeCollection) -> str | None:
        messages: list[ChatMessage] = [
            {"role": "system", "content": smc.prompt},
            {"role": "user", "content": doc.text},
        ]
        try:
            result = await smc.llm.complete(messages, max_tokens=smc.max_tokens)
            return result.get("content") or None
        except Exception:
            logger.exception(
                "ingestion_pipeline.generate_summary.failed name=%s doc_id=%s collection=%s",
                self.name, doc.doc_id, smc.name,
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
