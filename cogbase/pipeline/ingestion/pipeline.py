"""End-to-end ingestion pipeline: text → chunks → embeddings → vector store."""

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.stores.base import StructuredStoreBase, VectorStoreBase


async def setup_extraction(
    extractors: list[ExtractorBase],
    structured_store: StructuredStoreBase,
) -> None:
    """Create structured store collections for a set of extractors.

    Call this once at application startup before ingesting any documents.
    ``create_collection`` is idempotent, so re-running on restart is safe.

    Args:
        extractors:       Extractors whose collections should be initialised.
        structured_store: Store to create the collections in.
    """
    for extractor in extractors:
        await structured_store.create_collection(extractor.schema)


async def ingest(
    doc: Document,
    *,
    chunker: ChunkerBase,
    embedder: EmbeddingBase,
    vector_store: VectorStoreBase,
    extractors: list[ExtractorBase] | None = None,
    structured_store: StructuredStoreBase | None = None,
) -> list[Chunk]:
    """Chunk, embed, store, and optionally extract structured records from a document.

    This is the primary entry point for the ingestion layer.  It wires the
    pipeline steps together in order:

    1. **Chunk** — split ``doc.text`` into overlapping windows via *chunker*.
    2. **Embed** — generate a dense vector for each chunk via *embedder*.
    3. **Store (vector)** — upsert the embedded chunks into *vector_store*.
    4. **Extract** — run each extractor in *extractors* over *doc* and save
       results to *structured_store* (skipped when either is ``None``).

    Collections must already exist in *structured_store* before calling this
    function — call ``setup_extraction`` once at startup to create them.

    Args:
        doc:              Document to ingest.  ``doc.doc_id`` is the stable
                          identifier used for later retrieval and deletion.
        chunker:          ``ChunkerBase`` implementation that splits the document.
        embedder:         ``EmbeddingBase`` implementation that returns embeddings.
        vector_store:     ``VectorStoreBase`` implementation that persists chunks.
        extractors:       Optional list of ``ExtractorBase`` implementations.
                          Each extractor pulls a different record type (facts,
                          entities, clauses, events, …) from *doc*.  Ignored
                          when *structured_store* is ``None``.
        structured_store: ``StructuredStoreBase`` implementation that persists
                          extracted records.  Ignored when *extractors* is
                          ``None`` or empty.

    Returns:
        The embedded ``Chunk`` objects that were upserted, in chunk order.
        Returns an empty list when ``doc.text`` is empty or the chunker produces
        no chunks.
    """
    chunks = chunker.chunk(doc)
    if not chunks:
        return []

    embeddings = await embedder.embed([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks."
        )
    embedded = [
        chunk.model_copy(update={"embedding": embedding})
        for chunk, embedding in zip(chunks, embeddings)
    ]
    await vector_store.upsert(embedded)

    if extractors and structured_store:
        for extractor in extractors:
            record = await extractor.extract(doc)
            if record is not None:
                await structured_store.save(extractor.collection, [record])

    return embedded
