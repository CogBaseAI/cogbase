"""End-to-end ingestion pipeline: text → chunks → embeddings → vector store."""

from cogbase.core.models import Chunk
from cogbase.pipeline.extraction.base import ExtractorBase
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
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
    text: str,
    doc_id: str,
    *,
    chunker: ChunkerBase,
    embedder: EmbedderBase,
    vector_store: VectorStoreBase,
    extractors: list[ExtractorBase] | None = None,
    structured_store: StructuredStoreBase | None = None,
) -> list[Chunk]:
    """Chunk, embed, store, and optionally extract structured records from a document.

    This is the primary entry point for the ingestion layer.  It wires the
    pipeline steps together in order:

    1. **Chunk** — split *text* into overlapping windows via *chunker*.
    2. **Embed** — attach a dense vector to each chunk via *embedder*.
    3. **Store (vector)** — upsert the embedded chunks into *vector_store*.
    4. **Extract** — run each extractor in *extractors* over *text* and save
       results to *structured_store* (skipped when either is ``None``).

    Collections must already exist in *structured_store* before calling this
    function — call ``setup_extraction`` once at startup to create them.

    Args:
        text:             Full document text to ingest.
        doc_id:           Stable identifier for the source document.  Used for
                          later retrieval and deletion.
        chunker:          ``ChunkerBase`` implementation that splits *text*.
        embedder:         ``EmbedderBase`` implementation that populates embeddings.
        vector_store:     ``VectorStoreBase`` implementation that persists chunks.
        extractors:       Optional list of ``ExtractorBase`` implementations.
                          Each extractor pulls a different record type (facts,
                          entities, clauses, events, …) from *text*.  Ignored
                          when *structured_store* is ``None``.
        structured_store: ``StructuredStoreBase`` implementation that persists
                          extracted records.  Ignored when *extractors* is
                          ``None`` or empty.

    Returns:
        The embedded ``Chunk`` objects that were upserted, in chunk order.
        Returns an empty list when *text* is empty or the chunker produces no
        chunks.
    """
    chunks = chunker.chunk(text, doc_id)
    if not chunks:
        return []

    embedded = await embedder.embed(chunks)
    await vector_store.upsert(embedded)

    if extractors and structured_store:
        for extractor in extractors:
            record = await extractor.extract(text, doc_id)
            if record is not None:
                await structured_store.save(extractor.collection, [record])

    return embedded
