"""End-to-end ingestion pipeline: text → chunks → embeddings → vector store."""

from cogbase.core.models import Chunk
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.embedder import EmbedderBase
from cogbase.stores.base import VectorStoreBase


async def ingest(
    text: str,
    doc_id: str,
    *,
    chunker: ChunkerBase,
    embedder: EmbedderBase,
    vector_store: VectorStoreBase,
) -> list[Chunk]:
    """Chunk, embed, and store a document's text.

    This is the primary entry point for the ingestion layer.  It wires the
    three pipeline steps together in order:

    1. **Chunk** — split *text* into overlapping windows via *chunker*.
    2. **Embed** — attach a dense vector to each chunk via *embedder*.
    3. **Store** — upsert the embedded chunks into *vector_store*.

    All three dependencies are injected, so any combination of implementations
    can be composed without changing this function.

    Args:
        text:         Full document text to ingest.
        doc_id:       Stable identifier for the source document.  Used for
                      later retrieval and deletion.
        chunker:      ``ChunkerBase`` implementation that splits *text*.
        embedder:     ``EmbedderBase`` implementation that populates embeddings.
        vector_store: ``VectorStoreBase`` implementation that persists chunks.

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
    return embedded
