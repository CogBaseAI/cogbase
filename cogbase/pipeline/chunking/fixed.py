"""Fixed-size sliding-window chunker."""

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.chunking.base import ChunkerBase


class FixedSizeChunker(ChunkerBase):
    """Splits text into overlapping fixed-size windows measured in characters.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters from the end of one chunk that are
                    repeated at the start of the next.  Must be less than
                    ``chunk_size``.

    Each chunk carries ``{"chunk_index": str(n)}`` in its metadata.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []

        stride = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(doc.text):
            end = min(start + self.chunk_size, len(doc.text))
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{index}",
                    doc_id=doc.doc_id,
                    text=doc.text[start:end],
                    metadata={"chunk_index": str(index)},
                    char_offset=start,
                    char_length=end - start,
                )
            )
            index += 1
            start += stride

        return chunks
