"""Fixed-size sliding-window chunker."""

from cogbase.core.models import Chunk
from cogbase.pipeline.ingestion.base import ChunkerBase


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

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        if not text:
            return []

        stride = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=text[start:end],
                    metadata={"chunk_index": str(index)},
                )
            )
            index += 1
            start += stride

        return chunks
