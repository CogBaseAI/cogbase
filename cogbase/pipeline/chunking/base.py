"""Abstract contract for text chunkers."""

import abc

from cogbase.core.models import Chunk, Document


class ChunkerBase(abc.ABC):
    """Split a document's text into ``Chunk`` objects.

    Implement this class to plug in a custom chunking strategy.  The pipeline
    accepts any ``ChunkerBase`` instance via dependency injection — CogBase does
    not need to know about the implementation.

    Example::

        class MyChunker(ChunkerBase):
            def chunk(self, doc: Document) -> list[Chunk]:
                ...

    ``metadata`` may carry chunker-specific fields.
    Embeddings are ``None`` on output — the pipeline fills them in separately.
    """

    @abc.abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Return an ordered list of chunks for *doc*.

        Args:
            doc: Source document whose ``text`` is split and ``doc_id`` is
                 propagated onto every returned chunk.

        Returns:
            Ordered list of ``Chunk`` objects.  ``embedding`` is always ``None``
            — the pipeline attaches embeddings in a later step.
        """

    def _make_chunk(
        self,
        doc: Document,
        index: int,
        text: str,
        char_offset: int,
        char_length: int,
    ) -> Chunk:
        # pipeline will generate embedding, set chunk.embedding, and copy over doc.metadata to each chunk
        return Chunk(
            chunk_id=f"{doc.doc_id}_{index}",
            doc_id=doc.doc_id,
            text=text,
            metadata={},
            char_offset=char_offset,
            char_length=char_length,
        )
