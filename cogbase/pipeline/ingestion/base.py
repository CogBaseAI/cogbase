"""Abstract contract for text chunkers."""

import abc

from cogbase.core.models import Chunk


class ChunkerBase(abc.ABC):
    """Split a document's text into ``Chunk`` objects.

    Implement this class to plug in a custom chunking strategy.  The pipeline
    accepts any ``ChunkerBase`` instance via dependency injection — CogBase does
    not need to know about the implementation.

    Example::

        class MyChunker(ChunkerBase):
            def chunk(self, text: str, doc_id: str) -> list[Chunk]:
                ...

        await ingest(text, doc_id, chunker=MyChunker(), ...)

    ``metadata`` may carry chunker-specific fields (e.g. ``{"chunk_index": "0"}``).
    Embeddings are ``None`` on output — the pipeline fills them in separately.
    """

    @abc.abstractmethod
    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Return an ordered list of chunks for *text*.

        Args:
            text:   Full document text.
            doc_id: Identifier of the source document; set on every returned chunk.

        Returns:
            Ordered list of ``Chunk`` objects.  ``embedding`` is always ``None``
            — the pipeline attaches embeddings in a later step.
        """
