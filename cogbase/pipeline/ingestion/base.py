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

        await ingest(doc, chunker=MyChunker(), ...)

    ``metadata`` may carry chunker-specific fields (e.g. ``{"chunk_index": "0"}``).
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
