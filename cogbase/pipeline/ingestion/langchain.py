"""LangChain text-splitter adapter for ChunkerBase.

Wraps any ``langchain_text_splitters.TextSplitter`` so it can be used
wherever a ``ChunkerBase`` is expected.

Install the extra dependency before use::

    pip install "cogbase[langchain]"

Example::

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from cogbase.pipeline.ingestion.langchain import LangChainChunker

    chunker = LangChainChunker(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
    await ingest(text, doc_id, chunker=chunker, ...)
"""

from langchain_text_splitters import TextSplitter

from cogbase.core.models import Chunk
from cogbase.pipeline.ingestion.base import ChunkerBase


class LangChainChunker(ChunkerBase):
    """Adapts any LangChain ``TextSplitter`` to the ``ChunkerBase`` interface.

    Args:
        splitter: Any ``langchain_text_splitters.TextSplitter`` instance
                  (e.g. ``RecursiveCharacterTextSplitter``, ``TokenTextSplitter``).
    """

    def __init__(self, splitter: TextSplitter) -> None:
        self._splitter = splitter

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        if not text:
            return []
        return [
            Chunk(
                doc_id=doc_id,
                text=piece,
                metadata={"chunk_index": str(i)},
            )
            for i, piece in enumerate(self._splitter.split_text(text))
            if piece
        ]
