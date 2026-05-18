"""LangChain text-splitter adapter for ChunkerBase.

Wraps any ``langchain_text_splitters.TextSplitter`` so it can be used
wherever a ``ChunkerBase`` is expected.

Install the extra dependency before use::

    pip install "cogbase[langchain]"

Example::

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from cogbase.pipeline.chunking.langchain import LangChainChunker

    chunker = LangChainChunker(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
"""

from langchain_text_splitters import TextSplitter

from cogbase.core.models import Document
from cogbase.pipeline.chunking.base import ChunkerBase


class LangChainChunker(ChunkerBase):
    """Adapts any LangChain ``TextSplitter`` to the ``ChunkerBase`` interface.

    Args:
        splitter: Any ``langchain_text_splitters.TextSplitter`` instance
                  (e.g. ``RecursiveCharacterTextSplitter``, ``TokenTextSplitter``).
    """

    def __init__(self, splitter: TextSplitter) -> None:
        self._splitter = splitter

    def chunk(self, doc: Document) -> list:
        if not doc.text:
            return []
        chunks = []
        search_from = 0
        for i, piece in enumerate(self._splitter.split_text(doc.text)):
            if not piece:
                continue
            offset = doc.text.find(piece, search_from)
            if offset == -1:
                char_offset, char_length = None, None
            else:
                char_offset = offset
                char_length = len(piece)
                search_from = offset + 1
            chunks.append(self._make_chunk(doc, i, piece, char_offset, char_length))
        return chunks
