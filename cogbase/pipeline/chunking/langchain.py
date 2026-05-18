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

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.chunking.base import ChunkerBase


class LangChainChunker(ChunkerBase):
    """Adapts any LangChain ``TextSplitter`` to the ``ChunkerBase`` interface.

    Args:
        splitter: Any ``langchain_text_splitters.TextSplitter`` instance
                  (e.g. ``RecursiveCharacterTextSplitter``, ``TokenTextSplitter``).
    """

    def __init__(self, splitter: TextSplitter) -> None:
        self._splitter = splitter

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        chunks: list[Chunk] = []
        search_from = 0
        for i, piece in enumerate(self._splitter.split_text(doc.text)):
            if not piece:
                continue
            offset = doc.text.find(piece, search_from)
            if offset == -1:
                char_offset = None
                char_length = None
            else:
                char_offset = offset
                char_length = len(piece)
                search_from = offset + 1
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_{i}",
                doc_id=doc.doc_id,
                text=piece,
                metadata={"chunk_index": str(i)},
                char_offset=char_offset,
                char_length=char_length,
            ))
        return chunks
