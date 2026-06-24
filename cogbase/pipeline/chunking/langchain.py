"""LangChain text-splitter adapter for ChunkerBase.

Wraps any ``langchain_text_splitters.TextSplitter`` so it can be used
wherever a ``ChunkerBase`` is expected.

Example::

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from cogbase.pipeline.chunking.langchain import LangChainChunker

    chunker = LangChainChunker(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from cogbase.core.models import Document
from cogbase.llms.compaction import estimate_tokens
from cogbase.pipeline.chunking.base import ChunkerBase

_SENTENCE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "; ", ", ", " ", ""]


def build_recursive_chunker(chunk_size: int, overlap: int) -> "LangChainChunker":
    """Build a LangChainChunker backed by RecursiveCharacterTextSplitter.

    Separators are ordered so splits prefer sentence boundaries over mid-sentence cuts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_SENTENCE_SEPARATORS,
    )
    return LangChainChunker(splitter)


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split *text* into overlapping, token-bounded windows on the best boundary.

    Backed by ``RecursiveCharacterTextSplitter`` with token-based sizing
    (:func:`estimate_tokens`) and the shared sentence-preferring separator
    hierarchy, so cuts prefer paragraph > line > sentence > word > character
    boundaries and consecutive windows overlap by up to ~*overlap_tokens*. A
    document within budget yields ``[text]``.

    Window text is not a verbatim slice of *text* — the splitter strips
    surrounding whitespace and re-joins on separators — but content within a unit
    is preserved, which is all the extractor needs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        separators=_SENTENCE_SEPARATORS,
        length_function=estimate_tokens,
    )
    return splitter.split_text(text)


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
