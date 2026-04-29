"""Tests for ChunkerBase contract and FixedSizeChunker."""

import pytest

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker


# ---------------------------------------------------------------------------
# ChunkerBase — contract tests applied to any implementation
# ---------------------------------------------------------------------------

def assert_chunker_contract(chunker: ChunkerBase, text: str, doc_id: str) -> list[Chunk]:
    """Run invariants every compliant chunker must satisfy."""
    chunks = chunker.chunk(Document(doc_id=doc_id, text=text))
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == doc_id
        assert chunk.embedding is None  # pipeline fills this later
        assert chunk.text  # no empty chunks
    return chunks


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunkerInit:
    def test_defaults(self):
        c = FixedSizeChunker()
        assert c.chunk_size == 1000
        assert c.overlap == 200

    def test_custom(self):
        c = FixedSizeChunker(chunk_size=500, overlap=50)
        assert c.chunk_size == 500
        assert c.overlap == 50

    def test_zero_overlap_allowed(self):
        FixedSizeChunker(chunk_size=100, overlap=0)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            FixedSizeChunker(chunk_size=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=-1)

    def test_overlap_equals_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=150)


class TestFixedSizeChunkerChunk:
    def test_empty_text_returns_empty(self):
        chunks = FixedSizeChunker().chunk(Document(doc_id="doc-1", text=""))
        assert chunks == []

    def test_contract(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        assert_chunker_contract(chunker, "hello world foo bar baz", "doc-1")

    def test_single_chunk_when_text_fits(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(Document(doc_id="doc-1", text="short text"))
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_chunk_count(self):
        # text=30 chars, chunk_size=10, overlap=2, stride=8
        # starts: 0, 8, 16, 24  → 4 chunks
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk(Document(doc_id="doc-1", text="a" * 30))
        assert len(chunks) == 4

    def test_overlap_content(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        text = "abcdefghij"  # 10 chars, stride=3: starts 0,3,6,9
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        # chunk[0] ends at index 5: "abcde"
        # chunk[1] starts at index 3: "defgh"
        assert chunks[0].text[-2:] == chunks[1].text[:2]

    def test_chunk_index_metadata(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text="abcdeabcde"))
        assert [c.metadata["chunk_index"] for c in chunks] == ["0", "1"]

    def test_chunk_id_uses_doc_id_and_index(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text="abcdeabcde"))
        assert [c.chunk_id for c in chunks] == ["doc-1_0", "doc-1_1"]

    def test_doc_id_propagated(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="my-doc", text="hello world"))
        assert all(c.doc_id == "my-doc" for c in chunks)

    def test_all_characters_covered(self):
        # Every position in the original text must appear in at least one chunk.
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        text = "the quick brown fox jumps over the lazy dog"
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        stride = chunker.chunk_size - chunker.overlap
        for i, chunk in enumerate(chunks):
            start = i * stride
            assert text[start : start + chunker.chunk_size] == chunk.text

    def test_char_offset_and_length_set(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        text = "abcdefghij"
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert chunks[0].char_offset == 0
        assert chunks[0].char_length == 5
        assert chunks[1].char_offset == 5
        assert chunks[1].char_length == 5

    def test_char_offset_with_overlap(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        # stride=3: starts at 0, 3, 6, 9
        text = "abcdefghij"
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert chunks[0].char_offset == 0
        assert chunks[1].char_offset == 3

    def test_last_chunk_char_length_does_not_exceed_text(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        text = "abcde"  # shorter than chunk_size
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) == 1
        assert chunks[0].char_offset == 0
        assert chunks[0].char_length == 5

    def test_char_offset_matches_text_slice(self):
        chunker = FixedSizeChunker(chunk_size=8, overlap=2)
        text = "the quick brown fox"
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        for chunk in chunks:
            assert text[chunk.char_offset : chunk.char_offset + chunk.char_length] == chunk.text


class TestFixedSizeChunkerIsChunkerBase:
    def test_is_subclass(self):
        assert issubclass(FixedSizeChunker, ChunkerBase)

    def test_custom_chunker_satisfies_contract(self):
        """Third-party chunkers only need to extend ChunkerBase."""

        class WordChunker(ChunkerBase):
            def chunk(self, doc: Document) -> list[Chunk]:
                return [
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{i}",
                        doc_id=doc.doc_id,
                        text=word,
                        metadata={"chunk_index": str(i)},
                    )
                    for i, word in enumerate(doc.text.split())
                    if word
                ]

        assert_chunker_contract(WordChunker(), "hello world foo", "doc-x")
