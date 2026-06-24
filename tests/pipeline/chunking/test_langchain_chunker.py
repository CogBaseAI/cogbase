"""Tests for LangChainChunker and build_recursive_chunker."""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from cogbase.core.models import Document
from cogbase.pipeline.chunking.base import ChunkerBase
from cogbase.pipeline.chunking.langchain import LangChainChunker, build_recursive_chunker
from tests.pipeline.chunking.test_chunkers import assert_chunker_contract


class TestBuildRecursiveChunker:
    def test_returns_langchain_chunker(self):
        assert isinstance(build_recursive_chunker(chunk_size=200, overlap=20), LangChainChunker)

    def test_contract(self):
        assert_chunker_contract(build_recursive_chunker(200, 20), "hello world " * 50, "doc-1")

    def test_splits_at_separator_not_mid_word(self):
        # chunk_size=40 forces a split on 47-char text. The ". " separator is preferred
        # over a raw character cut. keep_separator puts ". " at the start of chunk[1].
        # FixedSizeChunker(40) would produce "...Sentence two end" / "nd here." instead.
        text = "Sentence one ends here. Sentence two ends here."
        chunker = build_recursive_chunker(chunk_size=40, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) == 2
        assert chunks[0].text == "Sentence one ends here"
        assert chunks[1].text == ". Sentence two ends here."

    def test_no_chunk_exceeds_chunk_size(self):
        chunker = build_recursive_chunker(chunk_size=50, overlap=0)
        text = "Word. " * 30
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert all(len(c.text) <= 50 for c in chunks)

    def test_overlap_propagated(self):
        # With overlap=10, consecutive chunks should share a suffix/prefix substring.
        chunker = build_recursive_chunker(chunk_size=30, overlap=10)
        text = "abcde fghij klmno pqrst uvwxy " * 5
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) > 1
        # The tail of chunk[i] must appear at the start of chunk[i+1].
        for a, b in zip(chunks, chunks[1:]):
            assert b.text.startswith(a.text[-10:]) or a.text[-5:] in b.text


class TestLangChainChunkerContract:
    def test_is_chunker_base(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        assert isinstance(LangChainChunker(splitter), ChunkerBase)

    def test_contract_recursive(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        assert_chunker_contract(LangChainChunker(splitter), "hello world " * 50, "doc-1")

    def test_contract_character(self):
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator=" ")
        assert_chunker_contract(LangChainChunker(splitter), "hello world " * 50, "doc-1")


class TestLangChainChunkerBehavior:
    def test_empty_text_returns_empty(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        assert LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text="")) == []

    def test_doc_id_propagated(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="my-doc", text="hello world " * 20))
        assert all(c.doc_id == "my-doc" for c in chunks)

    def test_chunk_id_uses_doc_id_and_index(self):
        splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0, separator=" ")
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text="hello world"))
        assert [c.chunk_id for c in chunks] == [f"doc-1_{i}" for i in range(len(chunks))]

    def test_embedding_is_none(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text="some text " * 10))
        assert all(c.embedding is None for c in chunks)

    def test_short_text_single_chunk(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text="short"))
        assert len(chunks) == 1
        assert chunks[0].text == "short"

    def test_char_offset_and_length_set(self):
        splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0, separator="")
        text = "abcdefghij"
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text=text))
        for chunk in chunks:
            assert chunk.char_offset is not None
            assert chunk.char_length is not None
            assert text[chunk.char_offset : chunk.char_offset + chunk.char_length] == chunk.text

    def test_char_offset_matches_text_slice_multi_chunk(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        text = "hello world foo bar baz qux quux corge grault"
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) > 1
        for chunk in chunks:
            if chunk.char_offset is not None:
                assert text[chunk.char_offset : chunk.char_offset + chunk.char_length] == chunk.text

    def test_doc_metadata_not_copied_by_chunker(self):
        # Chunker emits metadata={} — the pipeline copies filtered doc.metadata onto chunks later.
        splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0, separator=" ")
        doc = Document(doc_id="doc-1", text="hello world", metadata={"source": "test", "author": "alice"})
        chunks = LangChainChunker(splitter).chunk(doc)
        for chunk in chunks:
            assert chunk.metadata == {}


class TestChineseSeparators:
    def test_chinese_separators_present(self):
        from cogbase.pipeline.chunking.langchain import _SENTENCE_SEPARATORS

        assert "。" in _SENTENCE_SEPARATORS
        assert "！" in _SENTENCE_SEPARATORS
        assert "？" in _SENTENCE_SEPARATORS

    def test_chinese_separators_before_english_period(self):
        # Chinese punctuation must be tried before ". " so mixed-script text
        # prefers Chinese sentence boundaries.
        from cogbase.pipeline.chunking.langchain import _SENTENCE_SEPARATORS

        cn_period_idx = _SENTENCE_SEPARATORS.index("。")
        en_period_idx = _SENTENCE_SEPARATORS.index(". ")
        assert cn_period_idx < en_period_idx

    def test_splits_on_chinese_period(self):
        # "。" sits between two 4-char phrases; chunk_size=5 forces a split there.
        text = "你好世界。再见朋友"
        chunker = build_recursive_chunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) == 2
        assert chunks[0].text == "你好世界"
        assert chunks[1].text == "。再见朋友"

    def test_splits_on_chinese_exclamation(self):
        text = "你好世界！再见朋友"
        chunker = build_recursive_chunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) == 2
        assert chunks[0].text == "你好世界"
        assert chunks[1].text == "！再见朋友"

    def test_splits_on_chinese_question(self):
        text = "你好世界？再见朋友"
        chunker = build_recursive_chunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(Document(doc_id="doc-1", text=text))
        assert len(chunks) == 2
        assert chunks[0].text == "你好世界"
        assert chunks[1].text == "？再见朋友"

