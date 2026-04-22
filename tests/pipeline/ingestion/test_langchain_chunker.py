"""Tests for LangChainChunker."""

import pytest
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from cogbase.core.models import Document
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.pipeline.ingestion.langchain import LangChainChunker
from tests.pipeline.ingestion.test_chunkers import assert_chunker_contract


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

    def test_chunk_index_metadata(self):
        splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=0, separator=" ")
        chunks = LangChainChunker(splitter).chunk(Document(doc_id="doc-1", text="hello world"))
        assert [c.metadata["chunk_index"] for c in chunks] == [
            str(i) for i in range(len(chunks))
        ]

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
