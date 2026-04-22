"""Tests for the ingest() pipeline orchestrator."""

import pytest

from cogbase.core.models import Chunk, Document
from cogbase.pipeline.ingestion.base import ChunkerBase
from cogbase.embeddings import EmbeddingBase
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker
from cogbase.pipeline.ingestion.pipeline import ingest
from cogbase.stores.vector.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class StubEmbedding(EmbeddingBase):
    """Returns a fixed-dimension embedding for every chunk."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * self._dim for i, _ in enumerate(texts)]


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.fixture
    def chunker(self):
        return FixedSizeChunker(chunk_size=50, overlap=10)

    @pytest.fixture
    def embedder(self):
        return StubEmbedding(dim=4)

    @pytest.fixture
    def vector_store(self):
        return FAISSVectorStore(dim=4)

    @pytest.mark.asyncio
    async def test_returns_embedded_chunks(self, chunker, embedder, vector_store):
        text = "word " * 30  # 150 chars → several chunks
        result = await ingest(Document(doc_id="doc-1", text=text), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")

        assert isinstance(result, list)
        assert len(result) > 0
        for chunk in result:
            assert isinstance(chunk, Chunk)
            assert chunk.doc_id == "doc-1"
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 4

    @pytest.mark.asyncio
    async def test_chunks_stored_in_vector_store(self, chunker, embedder, vector_store):
        text = "sentence one. " * 20
        result = await ingest(Document(doc_id="doc-2", text=text), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")

        assert vector_store.ntotal == len(result)

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, chunker, embedder, vector_store):
        result = await ingest(Document(doc_id="doc-empty", text=""), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")
        assert result == []
        assert vector_store.ntotal == 0

    @pytest.mark.asyncio
    async def test_doc_id_preserved(self, chunker, embedder, vector_store):
        result = await ingest(Document(doc_id="my-doc", text="hello world " * 10), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")
        assert all(c.doc_id == "my-doc" for c in result)

    @pytest.mark.asyncio
    async def test_input_text_not_mutated(self, chunker, embedder, vector_store):
        text = "immutability test " * 10
        original = text
        await ingest(Document(doc_id="doc-3", text=text), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")
        assert text == original

    @pytest.mark.asyncio
    async def test_multiple_docs_stored_independently(self, chunker, embedder, vector_store):
        text = "alpha beta gamma delta " * 5
        r1 = await ingest(Document(doc_id="doc-a", text=text), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")
        r2 = await ingest(Document(doc_id="doc-b", text=text), chunker=chunker, embedder=embedder, vector_store=vector_store, collection="chunks")

        assert vector_store.ntotal == len(r1) + len(r2)

    @pytest.mark.asyncio
    async def test_embedder_called_with_chunker_output(self):
        """Embedder receives exactly the texts the chunker produced."""

        seen_texts: list[list[str]] = []

        class RecordingEmbedding(EmbeddingBase):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                seen_texts.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

        store = FAISSVectorStore(dim=2)
        chunker = FixedSizeChunker(chunk_size=20, overlap=0)
        text = "x" * 60  # exactly 3 chunks of 20 chars

        await ingest(Document(doc_id="doc-rec", text=text), chunker=chunker, embedder=RecordingEmbedding(), vector_store=store, collection="chunks")

        assert len(seen_texts) == 1
        assert len(seen_texts[0]) == 3
        assert all(chunk_text == "x" * 20 for chunk_text in seen_texts[0])
