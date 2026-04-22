"""Tests for SentenceTransformersEmbedding."""

import pytest

from cogbase.embeddings import EmbeddingBase, SentenceTransformersEmbedding
from tests.embeddings.test_embeddings import make_chunks, assert_embedder_contract


# ---------------------------------------------------------------------------
# SentenceTransformersEmbedding — requires the sentence-transformers package
# ---------------------------------------------------------------------------

sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed",
)


class TestSentenceTransformersEmbedding:
    @pytest.fixture(scope="class")
    def embedder(self):
        return SentenceTransformersEmbedding("all-MiniLM-L6-v2")

    @pytest.mark.asyncio
    async def test_contract(self, embedder):
        chunks = make_chunks(["The quick brown fox.", "Jumps over the lazy dog."])
        await assert_embedder_contract(embedder, chunks)

    @pytest.mark.asyncio
    async def test_embedding_dimension(self, embedder):
        # all-MiniLM-L6-v2 produces 384-dim vectors
        result = await embedder.embed(["test sentence"])
        assert len(result[0]) == 384

    @pytest.mark.asyncio
    async def test_empty_input(self, embedder):
        assert await embedder.embed([]) == []

    @pytest.mark.asyncio
    async def test_single_chunk(self, embedder):
        result = await embedder.embed(["single sentence"])
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_is_subclass(self):
        assert issubclass(SentenceTransformersEmbedding, EmbeddingBase)

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, embedder):
        result = await embedder.embed(["apple fruit", "quantum physics"])
        assert result[0] != result[1]

    @pytest.mark.asyncio
    async def test_same_text_same_embedding(self, embedder):
        text = "deterministic embedding"
        r1 = await embedder.embed([text])
        r2 = await embedder.embed([text])
        assert r1[0] == pytest.approx(r2[0], abs=1e-6)
