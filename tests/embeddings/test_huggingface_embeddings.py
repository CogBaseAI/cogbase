"""Tests for SentenceTransformersEmbedding."""

import os
from pathlib import Path

import pytest

from cogbase.core.models import Chunk
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
        chunks = make_chunks(["test sentence"])
        result = await embedder.embed(chunks)
        assert len(result[0].embedding) == 384

    @pytest.mark.asyncio
    async def test_empty_input(self, embedder):
        assert await embedder.embed([]) == []

    @pytest.mark.asyncio
    async def test_single_chunk(self, embedder):
        chunks = make_chunks(["single sentence"])
        result = await embedder.embed(chunks)
        assert len(result) == 1
        assert result[0].embedding is not None

    def test_is_subclass(self):
        assert issubclass(SentenceTransformersEmbedding, EmbeddingBase)

    @pytest.mark.asyncio
    async def test_input_not_mutated(self, embedder):
        chunks = make_chunks(["immutability check"])
        await embedder.embed(chunks)
        assert chunks[0].embedding is None

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, embedder):
        chunks = make_chunks(["apple fruit", "quantum physics"])
        result = await embedder.embed(chunks)
        assert result[0].embedding != result[1].embedding

    @pytest.mark.asyncio
    async def test_same_text_same_embedding(self, embedder):
        text = "deterministic embedding"
        c1, c2 = make_chunks([text, text])
        r1 = await embedder.embed([c1])
        r2 = await embedder.embed([c2])
        assert r1[0].embedding == pytest.approx(r2[0].embedding, abs=1e-6)
