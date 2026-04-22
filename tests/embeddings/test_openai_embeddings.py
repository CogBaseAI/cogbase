"""Tests for OpenAIEmbedding."""

import os
from pathlib import Path

import pytest

from cogbase.embeddings import EmbeddingBase, OpenAIEmbedding
from tests.embeddings.test_embeddings import make_chunks, assert_embedder_contract

# Load .env from the repo root so OPENAI_API_KEY is available when present.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


openai = pytest.importorskip("openai", reason="openai package not installed")

_openai_api_key = os.environ.get("OPENAI_API_KEY", "")
pytestmark_openai = pytest.mark.skipif(
    not _openai_api_key,
    reason="OPENAI_API_KEY not set in .env",
)


@pytestmark_openai
class TestOpenAIEmbedding:
    @pytest.fixture(scope="class")
    def embedder(self):
        client = openai.AsyncOpenAI(api_key=_openai_api_key)
        return OpenAIEmbedding(client, model="text-embedding-3-small")

    @pytest.fixture(scope="class")
    def embedder_with_dimensions(self):
        client = openai.AsyncOpenAI(api_key=_openai_api_key)
        return OpenAIEmbedding(client, model="text-embedding-3-small", dimensions=256)

    @pytest.mark.asyncio
    async def test_contract(self, embedder):
        chunks = make_chunks(["The quick brown fox.", "Jumps over the lazy dog."])
        await assert_embedder_contract(embedder, chunks)

    @pytest.mark.asyncio
    async def test_empty_input(self, embedder):
        assert await embedder.embed([]) == []

    @pytest.mark.asyncio
    async def test_single_chunk(self, embedder):
        result = await embedder.embed(["single sentence"])
        assert len(result) == 1
        assert len(result[0]) > 0

    @pytest.mark.asyncio
    async def test_embedding_dimension_default(self, embedder):
        # text-embedding-3-small native dimensionality is 1536
        result = await embedder.embed(["dimension check"])
        assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_embedding_dimension_truncated(self, embedder_with_dimensions):
        result = await embedder_with_dimensions.embed(["truncated dimension check"])
        assert len(result[0]) == 256

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, embedder):
        result = await embedder.embed(["apple fruit", "quantum physics"])
        assert result[0] != result[1]

    def test_is_subclass(self):
        assert issubclass(OpenAIEmbedding, EmbeddingBase)
