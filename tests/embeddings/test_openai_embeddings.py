"""Tests for the configured embedding backend.

The embedder is loaded from .env.yaml (same config that ``api/main.py`` uses).
Falls back to OpenAI ``text-embedding-3-small`` via ``OPENAI_API_KEY`` when
.env.yaml is absent or contains no ``embedding`` section.
"""

import pytest

from cogbase.embeddings import EmbeddingBase
from cogbase.embeddings.openai import OpenAIEmbedding
from tests.embeddings.test_embeddings import make_chunks, assert_embedder_contract
from tests.live_setup import make_embedding

openai = pytest.importorskip("openai", reason="openai package not installed")

_embedder = make_embedding()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_embedder is None, reason="No embedding configured: set embedding in .env.yaml or OPENAI_API_KEY"),
]


class TestOpenAIEmbedding:
    @pytest.fixture(scope="class")
    def embedder(self):
        return _embedder

    @pytest.fixture(scope="class")
    def embedder_with_dimensions(self):
        return make_embedding(dimensions=256)

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
