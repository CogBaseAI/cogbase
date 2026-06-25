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


@pytest.mark.live
@pytest.mark.skipif(
    _embedder is None,
    reason="No embedding configured: set embedding in .env.yaml or OPENAI_API_KEY",
)
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


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data
        self.usage = None


class _FakeEmbeddingsClient:
    """Records each request and echoes one deterministic vector per input."""

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.embeddings = self

    async def create(self, *, input, model, **kwargs):
        self.batch_sizes.append(len(input))
        # Encode the text so order/identity can be asserted by the caller.
        return _FakeEmbeddingResponse([_FakeEmbeddingItem([float(len(t))]) for t in input])


class TestOpenAIEmbeddingBatching:
    """Sub-batching is exercised with a fake client, so no live key is needed."""

    @pytest.mark.asyncio
    async def test_splits_into_sub_batches_preserving_order(self):
        client = _FakeEmbeddingsClient()
        embedder = OpenAIEmbedding(client, batch_size=3)

        texts = [str(i) * i for i in range(1, 8)]  # 7 texts -> 3 + 3 + 1
        result = await embedder.embed(texts)

        assert client.batch_sizes == [3, 3, 1]
        assert result == [[float(len(t))] for t in texts]

    @pytest.mark.asyncio
    async def test_single_request_when_under_batch_size(self):
        client = _FakeEmbeddingsClient()
        embedder = OpenAIEmbedding(client, batch_size=100)
        await embedder.embed(["a", "bb", "ccc"])
        assert client.batch_sizes == [3]

    @pytest.mark.asyncio
    async def test_empty_input_makes_no_request(self):
        client = _FakeEmbeddingsClient()
        embedder = OpenAIEmbedding(client)
        assert await embedder.embed([]) == []
        assert client.batch_sizes == []

    def test_rejects_invalid_batch_size(self):
        with pytest.raises(ValueError):
            OpenAIEmbedding(_FakeEmbeddingsClient(), batch_size=0)

    def test_context_window_defaults_to_8k(self):
        embedder = OpenAIEmbedding(_FakeEmbeddingsClient())
        assert embedder.context_window == 8192

    def test_context_window_override(self):
        embedder = OpenAIEmbedding(_FakeEmbeddingsClient(), context_window=2048)
        assert embedder.context_window == 2048

    def test_rejects_invalid_context_window(self):
        with pytest.raises(ValueError):
            OpenAIEmbedding(_FakeEmbeddingsClient(), context_window=0)
