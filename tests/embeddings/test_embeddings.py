"""Tests for EmbeddingBase contract and SentenceTransformersEmbedding."""

import os
from pathlib import Path

import pytest

from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase, OpenAIEmbedding, SentenceTransformersEmbedding

# Load .env from the repo root so OPENAI_API_KEY is available when present.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(texts: list[str], doc_id: str = "doc-1") -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, text=t, metadata={"chunk_index": str(i)})
        for i, t in enumerate(texts)
    ]


async def assert_embedder_contract(embedder: EmbeddingBase, chunks: list[Chunk]) -> list[Chunk]:
    """Invariants every compliant embedder must satisfy."""
    result = await embedder.embed(chunks)
    assert isinstance(result, list)
    assert len(result) == len(chunks)
    for original, embedded in zip(chunks, result):
        assert isinstance(embedded, Chunk)
        assert embedded.chunk_id == original.chunk_id
        assert embedded.doc_id == original.doc_id
        assert embedded.text == original.text
        assert embedded.embedding is not None
        assert isinstance(embedded.embedding, list)
        assert len(embedded.embedding) > 0
        assert all(isinstance(v, float) for v in embedded.embedding)
    # Input chunks must not be mutated
    for chunk in chunks:
        assert chunk.embedding is None
    return result


# ---------------------------------------------------------------------------
# EmbeddingBase — abstract interface
# ---------------------------------------------------------------------------

class TestEmbeddingBaseIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EmbeddingBase()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_custom_embedder_satisfies_contract(self):
        """Any EmbeddingBase subclass that sets a fixed vector passes the contract."""

        class ConstantEmbedding(EmbeddingBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                return [c.model_copy(update={"embedding": [0.1, 0.2, 0.3]}) for c in chunks]

        chunks = make_chunks(["hello world", "foo bar"])
        await assert_embedder_contract(ConstantEmbedding(), chunks)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        class ConstantEmbedding(EmbeddingBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                return [c.model_copy(update={"embedding": [1.0]}) for c in chunks]

        assert await ConstantEmbedding().embed([]) == []


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


# ---------------------------------------------------------------------------
# OpenAIEmbedding — requires openai package and OPENAI_API_KEY in .env
# ---------------------------------------------------------------------------

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
        chunks = make_chunks(["single sentence"])
        result = await embedder.embed(chunks)
        assert len(result) == 1
        assert result[0].embedding is not None

    @pytest.mark.asyncio
    async def test_embedding_dimension_default(self, embedder):
        # text-embedding-3-small native dimensionality is 1536
        chunks = make_chunks(["dimension check"])
        result = await embedder.embed(chunks)
        assert len(result[0].embedding) == 1536

    @pytest.mark.asyncio
    async def test_embedding_dimension_truncated(self, embedder_with_dimensions):
        chunks = make_chunks(["truncated dimension check"])
        result = await embedder_with_dimensions.embed(chunks)
        assert len(result[0].embedding) == 256

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

    def test_is_subclass(self):
        assert issubclass(OpenAIEmbedding, EmbeddingBase)
