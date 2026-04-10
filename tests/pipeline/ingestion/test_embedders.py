"""Tests for EmbedderBase contract and SentenceTransformersEmbedder."""

import pytest

from cogbase.core.models import Chunk
from cogbase.pipeline.ingestion.embedder import EmbedderBase, SentenceTransformersEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(texts: list[str], doc_id: str = "doc-1") -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, text=t, metadata={"chunk_index": str(i)})
        for i, t in enumerate(texts)
    ]


async def assert_embedder_contract(embedder: EmbedderBase, chunks: list[Chunk]) -> list[Chunk]:
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
# EmbedderBase — abstract interface
# ---------------------------------------------------------------------------

class TestEmbedderBaseIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EmbedderBase()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_custom_embedder_satisfies_contract(self):
        """Any EmbedderBase subclass that sets a fixed vector passes the contract."""

        class ConstantEmbedder(EmbedderBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                return [c.model_copy(update={"embedding": [0.1, 0.2, 0.3]}) for c in chunks]

        chunks = make_chunks(["hello world", "foo bar"])
        await assert_embedder_contract(ConstantEmbedder(), chunks)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        class ConstantEmbedder(EmbedderBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                return [c.model_copy(update={"embedding": [1.0]}) for c in chunks]

        assert await ConstantEmbedder().embed([]) == []


# ---------------------------------------------------------------------------
# SentenceTransformersEmbedder — requires the sentence-transformers package
# ---------------------------------------------------------------------------

sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed",
)


class TestSentenceTransformersEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        return SentenceTransformersEmbedder("all-MiniLM-L6-v2")

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
        assert issubclass(SentenceTransformersEmbedder, EmbedderBase)

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
