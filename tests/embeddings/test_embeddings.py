"""Tests for EmbeddingBase."""

import pytest

from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(texts: list[str], doc_id: str = "doc-1") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}_{i}", doc_id=doc_id, text=t, metadata={"chunk_index": str(i)})
        for i, t in enumerate(texts)
    ]


async def assert_embedder_contract(
    embedder: EmbeddingBase,
    chunks: list[Chunk],
) -> list[list[float]]:
    """Invariants every compliant embedder must satisfy."""
    result = await embedder.embed([chunk.text for chunk in chunks])
    assert isinstance(result, list)
    assert len(result) == len(chunks)
    for embedding in result:
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)
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
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        chunks = make_chunks(["hello world", "foo bar"])
        await assert_embedder_contract(ConstantEmbedding(), chunks)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        class ConstantEmbedding(EmbeddingBase):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

        assert await ConstantEmbedding().embed([]) == []
