"""Tests for EmbeddingBase."""

import pytest

from cogbase.core.models import Chunk
from cogbase.embeddings import EmbeddingBase
from cogbase.embeddings.openai import OpenAIEmbedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(texts: list[str], doc_id: str = "doc-1") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}_{i}", doc_id=doc_id, text=t)
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


# ---------------------------------------------------------------------------
# EmbeddingBase.dimensions — reported without an embedding call
# ---------------------------------------------------------------------------

class TestEmbeddingDimensions:
    def test_base_default_is_none(self):
        """A subclass that only implements ``embed`` reports unknown dimensions."""

        class ConstantEmbedding(EmbeddingBase):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        assert ConstantEmbedding().dimensions is None

    def test_subclass_can_override(self):
        class FixedEmbedding(EmbeddingBase):
            @property
            def dimensions(self) -> int:
                return 8

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 8 for _ in texts]

        assert FixedEmbedding().dimensions == 8

    def test_openai_reports_configured_dimensions(self):
        # No client call: the property just surfaces the configured override.
        embedder = OpenAIEmbedding(client=None, dimensions=256)
        assert embedder.dimensions == 256

    def test_openai_native_dimension_is_none(self):
        # Left at the provider default — only an embedding response reveals it.
        embedder = OpenAIEmbedding(client=None)
        assert embedder.dimensions is None
