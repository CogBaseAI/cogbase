"""Abstract contract and built-in implementations for chunk embedders."""

import abc
import asyncio
import functools
import logging
from typing import Any

from cogbase.core.models import Chunk

logger = logging.getLogger(__name__)


class EmbeddingBase(abc.ABC):
    """Attach embeddings to a list of ``Chunk`` objects.

    Implement this class to plug in a custom embedding backend.  The pipeline
    accepts any ``EmbeddingBase`` instance via dependency injection.

    Example::

        class MyEmbedding(EmbeddingBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                ...

        await ingest(text, doc_id, chunker=..., embedder=MyEmbedding(), ...)

    The interface is async because production embedders typically make HTTP
    calls (OpenAI, Cohere, etc.).  CPU-bound local models should offload to a
    thread pool via ``asyncio.get_event_loop().run_in_executor``.

    The input chunks are never mutated.  Implementations must return new
    ``Chunk`` objects (or copies) with ``embedding`` populated.
    """

    @abc.abstractmethod
    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return *chunks* with the ``embedding`` field populated.

        Args:
            chunks: Chunks to embed. May be empty — return ``[]`` in that case.

        Returns:
            Same chunks in the same order, each with ``embedding`` set to a
            non-None list of floats.  Input chunks are not mutated.
        """


class SentenceTransformersEmbedding(EmbeddingBase):
    """Embedder backed by a ``sentence-transformers`` model.

    Vectors are L2-normalised before being attached to chunks, which makes
    cosine similarity equivalent to dot-product — consistent with
    ``FAISSVectorStore`` (IndexFlatIP).

    Install the extra dependency before use::

        pip install "cogbase[sentence-transformers]"

    Args:
        model_name: Any model name accepted by ``SentenceTransformer``.
                    Defaults to ``"all-MiniLM-L6-v2"`` (384-dim, fast, good
                    general-purpose quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            logger.exception("sentence_transformers_import_failed")
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersEmbedding. "
                'Install it with: pip install "cogbase[sentence-transformers]"'
            ) from exc

        self._model = SentenceTransformer(model_name)

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        loop = asyncio.get_event_loop()
        encode = functools.partial(
            self._model.encode,
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = await loop.run_in_executor(None, encode)
        return [
            c.model_copy(update={"embedding": vec.tolist()})
            for c, vec in zip(chunks, vectors)
        ]


class OpenAIEmbedding(EmbeddingBase):
    """Embedder backed by the OpenAI Embeddings API.

    Sends all chunk texts in a single batched API call and attaches the
    returned vectors to the chunks.  The client must be an async
    OpenAI-compatible client (``openai.AsyncOpenAI`` or any compatible
    drop-in).

    Install the extra dependency before use::

        pip install "cogbase[openai]"

    Args:
        client:     Async OpenAI-compatible client.
        model:      Embedding model name.  Defaults to
                    ``"text-embedding-3-small"`` (1536-dim).
        dimensions: Optional output dimension.  When set, the API truncates
                    the embedding to this length (supported by
                    ``text-embedding-3-*`` models).  ``None`` returns the
                    model's native dimensionality.

    Example::

        import openai
        from cogbase.embeddings import OpenAIEmbedding

        client = openai.AsyncOpenAI(api_key="...")
        embedder = OpenAIEmbedding(client, model="text-embedding-3-small")
        chunks = await embedder.embed(chunks)
    """

    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        kwargs: dict[str, Any] = {
            "input": [c.text for c in chunks],
            "model": self._model,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        logger.debug("openai_embedder.request model=%s chunks=%d", self._model, len(chunks))
        response = await self._client.embeddings.create(**kwargs)
        logger.debug("openai_embedder.response usage=%s", response.usage)

        # The API returns embeddings in the same order as the input.
        vectors = [item.embedding for item in response.data]
        return [
            c.model_copy(update={"embedding": vec})
            for c, vec in zip(chunks, vectors)
        ]
