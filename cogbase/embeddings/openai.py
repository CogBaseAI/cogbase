"""OpenAI embedding api based implementation of EmbeddingBase.

The provider that provides OpenAI compatible API can use this implementation.
"""

import logging
from typing import Any

from cogbase.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingBase):
    """Embedder backed by the OpenAI Embeddings API.

    Sends all texts in a single batched API call and returns the vectors. The
    client must be an async OpenAI-compatible client (``openai.AsyncOpenAI`` or
    any compatible drop-in).

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
        vectors = await embedder.embed(["hello world", "foo bar"])
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        logger.debug("openai_embedder.request model=%s texts=%d", self._model, len(texts))
        response = await self._client.embeddings.create(**kwargs)
        logger.debug("openai_embedder.response usage=%s", response.usage)

        # The API returns embeddings in the same order as the input.
        return [item.embedding for item in response.data]
