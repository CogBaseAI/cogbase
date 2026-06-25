"""OpenAI embedding api based implementation of EmbeddingBase.

The provider that provides OpenAI compatible API can use this implementation.
"""

import logging
from typing import Any

from cogbase.embeddings.base import DEFAULT_CONTEXT_WINDOW, EmbeddingBase

logger = logging.getLogger(__name__)


#: Default maximum number of texts sent per embedding API request.  The OpenAI
#: Embeddings endpoint caps the input array (currently 2048 entries) and the
#: total tokens per request, so a document that chunks into thousands of
#: passages must be split across several requests.  Kept in sync with
#: ``EmbeddingConfig.batch_size`` so the config-driven path and a directly
#: constructed embedder behave identically.
DEFAULT_BATCH_SIZE = 500


class OpenAIEmbedding(EmbeddingBase):
    """Embedder backed by the OpenAI Embeddings API.

    Texts are embedded in sub-batches of at most ``batch_size`` per API request
    and the results concatenated, so arbitrarily long inputs (e.g. a document
    that chunks into thousands of passages) stay within the endpoint's
    per-request array/token limits.  The client must be an async
    OpenAI-compatible client (``openai.AsyncOpenAI`` or any compatible drop-in).

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
        batch_size: Maximum number of texts per API request.  Defaults to
                    :data:`DEFAULT_BATCH_SIZE`.
        context_window: Maximum tokens accepted in a single input text.
                    Defaults to :data:`~cogbase.embeddings.base.DEFAULT_CONTEXT_WINDOW`
                    (8192), matching ``text-embedding-3-*``'s 8191-token cap.

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
        batch_size: int = DEFAULT_BATCH_SIZE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._context_window = context_window

    @property
    def context_window(self) -> int:
        """The configured maximum tokens per input text."""
        return self._context_window

    @property
    def dimensions(self) -> int | None:
        """The configured output dimension, or ``None`` for the model's native
        dimensionality (which the API only reveals in an embedding response)."""
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]

            kwargs: dict[str, Any] = {
                "input": batch,
                "model": self._model,
            }
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions

            logger.debug(
                "openai_embedder.request model=%s texts=%d offset=%d total=%d",
                self._model, len(batch), start, len(texts),
            )
            response = await self._client.embeddings.create(**kwargs)
            logger.debug("openai_embedder.response usage=%s", response.usage)

            # The API returns embeddings in the same order as the input.
            embeddings.extend(item.embedding for item in response.data)

        return embeddings
