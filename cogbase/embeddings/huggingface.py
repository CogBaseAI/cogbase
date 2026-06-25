"""HuggingFace sentence-transformers based implementation of EmbeddingBase."""

import asyncio
import functools
import logging

from cogbase.embeddings.base import DEFAULT_CONTEXT_WINDOW, EmbeddingBase

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedding(EmbeddingBase):
    """Embedder backed by a ``sentence-transformers`` model.

    Vectors are L2-normalised, which makes cosine similarity equivalent to
    dot-product and stays consistent with ``FAISSVectorStore`` (IndexFlatIP).

    Install the extra dependency before use::

        pip install "cogbase[sentence-transformers]"

    Args:
        model_name: Any model name accepted by ``SentenceTransformer``.
                    Defaults to ``"all-MiniLM-L6-v2"`` (384-dim, fast, good
                    general-purpose quality).
        context_window: Maximum tokens accepted in a single input text. When
                    ``None`` (the default), the model's own ``max_seq_length``
                    is used, falling back to
                    :data:`~cogbase.embeddings.base.DEFAULT_CONTEXT_WINDOW`
                    when the model does not report one.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        context_window: int | None = None,
    ) -> None:
        if context_window is not None and context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            logger.exception("sentence_transformers_import_failed")
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersEmbedding. "
                'Install it with: pip install "cogbase[sentence-transformers]"'
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        self._context_window = (
            context_window
            or getattr(self._model, "max_seq_length", None)
            or DEFAULT_CONTEXT_WINDOW
        )

    @property
    def context_window(self) -> int:
        """Maximum tokens per input text — the override or the model's own."""
        return self._context_window

    @property
    def dimensions(self) -> int | None:
        """The model's embedding dimensionality, known once it is loaded."""
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        encode = functools.partial(
            self._model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = await loop.run_in_executor(None, encode)
        return [vec.tolist() for vec in vectors]
