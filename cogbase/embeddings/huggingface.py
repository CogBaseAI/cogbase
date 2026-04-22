"""HuggingFace sentence-transformers based implementation of EmbeddingBase."""

import asyncio
import functools
import logging

from cogbase.embeddings.base import EmbeddingBase

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
