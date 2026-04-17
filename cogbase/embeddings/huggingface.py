"""HuggingFace sentence-transformers based implementation of EmbeddingBase."""

import asyncio
import functools
import logging

from cogbase.core.models import Chunk
from cogbase.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)


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
