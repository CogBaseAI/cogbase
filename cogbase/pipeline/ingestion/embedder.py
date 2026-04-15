"""Abstract contract and built-in implementations for chunk embedders."""

import abc
import asyncio
import functools
import logging

from cogbase.core.models import Chunk

logger = logging.getLogger(__name__)


class EmbedderBase(abc.ABC):
    """Attach embeddings to a list of ``Chunk`` objects.

    Implement this class to plug in a custom embedding backend.  The pipeline
    accepts any ``EmbedderBase`` instance via dependency injection.

    Example::

        class MyEmbedder(EmbedderBase):
            async def embed(self, chunks: list[Chunk]) -> list[Chunk]:
                ...

        await ingest(text, doc_id, chunker=..., embedder=MyEmbedder(), ...)

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


class SentenceTransformersEmbedder(EmbedderBase):
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
                "sentence-transformers is required for SentenceTransformersEmbedder. "
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
