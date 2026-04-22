"""Abstract contract and built-in implementations for text embedders."""

import abc
import logging

logger = logging.getLogger(__name__)


class EmbeddingBase(abc.ABC):
    """Embed a list of texts into dense vectors.

    Implement this class to plug in a custom embedding backend.  The pipeline
    accepts any ``EmbeddingBase`` instance via dependency injection.

    Example::

        class MyEmbedding(EmbeddingBase):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                ...

        await ingest(text, doc_id, chunker=..., embedder=MyEmbedding(), ...)

    The interface is async because production embedders typically make HTTP
    calls (OpenAI, Cohere, etc.).  CPU-bound local models should offload to a
    thread pool via ``asyncio.get_event_loop().run_in_executor``.

    Implementations must return one embedding per input text, preserving
    order.
    """

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for *texts*.

        Args:
            texts: Texts to embed. May be empty; return ``[]`` in that case.

        Returns:
            One embedding vector per input text, in the same order.
        """
