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
