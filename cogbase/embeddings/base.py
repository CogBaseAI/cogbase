"""Abstract contract and built-in implementations for text embedders."""

import abc
import logging

logger = logging.getLogger(__name__)

# Conservative fallback context window (tokens) for a single input text. Most
# hosted embedding models cap one input around 8k tokens (e.g. OpenAI's
# ``text-embedding-3-*`` at 8191); a text beyond the window is silently
# truncated or rejected by the provider, so callers size passage chunks against
# this value. Concrete backends override with the configured window.
DEFAULT_CONTEXT_WINDOW = 8192


class EmbeddingBase(abc.ABC):
    """Embed a list of texts into dense vectors.

    Implement this class to plug in a custom embedding backend.  The pipeline
    accepts any ``EmbeddingBase`` instance via dependency injection.

    Example::

        class MyEmbedding(EmbeddingBase):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                ...

    The interface is async because production embedders typically make HTTP
    calls (OpenAI, Cohere, etc.).  CPU-bound local models should offload to a
    thread pool via ``asyncio.get_event_loop().run_in_executor``.

    Implementations must return one embedding per input text, preserving
    order.
    """

    @property
    def context_window(self) -> int:
        """Maximum number of tokens accepted in a single input text.

        Inputs longer than this are truncated or rejected by the backend, so
        callers (e.g. the chunk-embed-upsert step) size passages against it.
        The base returns a conservative :data:`DEFAULT_CONTEXT_WINDOW`;
        concrete embedders override with the configured window.
        """
        return DEFAULT_CONTEXT_WINDOW

    @property
    def dimensions(self) -> int | None:
        """Output vector dimensionality, when known without an embedding call.

        Returns the length of the vectors :meth:`embed` produces — derived from
        a configured override or the model itself — or ``None`` when it can only
        be determined by actually embedding (e.g. an API model whose dimension
        was left at the provider default).  The base implementation returns
        ``None``; concrete embedders override it when they can report the value.
        Callers that need it unconditionally can embed a probe and measure
        ``len`` of a result.
        """
        return None

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for *texts*.

        Args:
            texts: Texts to embed. May be empty; return ``[]`` in that case.

        Returns:
            One embedding vector per input text, in the same order.
        """
