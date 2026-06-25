"""Embedding builder from typed config.

Centralises backend selection so callers can stay config-driven without
importing concrete embedding implementations directly.
"""

from __future__ import annotations

from cogbase.config.models import EmbeddingConfig
from cogbase.embeddings.base import EmbeddingBase


def build_embedding(cfg: EmbeddingConfig) -> EmbeddingBase:
    """Instantiate an embedder from config."""
    if cfg.provider in ("openai", "openai-compatible"):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        from cogbase.embeddings.openai import OpenAIEmbedding
        client = openai.AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        return OpenAIEmbedding(
            client,
            model=cfg.model,
            dimensions=cfg.dimensions,
            batch_size=cfg.batch_size,
            context_window=cfg.context_window,
        )
    if cfg.provider == "sentence-transformers":
        from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
        # Only override the model's own max_seq_length when the window was set
        # explicitly; otherwise let the local model report its true limit.
        explicit_window = (
            cfg.context_window if "context_window" in cfg.model_fields_set else None
        )
        return SentenceTransformersEmbedding(
            model_name=cfg.model,
            context_window=explicit_window,
        )
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
