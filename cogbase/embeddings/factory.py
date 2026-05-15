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
        client_kwargs: dict = {"api_key": cfg.resolved_api_key()}
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        client = openai.AsyncOpenAI(**client_kwargs)
        embed_kwargs = {}
        if cfg.dimensions is not None:
            embed_kwargs["dimensions"] = cfg.dimensions
        return OpenAIEmbedding(client, model=cfg.model, **embed_kwargs)
    if cfg.provider == "sentence-transformers":
        from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding(model_name=cfg.model)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
