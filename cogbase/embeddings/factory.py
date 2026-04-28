"""Embedding builder from typed config.

Centralises backend selection so callers can stay config-driven without
importing concrete embedding implementations directly.
"""

from __future__ import annotations

import os

from cogbase.config.models import EmbeddingConfig
from cogbase.embeddings.base import EmbeddingBase


def build_embedding(cfg: EmbeddingConfig) -> EmbeddingBase:
    """Instantiate an embedder from config."""
    if cfg.provider == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        from cogbase.embeddings.openai import OpenAIEmbedding
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)
        kwargs = {}
        if cfg.dimensions is not None:
            kwargs["dimensions"] = cfg.dimensions
        return OpenAIEmbedding(client, model=cfg.model, **kwargs)
    if cfg.provider == "sentence-transformers":
        from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding(model_name=cfg.model)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
