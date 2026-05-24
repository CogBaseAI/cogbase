"""LLM builder from typed config.

Centralises backend selection so callers can stay config-driven without
importing concrete LLM implementations directly.
"""

from __future__ import annotations

from cogbase.config.models import LLMConfig
from cogbase.llms.base import LLMBase


def build_llm(cfg: LLMConfig) -> LLMBase:
    """Instantiate an LLM backend from config."""
    if cfg.provider in ("openai", "openai-compatible"):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        from cogbase.llms.openai import OpenAILLM
        client = openai.AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        return OpenAILLM(client, model=cfg.model, mini_model=cfg.mini_model)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")
