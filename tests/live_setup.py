"""Shared helpers for live integration tests.

Loads LLM and embedding configuration from .env.yaml (the same config that
api/main.py uses at startup).  Falls back to OpenAI defaults when .env.yaml
is absent or the relevant section is unset and OPENAI_API_KEY is in the
environment.

Typical usage in a live test module::

    from tests.live_setup import make_llm, make_embedding

    _llm = make_llm()
    _embedder = make_embedding()

    pytestmark = [
        pytest.mark.live,
        pytest.mark.skipif(_llm is None, reason="No LLM configured"),
    ]

    @pytest.fixture(scope="module")
    def llm():
        return _llm
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_YAML = Path(__file__).resolve().parent.parent / ".env.yaml"


def _load_system_config():
    from api.system_config import SystemConfig
    return SystemConfig.load(path=str(_ENV_YAML)) if _ENV_YAML.exists() else None


# Loaded once at import time so all callers share the same config.
_system_cfg = _load_system_config()


def make_llm():
    """Return a live LLM instance from .env.yaml, or None if not configured."""
    from cogbase.llms.factory import build_llm
    if _system_cfg is None or _system_cfg.llm is None:
        return None
    llm = build_llm(_system_cfg.llm)

    from cogbase.llms.openai import OpenAILLM
    if isinstance(llm, OpenAILLM):
        llm.enable_flex_tier()
    return llm


def make_embedding(*, dimensions: int | None = None):
    """Return a live embedder instance from .env.yaml, or None if not configured.

    Pass ``dimensions`` to override the vector size (e.g. for truncation tests).
    """
    from cogbase.embeddings.factory import build_embedding
    if _system_cfg is None or _system_cfg.embedding is None:
        return None
    cfg = _system_cfg.embedding
    if dimensions is not None:
        cfg = cfg.model_copy(update={"dimensions": dimensions})
    return build_embedding(cfg)
