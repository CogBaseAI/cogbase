"""Tests for cogbase.embeddings.factory."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogbase.config.models import EmbeddingConfig
from cogbase.embeddings.factory import build_embedding
from cogbase.embeddings.openai import OpenAIEmbedding


def _openai_cfg(**kwargs) -> EmbeddingConfig:
    return EmbeddingConfig(provider="openai", model="text-embedding-3-small", **kwargs)


def _compatible_cfg(**kwargs) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="openai-compatible",
        model="text-embedding-v3",
        base_url="http://localhost:8000/v1",
        **kwargs,
    )


# --- openai provider ---

def test_build_embedding_openai_returns_openai_embedding():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        embedder = build_embedding(_openai_cfg(api_key="sk-test"))
    assert isinstance(embedder, OpenAIEmbedding)


def test_build_embedding_openai_passes_model():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        embedder = build_embedding(EmbeddingConfig(
            provider="openai", model="text-embedding-3-large", api_key="sk-test"
        ))
    assert embedder._model == "text-embedding-3-large"


def test_build_embedding_openai_passes_dimensions():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        embedder = build_embedding(_openai_cfg(api_key="sk-test", dimensions=256))
    assert embedder._dimensions == 256


def test_build_embedding_openai_no_dimensions_by_default():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        embedder = build_embedding(_openai_cfg(api_key="sk-test"))
    assert embedder._dimensions is None


def test_build_embedding_openai_uses_config_api_key():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(_openai_cfg(api_key="sk-explicit"))
    mock_cls.assert_called_once_with(api_key="sk-explicit")


def test_build_embedding_openai_falls_back_to_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(EmbeddingConfig(provider="openai", model="text-embedding-3-small"))
    mock_cls.assert_called_once_with(api_key="sk-from-env")


def test_build_embedding_openai_no_base_url_by_default():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(_openai_cfg(api_key="sk-test"))
    _, kwargs = mock_cls.call_args
    assert "base_url" not in kwargs


# --- openai-compatible provider ---

def test_build_embedding_compatible_returns_openai_embedding():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        embedder = build_embedding(_compatible_cfg(api_key="dummy"))
    assert isinstance(embedder, OpenAIEmbedding)


def test_build_embedding_compatible_passes_base_url():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(_compatible_cfg(api_key="dummy"))
    mock_cls.assert_called_once_with(api_key="dummy", base_url="http://localhost:8000/v1")


def test_build_embedding_compatible_uses_api_key_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(EmbeddingConfig(
            provider="openai-compatible",
            model="text-embedding-v3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
        ))
    mock_cls.assert_called_once_with(
        api_key="sk-dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def test_build_embedding_compatible_api_key_takes_priority_over_env(monkeypatch):
    monkeypatch.setenv("MY_KEY_ENV", "from-env")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_embedding(_compatible_cfg(api_key="explicit-key", api_key_env="MY_KEY_ENV"))
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["api_key"] == "explicit-key"


def test_build_embedding_compatible_passes_dimensions():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        embedder = build_embedding(_compatible_cfg(api_key="dummy", dimensions=512))
    assert embedder._dimensions == 512


def test_build_embedding_compatible_requires_base_url():
    with pytest.raises(ValueError, match="base_url is required"):
        EmbeddingConfig(provider="openai-compatible", model="text-embedding-v3")


# --- sentence-transformers ---

def test_build_embedding_sentence_transformers(monkeypatch):
    mock_st_module = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model_instance
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_module)

    from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
    cfg = EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2")
    embedder = build_embedding(cfg)
    assert isinstance(embedder, SentenceTransformersEmbedding)


# --- shared ---

def test_build_embedding_unknown_provider_raises():
    cfg = SimpleNamespace(provider="cohere", model="embed-v3", api_key=None, dimensions=None)
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        build_embedding(cfg)


def test_build_embedding_missing_openai_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai package required"):
        build_embedding(_openai_cfg(api_key="sk-test"))
