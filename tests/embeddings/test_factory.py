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
    assert embedder._dimensions == 1536



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



def test_build_embedding_compatible_passes_dimensions():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        embedder = build_embedding(_compatible_cfg(api_key="dummy", dimensions=512))
    assert embedder._dimensions == 512



# --- sentence-transformers ---

def test_build_embedding_sentence_transformers(monkeypatch):
    mock_st_module = MagicMock()
    mock_model_instance = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model_instance
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_module)

    from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
    cfg = EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2", api_key="EMPTY")
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
