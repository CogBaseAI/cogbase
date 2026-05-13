"""Tests for cogbase.llms.factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogbase.config.models import LLMConfig
from cogbase.llms.factory import build_llm
from cogbase.llms.openai import OpenAILLM


def _make_openai_cfg(**kwargs) -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o", **kwargs)


def test_build_llm_openai_returns_openai_llm():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        llm = build_llm(_make_openai_cfg(api_key="sk-test"))
    assert isinstance(llm, OpenAILLM)


def test_build_llm_openai_passes_model():
    mock_client = MagicMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        llm = build_llm(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test"))
    assert llm._model == "gpt-4o-mini"


def test_build_llm_openai_uses_config_api_key():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(_make_openai_cfg(api_key="sk-explicit"))
    mock_cls.assert_called_once_with(api_key="sk-explicit")


def test_build_llm_openai_falls_back_to_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(LLMConfig(provider="openai", model="gpt-4o"))
    mock_cls.assert_called_once_with(api_key="sk-from-env")


def test_build_llm_unknown_provider_raises():
    cfg = SimpleNamespace(provider="anthropic", model="claude-3", resolved_api_key=lambda: None)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(cfg)


def test_build_llm_missing_openai_package_raises(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai package required"):
        build_llm(_make_openai_cfg(api_key="sk-test"))


def test_build_llm_passes_mini_model():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        llm = build_llm(LLMConfig(provider="openai", model="gpt-4o", mini_model="gpt-4o-mini", api_key="sk-test"))
    assert llm._mini_model == "gpt-4o-mini"


def test_build_llm_mini_model_none_when_not_configured():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        llm = build_llm(_make_openai_cfg(api_key="sk-test"))
    assert llm._mini_model is None
