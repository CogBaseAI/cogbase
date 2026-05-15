"""Tests for cogbase.llms.factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogbase.config.models import LLMConfig
from cogbase.llms.factory import build_llm
from cogbase.llms.openai import OpenAILLM


def _make_openai_cfg(**kwargs) -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o", **kwargs)


def _make_compatible_cfg(**kwargs) -> LLMConfig:
    return LLMConfig(provider="openai-compatible", model="qwen-max", base_url="http://localhost:8000/v1", **kwargs)


# --- openai provider ---

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


def test_build_llm_openai_no_base_url_by_default():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(_make_openai_cfg(api_key="sk-test"))
    _, kwargs = mock_cls.call_args
    assert "base_url" not in kwargs


def test_build_llm_passes_mini_model():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        llm = build_llm(LLMConfig(provider="openai", model="gpt-4o", mini_model="gpt-4o-mini", api_key="sk-test"))
    assert llm._mini_model == "gpt-4o-mini"


def test_build_llm_mini_model_none_when_not_configured():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        llm = build_llm(_make_openai_cfg(api_key="sk-test"))
    assert llm._mini_model is None


# --- openai-compatible provider ---

def test_build_llm_compatible_returns_openai_llm():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        llm = build_llm(_make_compatible_cfg(api_key="dummy"))
    assert isinstance(llm, OpenAILLM)


def test_build_llm_compatible_passes_base_url():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(_make_compatible_cfg(api_key="dummy"))
    mock_cls.assert_called_once_with(api_key="dummy", base_url="http://localhost:8000/v1")


def test_build_llm_compatible_uses_api_key_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(LLMConfig(
            provider="openai-compatible",
            model="qwen-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
        ))
    mock_cls.assert_called_once_with(
        api_key="sk-dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def test_build_llm_api_key_takes_priority_over_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY_ENV", "from-env")
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(_make_compatible_cfg(api_key="explicit-key", api_key_env="MY_KEY_ENV"))
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["api_key"] == "explicit-key"


def test_build_llm_compatible_requires_base_url():
    with pytest.raises(ValueError, match="base_url is required"):
        LLMConfig(provider="openai-compatible", model="qwen-max")


# --- shared ---

def test_build_llm_unknown_provider_raises():
    cfg = SimpleNamespace(provider="anthropic", model="claude-3", resolved_api_key=lambda: None)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(cfg)


def test_build_llm_missing_openai_package_raises(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai package required"):
        build_llm(_make_openai_cfg(api_key="sk-test"))
