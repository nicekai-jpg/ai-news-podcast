"""Tests for scriptwriter _call_llm with mocked openai client."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from ai_news_podcast.pipeline.scriptwriter import _call_llm


class FakeCompletion:
    def __init__(self, content: str | None):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class FakeChatCompletions:
    def __init__(self, content: str | None):
        self._content = content

    def create(self, **kwargs: Any) -> FakeCompletion:
        return FakeCompletion(self._content)


class FakeChat:
    def __init__(self, content: str | None):
        self.completions = FakeChatCompletions(content)


class FakeOpenAI:
    def __init__(self, *, api_key: str, base_url: str | None, timeout: int):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = FakeChat("Generated script content")


class FakeOpenAIEmpty:
    def __init__(self, **kwargs: Any):
        self.chat = FakeChat(None)


class FakeOpenAIFailing:
    def __init__(self, **kwargs: Any):
        pass

    @property
    def chat(self):
        raise RuntimeError("API down")


class TestCallLlm:
    def test_success(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI = FakeOpenAI

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            result = _call_llm("prompt", {"api_key_env": "FAKE_KEY", "model": "test"})
        assert result == "Generated script content"

    def test_empty_content_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI = FakeOpenAIEmpty

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            result = _call_llm("prompt", {"api_key_env": "FAKE_KEY", "model": "test"})
        assert result is None

    def test_retries_then_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI = FakeOpenAIFailing

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            with patch("time.sleep"):  # speed up retries
                result = _call_llm("prompt", {"api_key_env": "FAKE_KEY", "model": "test"})
        assert result is None

    def test_missing_openai_import_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        with patch.dict("sys.modules", {"openai": None}):
            result = _call_llm("prompt", {"api_key_env": "FAKE_KEY"})
        assert result is None

    def test_uses_base_url_and_model(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        captured: dict[str, Any] = {}

        class CapturingOpenAI(FakeOpenAI):
            def __init__(self, *, api_key: str, base_url: str | None, timeout: int):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                captured["timeout"] = timeout
                self.chat = FakeChat("ok")

        fake_openai_module = MagicMock()
        fake_openai_module.OpenAI = CapturingOpenAI

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            _call_llm(
                "prompt",
                {
                    "api_key_env": "FAKE_KEY",
                    "model": "custom-model",
                    "base_url": "https://custom.api/v1",
                    "timeout": 30,
                },
            )
        assert captured["base_url"] == "https://custom.api/v1"
        assert captured["timeout"] == 30
