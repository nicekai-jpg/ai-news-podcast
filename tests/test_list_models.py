"""Tests for ai_news_podcast.cli.list_models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_news_podcast.cli.list_models import list_models


class TestListModels:
    def test_no_api_key(self, capsys) -> None:
        with patch.dict("os.environ", {}, clear=True):
            list_models()
        captured = capsys.readouterr()
        assert "Error: No API key found" in captured.out

    def test_lists_models_success(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        fake_model = MagicMock()
        fake_model.name = "models/gemini-pro"
        fake_model.supported_actions = ["generateContent"]

        fake_client = MagicMock()
        fake_client.models.list.return_value = [fake_model]

        mock_genai = MagicMock()
        mock_genai.Client = MagicMock(return_value=fake_client)

        # Patch the module-level genai imported by list_models.py
        with patch("ai_news_podcast.cli.list_models.genai", mock_genai):
            list_models()

        captured = capsys.readouterr()
        assert "Available models:" in captured.out
        assert "models/gemini-pro" in captured.out
        assert "generateContent" in captured.out

    def test_error_handling(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.models.list.side_effect = Exception("API Error")

        mock_genai = MagicMock()
        mock_genai.Client = MagicMock(return_value=fake_client)

        with patch("ai_news_podcast.cli.list_models.genai", mock_genai):
            list_models()

        captured = capsys.readouterr()
        assert "Failed to list models: API Error" in captured.out
