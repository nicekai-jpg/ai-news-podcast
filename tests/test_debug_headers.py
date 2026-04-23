"""Tests for ai_news_podcast.cli.debug_headers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_news_podcast.cli.debug_headers import debug_raw_request


class TestDebugRawRequest:
    def test_success_with_proxy(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = '{"ok": true}'

        with patch("requests.post", return_value=mock_resp) as mock_post:
            debug_raw_request()

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["proxies"] == {"https": "http://proxy:8080"}

        captured = capsys.readouterr()
        assert "Status: 200" in captured.out
        assert "Content-Type: application/json" in captured.out
        assert '{"ok": true}' in captured.out

    def test_success_without_proxy(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("HTTPS_PROXY", raising=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.text = "{}"

        with patch("requests.post", return_value=mock_resp) as mock_post:
            debug_raw_request()

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["proxies"] is None

    def test_error_handling(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("HTTPS_PROXY", raising=False)

        with patch("requests.post", side_effect=Exception("Connection refused")):
            debug_raw_request()

        captured = capsys.readouterr()
        assert "Error: Connection refused" in captured.out
