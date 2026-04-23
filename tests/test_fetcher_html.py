"""Tests for fetcher HTML stripping and keyword extraction edge cases."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from ai_news_podcast.pipeline.fetcher import _strip_html
from ai_news_podcast.pipeline.processor import _extract_keywords


class TestStripHtml:
    def test_plain_text_no_html(self) -> None:
        assert _strip_html("Hello   world") == "Hello world"

    def test_empty_string(self) -> None:
        assert _strip_html("") == ""

    def test_with_html_tags(self) -> None:
        mock_bs4 = MagicMock()
        soup = MagicMock()
        soup.get_text = MagicMock(return_value="Hello world")
        mock_bs4.BeautifulSoup = MagicMock(return_value=soup)

        with patch.dict(sys.modules, {"bs4": mock_bs4}):
            result = _strip_html("<p>Hello</p>")
        assert result == "Hello world"

    def test_with_ampersand(self) -> None:
        mock_bs4 = MagicMock()
        soup = MagicMock()
        soup.get_text = MagicMock(return_value="A & B")
        mock_bs4.BeautifulSoup = MagicMock(return_value=soup)

        with patch.dict(sys.modules, {"bs4": mock_bs4}):
            result = _strip_html("A &amp; B")
        assert result == "A & B"


class TestExtractKeywords:
    def test_jieba_analyse_success(self) -> None:
        mock_jieba = MagicMock()
        mock_jieba.analyse.extract_tags.return_value = ["AI", "模型"]
        sys.modules["jieba"] = mock_jieba
        sys.modules["jieba.analyse"] = mock_jieba.analyse

        try:
            result = _extract_keywords("AI大模型")
            assert result == {"AI", "模型"}
        finally:
            del sys.modules["jieba"]
            del sys.modules["jieba.analyse"]

    def test_jieba_analyse_fallback(self) -> None:
        mock_jieba = MagicMock()
        mock_jieba.analyse.extract_tags.side_effect = Exception("fail")
        mock_jieba.cut.return_value = ["AI", "大", "模型"]
        sys.modules["jieba"] = mock_jieba
        sys.modules["jieba.analyse"] = mock_jieba.analyse

        try:
            result = _extract_keywords("AI大模型")
            assert "AI" in result
            assert "模型" in result
            assert "大" not in result  # filtered by length >= 2
        finally:
            del sys.modules["jieba"]
            del sys.modules["jieba.analyse"]
