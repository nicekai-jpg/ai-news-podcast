"""Tests for ai_news_podcast.pipeline.fetcher pure functions."""

from __future__ import annotations

import pytest

from ai_news_podcast.pipeline.fetcher import (
    _detect_lang,
    _infer_category,
    _is_junk_summary,
    _item_id,
    normalize_url,
)


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://example.com/path/", "https://example.com/path"),
            ("http://example.com/path", "https://example.com/path"),
            ("https://EXAMPLE.COM/PATH", "https://example.com/PATH"),
            ("", ""),
            ("  ", ""),
            (
                "https://example.com?utm_source=xyz&utm_campaign=abc&id=1",
                "https://example.com?id=1",
            ),
            (
                "https://example.com?utm_medium=email&id=2&foo=bar",
                "https://example.com?id=2&foo=bar",
            ),
        ],
    )
    def test_cases(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected


class TestItemId:
    def test_deterministic(self) -> None:
        url = "https://example.com/article"
        assert _item_id(url) == _item_id(url)

    def test_different_urls_different_ids(self) -> None:
        assert _item_id("https://a.com") != _item_id("https://b.com")


class TestIsJunkSummary:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "photo of a cat",
            "An illustration of machine learning",
            "featured image for the blog post",
        ],
    )
    def test_junk(self, text: str) -> None:
        assert _is_junk_summary(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "This is a comprehensive article about AI advances in 2025, covering model training and deployment strategies.",
            "OpenAI releases GPT-5 with significant improvements.",
        ],
    )
    def test_not_junk(self, text: str) -> None:
        assert _is_junk_summary(text) is False


class TestDetectLang:
    def test_chinese(self) -> None:
        assert _detect_lang("这是一篇关于人工智能的文章") == "zh"

    def test_english(self) -> None:
        assert _detect_lang("This is an article about AI.") == "en"

    def test_mixed_prefers_zh(self) -> None:
        assert _detect_lang("AI 人工智能 future") == "zh"

    def test_empty(self) -> None:
        assert _detect_lang("") == "en"


class TestInferCategory:
    def test_model(self) -> None:
        assert _infer_category("New model released", "benchmark results") == "model"

    def test_product(self) -> None:
        assert _infer_category("Launch of new API", "") == "product"

    def test_research(self) -> None:
        assert _infer_category("Paper on arxiv", "research method") == "research"

    def test_open_source(self) -> None:
        assert _infer_category("GitHub repo open sourced", "") == "open_source"

    def test_policy(self) -> None:
        assert _infer_category("New regulation on AI safety", "policy") == "policy"

    def test_tool(self) -> None:
        assert _infer_category("New framework and SDK", "tool plugin") == "tool"

    def test_other(self) -> None:
        assert _infer_category("Random thoughts", "") == "other"
