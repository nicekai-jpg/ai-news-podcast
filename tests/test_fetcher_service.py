"""Tests for ai_news_podcast.business.pipeline.fetcher_service pure functions."""

from __future__ import annotations

import pytest

from ai_news_podcast.business.pipeline.fetcher_service import (
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


class TestContentEncoded:
    """wechat2rss 等源只有 content:encoded 有全文, summary 只有几十字。"""

    @pytest.mark.asyncio
    async def test_uses_content_encoded_without_extra_page_fetch(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from ai_news_podcast.business.pipeline import fetcher_service
        from ai_news_podcast.business.pipeline.fetcher_service import _process_single_entry

        full_body = "机器之心正文内容。" * 200  # 1400 字
        entry = SimpleNamespace(
            title="机器之心：某模型发布",
            link="https://mp.weixin.qq.com/s?__biz=xxx&mid=1&idx=1&sn=abc",
            summary="原创",
            content=[{"value": f"<p>{full_body}</p>"}],
            published_parsed=(2026, 11, 5, 8, 21, 0),
        )
        http_get = AsyncMock(side_effect=AssertionError("content:encoded 已含正文, 不应再抓页面"))

        with patch.object(fetcher_service, "_http_get", new=http_get):
            item = await _process_single_entry(
                entry,
                name="机器之心",
                src_category="news",
                client=None,
                throttle=None,
                max_pages=10,
                pages_counter=[0],
            )

        assert item is not None
        assert item.summary.startswith("机器之心正文内容")
        assert len(item.summary) <= 500
        assert len(item.full_text_snippet) > len(item.summary)
        assert "机器之心正文内容" in item.full_text_snippet
        assert http_get.await_count == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_page_fetch_without_content(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from ai_news_podcast.business.pipeline import fetcher_service
        from ai_news_podcast.business.pipeline.fetcher_service import _process_single_entry

        entry = SimpleNamespace(
            title="普通 RSS 源新闻",
            link="https://example.com/news/1",
            summary="",
            published_parsed=(2026, 11, 5, 8, 21, 0),
        )
        page = "<html><body><article><p>" + ("页面正文。" * 300) + "</p></article></body></html>"
        http_get = AsyncMock(return_value=SimpleNamespace(text=page))

        with patch.object(fetcher_service, "_http_get", new=http_get):
            item = await _process_single_entry(
                entry,
                name="Example",
                src_category="news",
                client=None,
                throttle=None,
                max_pages=10,
                pages_counter=[0],
            )

        assert item is not None
        assert item.full_text_snippet
        assert http_get.await_count == 1
