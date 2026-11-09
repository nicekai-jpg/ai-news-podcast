"""Tests for dedup_service — cross-episode URL / semantic deduplication.

重点覆盖两个真实回归：
- 正文链接以 ``.html`` 结尾时不能被 URL 去重整体跳过(量子位等站点全部如此)；
- 正文里 ``DeepSeek Harness 0.2.1`` 之类的噪声不能把「同一篇文章」误判成「新版本」。
"""

from __future__ import annotations

import types
from pathlib import Path

from ai_news_podcast.business.pipeline.dedup_service import (
    HistoricalRecord,
    _are_distinct_model_versions,
    _extract_model_versions,
    get_recent_broadcasted_urls,
    semantic_dedup,
)
from ai_news_podcast.data.utils_dao import write_json


class TestBroadcastedUrls:
    def test_keeps_html_article_links(self, tmp_path: Path) -> None:
        path = tmp_path / "episodes.json"
        write_json(
            path,
            [
                {
                    "id": "2026-10-19",
                    "description": (
                        '<ol><li><a href="https://www.qbitai.com/2026/10/504067.html">'
                        "DeepSeek 新模型</a></li></ol>"
                    ),
                    "link": "https://example.github.io/episodes/2026-10-19.html",
                }
            ],
        )

        urls = get_recent_broadcasted_urls(path, limit=14)

        # 正文 .html 链接必须被收集, 否则量子位等站点的文章永远无法按 URL 去重。
        assert "https://www.qbitai.com/2026/10/504067.html" in urls

    def test_excludes_site_assets_and_episode_page(self, tmp_path: Path) -> None:
        path = tmp_path / "episodes.json"
        write_json(
            path,
            [
                {
                    "id": "2026-10-19",
                    "description": (
                        '<p><a href="https://example.github.io/feed.xml">rss</a>'
                        '<a href="https://example.github.io/episodes/2026-10-19.html">ep</a>'
                        '<a href="https://example.github.io/episodes/2026-10-19.mp3">mp3</a>'
                        '<a href="https://example.com/news/story.html">story</a></p>'
                    ),
                }
            ],
        )

        urls = get_recent_broadcasted_urls(path, limit=14)

        assert "https://example.github.io/feed.xml" not in urls
        assert "https://example.github.io/episodes/2026-10-19.html" not in urls
        assert "https://example.github.io/episodes/2026-10-19.mp3" not in urls
        assert "https://example.com/news/story.html" in urls


class TestModelVersionGuard:
    def test_noise_decimal_is_not_a_model_version(self) -> None:
        # "0.2.1" 紧跟在 "DeepSeek Harness" 后面, 不能被拆出 "2.1" 当成 DeepSeek 版本。
        versions = _extract_model_versions(
            "DeepSeek Harness全新0.2.1版本 也迎来增强, 需升级到0.2.1-rc.1及以上"
        )
        assert ("deepseek", "2.1") not in versions
        assert versions == set()

    def test_same_release_is_not_a_distinct_version(self) -> None:
        new_text = (
            "突发！DeepSeek新模型发布，超级性价比斩杀一切 "
            "DeepSeek-V4.1-Flash-1017，还开放了自研搜索API "
            "DeepSeek Harness全新0.2.1版本也迎来增强"
        )
        hist_text = (
            "突发！DeepSeek新模型发布，超级性价比斩杀一切 "
            "DeepSeek-V4.1-Flash-1017，还开放了自研搜索API"
        )

        assert _extract_model_versions(new_text) == {("deepseek", "4.1")}
        assert _are_distinct_model_versions(new_text, hist_text) is False

    def test_new_version_is_kept(self) -> None:
        assert (
            _are_distinct_model_versions(
                "Introducing Gemini 3.9 Flash", "Introducing Gemini 3.8 Flash"
            )
            is True
        )
        assert _are_distinct_model_versions("Qwen3.9 released", "Qwen3.8 released") is True

    def test_hyphenated_versions_are_compared(self) -> None:
        # claude-3-5-sonnet / claude-3-7-sonnet 这类连字符版本号不能都退化成 "3"。
        assert _extract_model_versions("claude-3-5-sonnet") == {("claude", "3.5")}
        assert _are_distinct_model_versions("claude-3-7-sonnet", "claude-3-5-sonnet") is True
        assert (
            _are_distinct_model_versions(
                "claude-3-5-sonnet", "Anthropic releases Claude 3.5 Sonnet"
            )
            is False
        )

    def test_unrelated_family_is_not_distinct(self) -> None:
        # 历史素材完全没提该家族的版本 → 不构成「新版本事件」, 按重复处理, 防止漏去重。
        assert (
            _are_distinct_model_versions(
                "DeepSeek V4.1 release notes", "Some other news without model names"
            )
            is False
        )


class TestSemanticDedup:
    def _item(self, title: str, summary: str, snippet: str) -> object:
        return types.SimpleNamespace(
            title=title,
            summary=summary,
            full_text_snippet=snippet,
            source_name="量子位",
            link="https://www.qbitai.com/2026/10/504067.html",
        )

    def test_repeat_article_filtered_despite_version_noise(self) -> None:
        title = "突发！DeepSeek新模型发布，超级性价比斩杀一切"
        summary = "新版本V4.1-Flash-1017，还开放了自研搜索API"
        new_item = self._item(
            title,
            summary,
            "DeepSeek Harness全新0.2.1版本也迎来增强 新版本V4.1-Flash-1017",
        )
        records = [
            HistoricalRecord(
                text=f"{title} {summary} DeepSeek 模型 1017 V4.1 Flash",
                story_title=title,
                episode_id="2026-10-19",
            )
        ]

        kept, details = semantic_dedup([new_item], records, {"semantic_sim_threshold": 0.2})

        assert kept == []
        assert len(details) == 1
        assert details[0]["matched_episode_id"] == "2026-10-19"
        assert details[0]["reason"] == "cross_episode_semantic"
