"""Tests for the DeepSeek official-news fetcher (sitemap discovery, no RSS)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ai_news_podcast.business.pipeline import fetcher_service
from ai_news_podcast.business.pipeline.fetcher_service import _fetch_deepseek_news

_SRC = {
    "name": "DeepSeek 官方新闻",
    "url": "https://api-docs.deepseek.com/sitemap.xml",
    "category": "official",
    "fetcher_type": "deepseek_news",
}


def _slug(dt: datetime) -> str:
    return f"news{dt:%y%m%d}"


def _sitemap(*dates: datetime) -> str:
    locs = "".join(
        f"<url><loc>https://api-docs.deepseek.com/news/{_slug(d)}</loc></url>" for d in dates
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><urlset>'
        "<url><loc>https://api-docs.deepseek.com/quick_start/pricing</loc></url>"
        f"{locs}</urlset>"
    )


def _page(title: str, body: str) -> str:
    return (
        f"<html><head><title>{title} | DeepSeek API Docs</title></head>"
        f"<body><article><p>{body}</p></article></body></html>"
    )


def _zh_url(d: datetime) -> str:
    return f"https://api-docs.deepseek.com/zh-cn/news/{_slug(d)}"


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


@pytest.mark.asyncio
async def test_fetches_recent_official_news_in_chinese() -> None:
    recent = datetime.now(tz=UTC) - timedelta(days=1)
    old = datetime.now(tz=UTC) - timedelta(days=400)
    pages = {
        _zh_url(recent): _page(
            "DeepSeek V4.1 Flash 新版本：许愿式编程一键成真",
            "DeepSeek V4.1 Flash 现已更新至 V4.1-Flash-1017 版本，官方网页端、APP 与 API 模型同步变更。",
        )
    }

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(recent, old))
        if url in pages:
            return _resp(pages[url])
        raise AssertionError(f"unexpected url: {url}")

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(_SRC, client=None, throttle=None, max_items=30)

    assert len(items) == 1
    item = items[0]
    assert item.source_name == "DeepSeek 官方新闻"
    assert item.source_category == "official"
    assert item.language == "zh"
    assert item.title == "DeepSeek V4.1 Flash 新版本：许愿式编程一键成真"
    assert item.link == _zh_url(recent)
    # 发布日期来自 newsYYMMDD slug, 按 UTC 12:00 近似(北京时间 20:00)。
    assert (
        item.published_at == recent.replace(hour=12, minute=0, second=0, microsecond=0).isoformat()
    )
    assert "V4.1-Flash-1017" in item.full_text_snippet
    assert item.summary


@pytest.mark.asyncio
async def test_falls_back_to_english_page_when_zh_missing() -> None:
    recent = datetime.now(tz=UTC) - timedelta(days=1)
    en_url = f"https://api-docs.deepseek.com/news/{_slug(recent)}"
    zh_url = _zh_url(recent)

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(recent))
        if url == zh_url:
            raise httpx.ConnectError("zh page unavailable")
        if url == en_url:
            return _resp(_page("DeepSeek-V4.1-Flash-1017 Release", "Wish-driven programming."))
        raise AssertionError(f"unexpected url: {url}")

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(_SRC, client=None, throttle=None, max_items=30)

    assert len(items) == 1
    assert items[0].title == "DeepSeek-V4.1-Flash-1017 Release"
    assert items[0].link == en_url


@pytest.mark.asyncio
async def test_sitemap_failure_returns_empty() -> None:
    async def fake_http_get(client, url, throttle):
        raise httpx.ConnectError("sitemap unavailable")

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(_SRC, client=None, throttle=None, max_items=30)

    assert items == []


@pytest.mark.asyncio
async def test_quiet_period_still_fetches_latest_entry() -> None:
    oldest = datetime.now(tz=UTC) - timedelta(days=300)
    newest = datetime.now(tz=UTC) - timedelta(days=200)
    pages = {
        _zh_url(newest): _page("DeepSeek V4 预览版发布", "新版本预览。"),
        _zh_url(oldest): _page("DeepSeek V3.2 正式版发布", "正式版。"),
    }

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(oldest, newest))
        return _resp(pages[url])

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(_SRC, client=None, throttle=None, max_items=30)

    assert len(items) == 1
    assert items[0].title == "DeepSeek V4 预览版发布"


@pytest.mark.asyncio
async def test_max_items_limits_detail_fetches() -> None:
    now = datetime.now(tz=UTC)
    dates = [now - timedelta(days=i) for i in range(1, 4)]
    pages = {_zh_url(d): _page(f"新闻 {i}", "正文。") for i, d in enumerate(dates)}
    requested: list[str] = []

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(*dates))
        requested.append(url)
        return _resp(pages[url])

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(_SRC, client=None, throttle=None, max_items=2)

    assert len(items) == 2
    assert requested == [_zh_url(dates[0]), _zh_url(dates[1])]


_CHANGELOG_URL = "https://api-docs.deepseek.com/zh-cn/updates"
_SRC_WITH_CHANGELOG = {**_SRC, "changelog_url": _CHANGELOG_URL}


def _changelog(*sections: tuple[str, list[tuple[str, str]]]) -> str:
    parts: list[str] = []
    for date, entries in sections:
        parts.append(f"<h2>时间: {date}</h2>")
        for title, body in entries:
            parts.append(f"<h3>{title}</h3><p>{body}</p>")
    return "<html><body>" + "".join(parts) + "</body></html>"


@pytest.mark.asyncio
async def test_changelog_entries_captured_with_unique_links() -> None:
    """稳定地址的更新日志: 每个条目要生成唯一链接, 否则跨期 URL 去重会误杀新条目。"""
    news_day = datetime.now(tz=UTC) - timedelta(days=2)
    changelog_day = datetime.now(tz=UTC) - timedelta(days=1)
    pages = {_zh_url(news_day): _page("DeepSeek-V4.1-Flash 发布", "模型发布。")}
    changelog = _changelog(
        (
            f"{changelog_day:%Y-%m-%d}",
            [
                ("开放自研搜索 API", "搜索 API 正式向公众开放。"),
                ("DeepSeek-V4.1-Flash-1017 更新", "新版本模型上线。"),
            ],
        )
    )

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(news_day))
        if url == _CHANGELOG_URL:
            return _resp(changelog)
        if url in pages:
            return _resp(pages[url])
        raise AssertionError(f"unexpected url: {url}")

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(
            _SRC_WITH_CHANGELOG, client=None, throttle=None, max_items=30
        )

    assert len(items) == 3
    news_items = [item for item in items if "updates" not in item.link]
    changelog_items = [item for item in items if "updates" in item.link]
    assert len(news_items) == 1
    assert len(changelog_items) == 2
    assert {item.title for item in changelog_items} == {
        "开放自研搜索 API",
        "DeepSeek-V4.1-Flash-1017 更新",
    }
    assert len({item.normalized_link for item in changelog_items}) == 2
    assert all("?date=" in item.link and "entry=" in item.link for item in changelog_items)
    expected_published = changelog_day.replace(
        hour=12, minute=0, second=0, microsecond=0
    ).isoformat()
    assert all(item.published_at == expected_published for item in changelog_items)


@pytest.mark.asyncio
async def test_changelog_skips_dates_already_covered_by_news_page() -> None:
    """同一天既有 news 页又有更新日志时, 只保留信息更全的 news 页, 避免重复播报。"""
    day = datetime.now(tz=UTC) - timedelta(days=1)
    pages = {_zh_url(day): _page("DeepSeek-V4.1-Flash 发布", "模型发布。")}
    changelog = _changelog((f"{day:%Y-%m-%d}", [("同一天的更新日志条目", "内容。")]))

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(day))
        if url == _CHANGELOG_URL:
            return _resp(changelog)
        return _resp(pages[url])

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(
            _SRC_WITH_CHANGELOG, client=None, throttle=None, max_items=30
        )

    assert len(items) == 1
    assert "updates" not in items[0].link


@pytest.mark.asyncio
async def test_old_changelog_entries_are_skipped() -> None:
    news_day = datetime.now(tz=UTC) - timedelta(days=2)
    old_day = datetime.now(tz=UTC) - timedelta(days=60)
    pages = {_zh_url(news_day): _page("DeepSeek-V4.1-Flash 发布", "模型发布。")}
    changelog = _changelog((f"{old_day:%Y-%m-%d}", [("很久以前的更新", "旧内容。")]))

    async def fake_http_get(client, url, throttle):
        if url.endswith("sitemap.xml"):
            return _resp(_sitemap(news_day))
        if url == _CHANGELOG_URL:
            return _resp(changelog)
        return _resp(pages[url])

    with patch.object(fetcher_service, "_http_get", new=AsyncMock(side_effect=fake_http_get)):
        items = await _fetch_deepseek_news(
            _SRC_WITH_CHANGELOG, client=None, throttle=None, max_items=30
        )

    assert len(items) == 1
    assert items[0].title == "DeepSeek-V4.1-Flash 发布"
