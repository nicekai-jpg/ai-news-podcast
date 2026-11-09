"""Stage 1 C4 网页通道：列表页解析（选择器/启发式）、多 URL 失败隔离、DeepSeek 解析。

全部用例使用假的 httpx 客户端 + 本地 HTML fixture，不联网。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from ai_news_podcast.business.collect import web
from ai_news_podcast.business.collect.base import (
    CollectError,
    ConfigError,
    FetchContext,
    SourceConfig,
)

NOW = datetime(2026, 11, 9, 16, 0, tzinfo=UTC)
PAGE_URL = "https://zhipuai.cn/zh/news"

LIST_HTML = """
<html><body>
  <ul class="news-list">
    <li class="item">
      <a href="/news/glm-5">智谱发布新一代模型 GLM-5</a>
      <span class="date">2026-11-08</span>
    </li>
    <li class="item">
      <a href="/news/agent">智谱开源 Agent 框架</a>
      <span class="date">2026-11-07</span>
    </li>
    <li class="nav"><a href="/">首页</a></li>
  </ul>
</body></html>
"""

HEURISTIC_HTML = """
<html><body><ul>
  <li><a href="/news/1">工信部发布人工智能产业发展新政策</a><span>2026-11-08</span></li>
  <li><a href="/about">关于本站的介绍页面</a></li>
</ul></body></html>
"""

URL_DATE_HTML = """
<html><body><ul>
  <li><a href="/2026-11/09/c_123.htm">网信办发布生成式人工智能管理新规</a></li>
</ul></body></html>
"""

DEEPSEEK_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://api-docs.deepseek.com/news/news261108</loc></url>
  <url><loc>https://api-docs.deepseek.com/news/news261001</loc></url>
  <url><loc>https://api-docs.deepseek.com/zh-cn/quick_start/pricing</loc></url>
</urlset>
"""

DEEPSEEK_PAGE = """<html><head>
<title>DeepSeek V4 发布 | DeepSeek API Docs</title>
</head><body><article><p>这是新闻正文。</p></article></body></html>
"""

DEEPSEEK_CHANGELOG = """
<html><body>
  <h2>2026-11-09 更新</h2>
  <h3>V4.1 API 更新</h3>
  <p>新增 JSON 输出格式。</p>
  <h2>2026-11-08 更新</h2>
  <h3>同日期新闻页已覆盖的条目</h3>
  <p>不应重复产出。</p>
</body></html>
"""


class _FakeResponse:
    def __init__(self, *, text: str = "", status_code: int = 200) -> None:
        self.content = text.encode("utf-8")
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes = routes or {}
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        value = self.routes.get(url)
        if value is None:
            raise httpx.ConnectError(f"no route: {url}")
        if isinstance(value, _FakeResponse):
            return value
        return _FakeResponse(text=str(value))


def _ctx(client: Any, **overrides: Any) -> FetchContext:
    params: dict[str, Any] = {
        "date_str": "2026-11-09",
        "window_hours": 72,
        "http_client": client,
        "now": NOW,
        "retry_wait": 0.0,
        "throttle_interval": 0.0,
        "fetch_fulltext": False,
    }
    params.update(overrides)
    return FetchContext(**params)


def _source(**overrides: Any) -> SourceConfig:
    params: dict[str, Any] = {
        "name": "zhipu_news",
        "label": "智谱新闻",
        "channel": "c4_web",
        "parser": "generic",
        "url": PAGE_URL,
        "category": "official",
        "limit": 10,
        "selectors": {"item": "li.item", "link": "a", "date": ".date"},
    }
    params.update(overrides)
    return SourceConfig(**params)


class TestParseHtmlList:
    def test_selector_based_parsing(self) -> None:
        entries = web.parse_html_list(
            LIST_HTML,
            page_url=PAGE_URL,
            selectors={"item": "li.item", "link": "a", "date": ".date"},
            now=NOW,
        )

        assert len(entries) == 2
        assert entries[0].title == "智谱发布新一代模型 GLM-5"
        assert entries[0].link == "https://zhipuai.cn/news/glm-5"
        assert entries[0].published_at == datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
        assert entries[1].published_at == datetime(2026, 11, 7, 12, 0, tzinfo=UTC)

    def test_heuristic_requires_neighbour_date(self) -> None:
        entries = web.parse_html_list(HEURISTIC_HTML, page_url=PAGE_URL, selectors={}, now=NOW)

        assert len(entries) == 1
        assert entries[0].title == "工信部发布人工智能产业发展新政策"
        assert entries[0].link == "https://zhipuai.cn/news/1"

    def test_date_falls_back_to_url(self) -> None:
        entries = web.parse_html_list(
            URL_DATE_HTML,
            page_url="https://www.cac.gov.cn/",
            selectors={},
            now=NOW,
        )

        assert len(entries) == 1
        assert entries[0].published_at == datetime(2026, 11, 9, 12, 0, tzinfo=UTC)

    def test_dedupes_links_and_strips_title_date(self) -> None:
        html = """
        <ul>
          <li><a href="/news/1">某公司发布新模型 2026-11-08</a></li>
          <li><a href="/news/1">某公司发布新模型</a></li>
        </ul>
        """
        entries = web.parse_html_list(
            html,
            page_url=PAGE_URL,
            selectors={"item": "li", "link": "a"},
            now=NOW,
        )

        assert len(entries) == 1
        assert entries[0].title == "某公司发布新模型"


class TestFetchGeneric:
    def test_fetches_and_builds_items(self) -> None:
        client = _FakeClient({PAGE_URL: LIST_HTML})
        items = web.fetch_web(_source(), _ctx(client))

        assert len(items) == 2
        assert items[0].channel == "c4_web"
        assert items[0].full_text_status == "none"
        assert items[0].link == "https://zhipuai.cn/news/glm-5"
        assert client.calls == [PAGE_URL]

    def test_fetches_article_full_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            web,
            "extract_article_text",
            lambda html, url="", **kwargs: "网页正文" * 60,
        )
        client = _FakeClient(
            {
                PAGE_URL: LIST_HTML,
                "https://zhipuai.cn/news/glm-5": "<html/>",
                "https://zhipuai.cn/news/agent": "<html/>",
            }
        )
        items = web.fetch_web(_source(), _ctx(client, fetch_fulltext=True))

        assert [item.full_text_status for item in items] == ["ok", "ok"]
        assert "网页正文" in items[0].full_text

    def test_failure_isolation_between_list_urls(self) -> None:
        good_url = "https://seed.bytedance.com/research"
        source = _source(url="", urls=["https://bad.example.com/list", good_url])
        client = _FakeClient({good_url: LIST_HTML})
        items = web.fetch_web(source, _ctx(client))

        assert len(items) == 2
        assert "https://bad.example.com/list" in client.calls

    def test_all_list_urls_failed_raises(self) -> None:
        source = _source(url="https://bad.example.com/list")
        with pytest.raises(CollectError):
            web.fetch_web(source, _ctx(_FakeClient()))

    def test_unknown_parser_raises(self) -> None:
        with pytest.raises(ConfigError):
            web.fetch_web(_source(parser="unknown"), _ctx(_FakeClient()))


class TestDeepSeekParsing:
    def test_parse_sitemap_news(self) -> None:
        entries = web.parse_sitemap_news(DEEPSEEK_SITEMAP)

        assert len(entries) == 2
        assert entries[0][0] == datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
        assert entries[0][1].endswith("/news/news261108")
        assert entries[1][1].endswith("/news/news261001")

    def test_parse_page_title_strips_site_suffix(self) -> None:
        assert web.parse_page_title(DEEPSEEK_PAGE) == "DeepSeek V4 发布"

    def test_parse_changelog(self) -> None:
        entries = web.parse_changelog(DEEPSEEK_CHANGELOG)

        assert len(entries) == 2
        assert entries[0][0] == datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
        assert entries[0][1] == "V4.1 API 更新"
        assert "JSON" in entries[0][2]

    def test_to_zh_url(self) -> None:
        assert (
            web.to_zh_url("https://api-docs.deepseek.com/news/news261108")
            == "https://api-docs.deepseek.com/zh-cn/news/news261108"
        )


class TestFetchDeepSeek:
    SITEMAP_URL = "https://api-docs.deepseek.com/sitemap.xml"
    CHANGELOG_URL = "https://api-docs.deepseek.com/zh-cn/updates"
    PAGE_ZH_URL = "https://api-docs.deepseek.com/zh-cn/news/news261108"

    def _deepseek_source(self) -> SourceConfig:
        return _source(
            name="deepseek_web",
            label="DeepSeek 官方新闻",
            parser="deepseek",
            url=self.SITEMAP_URL,
            selectors={},
            changelog_url=self.CHANGELOG_URL,
        )

    def test_fetches_news_and_changelog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            web,
            "extract_article_text",
            lambda html, url="", **kwargs: "DeepSeek 新闻正文" * 50,
        )
        client = _FakeClient(
            {
                self.SITEMAP_URL: DEEPSEEK_SITEMAP,
                self.PAGE_ZH_URL: DEEPSEEK_PAGE,
                self.CHANGELOG_URL: DEEPSEEK_CHANGELOG,
            }
        )
        items = web.fetch_web(self._deepseek_source(), _ctx(client))

        titles = {item.title for item in items}
        assert "DeepSeek V4 发布" in titles
        assert "V4.1 API 更新" in titles
        # 同日期（2026-11-08）已有 news 页，更新日志条目跳过
        assert len(items) == 2
        # 窗口外的 news261001 不抓取
        assert not any("261001" in url for url in client.calls)

    def test_skips_changelog_entry_when_news_same_day(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            web,
            "extract_article_text",
            lambda html, url="", **kwargs: "正文" * 100,
        )
        changelog_same_day = DEEPSEEK_CHANGELOG.replace("2026-11-09", "2026-11-08")
        client = _FakeClient(
            {
                self.SITEMAP_URL: DEEPSEEK_SITEMAP,
                self.PAGE_ZH_URL: DEEPSEEK_PAGE,
                self.CHANGELOG_URL: changelog_same_day,
            }
        )
        items = web.fetch_web(self._deepseek_source(), _ctx(client))

        assert [item.title for item in items] == ["DeepSeek V4 发布"]

    def test_sitemap_failure_raises(self) -> None:
        with pytest.raises(CollectError):
            web.fetch_web(self._deepseek_source(), _ctx(_FakeClient()))
