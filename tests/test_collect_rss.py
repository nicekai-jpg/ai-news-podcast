"""Stage 1 采集：RSS/Atom 解析、正文优先级、URL 去重与时效/条数收尾。

全部用例使用假的 httpx 客户端，不联网。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from ai_news_podcast.business.collect import rss
from ai_news_podcast.business.collect.base import (
    CollectError,
    FetchContext,
    ParseError,
    SourceConfig,
    build_item,
    detect_lang,
    finalize_items,
    strip_html,
)
from ai_news_podcast.business.contracts import canonical_link

NOW = datetime(2026, 11, 9, 16, 0, tzinfo=UTC)
FEED_URL = "https://feed.example.com/rss"
ARTICLE_URL = "https://example.com/posts/1"

RSS_WITH_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>示例 Feed</title>
    <link>https://example.com</link>
    <item>
      <title>某模型发布</title>
      <link>https://example.com/a?utm_source=newsletter</link>
      <description>短摘要</description>
      <content:encoded><![CDATA[<p>这里是完整正文，长度足够。</p>]]></content:encoded>
      <pubDate>Sun, 09 Nov 2026 08:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

RSS_NO_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>示例 Feed</title>
    <item>
      <title>普通 RSS 新闻</title>
      <link>https://example.com/posts/1</link>
      <description>只有很短的一段摘要</description>
      <pubDate>Sun, 09 Nov 2026 07:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

RSS_LONG_SUMMARY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>长摘要 Feed</title>
    <item>
      <title>只有长摘要的新闻</title>
      <link>https://example.com/posts/long</link>
      <description>{body}</description>
      <pubDate>Sun, 09 Nov 2026 06:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
""".format(body="这是一段足够长的摘要，可以直接当作正文使用。" * 60)

RSS_NO_DATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>无日期 Feed</title>
    <item>
      <title>没有发布时间的条目</title>
      <link>https://example.com/no-date</link>
      <description>摘要</description>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>OpenAI</title>
  <entry>
    <title>Atom 文章</title>
    <link rel="alternate" href="https://example.org/posts/1"/>
    <updated>2026-11-09T06:30:00Z</updated>
    <content type="html">&lt;p&gt;Atom 正文内容&lt;/p&gt;</content>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, *, content: bytes = b"", text: str = "", status_code: int = 200) -> None:
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


class _FakeClient:
    """按 URL 返回预置响应；未声明的 URL 抛 ConnectError（模拟网络失败）。"""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes = routes or {}
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        value = self.routes.get(url)
        if value is None:
            raise httpx.ConnectError(f"no route: {url}")
        if isinstance(value, Exception):
            raise value
        if isinstance(value, _FakeResponse):
            return value
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        return _FakeResponse(content=raw, text=raw.decode("utf-8", errors="replace"))


def _ctx(client: Any, **overrides: Any) -> FetchContext:
    params: dict[str, Any] = {
        "date_str": "2026-11-09",
        "window_hours": 72,
        "http_client": client,
        "now": NOW,
        "retry_wait": 0.0,
        "throttle_interval": 0.0,
    }
    params.update(overrides)
    return FetchContext(**params)


def _source(**overrides: Any) -> SourceConfig:
    params: dict[str, Any] = {
        "name": "test_feed",
        "label": "Test Feed",
        "channel": "c1_rss",
        "url": FEED_URL,
        "category": "news",
        "limit": 10,
    }
    params.update(overrides)
    return SourceConfig(**params)


class TestParseFeed:
    def test_rss_content_encoded_is_captured(self) -> None:
        entries = rss.parse_feed(RSS_WITH_CONTENT.encode("utf-8"))
        assert len(entries) == 1
        entry = entries[0]
        assert entry.title == "某模型发布"
        assert entry.link == "https://example.com/a?utm_source=newsletter"
        assert "完整正文" in entry.content_html
        assert entry.published_at == datetime(2026, 11, 9, 8, 0, tzinfo=UTC)

    def test_atom_entry(self) -> None:
        entries = rss.parse_feed(ATOM_FEED.encode("utf-8"))
        assert len(entries) == 1
        entry = entries[0]
        assert entry.title == "Atom 文章"
        assert entry.link == "https://example.org/posts/1"
        assert entry.published_at == datetime(2026, 11, 9, 6, 30, tzinfo=UTC)
        assert "Atom 正文" in entry.content_html

    def test_missing_date_is_none(self) -> None:
        entries = rss.parse_feed(RSS_NO_DATE.encode("utf-8"))
        assert entries[0].published_at is None

    def test_broken_feed_raises_parse_error(self) -> None:
        with pytest.raises(ParseError):
            rss.parse_feed(b"not a feed")

    def test_empty_feed_returns_empty_list(self) -> None:
        content = b"<rss version='2.0'><channel><title>x</title></channel></rss>"
        assert rss.parse_feed(content) == []


class TestFetchFeedBodyPriority:
    def test_content_encoded_used_without_page_fetch(self) -> None:
        client = _FakeClient({FEED_URL: RSS_WITH_CONTENT.encode("utf-8")})
        items = rss.fetch_feed(_source(), _ctx(client))

        assert len(items) == 1
        item = items[0]
        assert item.channel == "c1_rss"
        assert item.full_text_status == "ok"
        assert "完整正文" in item.full_text
        assert item.summary == "短摘要"
        assert item.published_at == "2026-11-09T08:00:00+00:00"
        # content:encoded 已含正文，不应再回源抓文章页
        assert client.calls == [FEED_URL]

    def test_short_summary_fetches_article_page(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            rss,
            "extract_article_text",
            lambda html, url="", **kwargs: "页面抽出的正文" * 60,
        )
        client = _FakeClient({FEED_URL: RSS_NO_CONTENT.encode("utf-8"), ARTICLE_URL: "<html/>"})
        items = rss.fetch_feed(_source(), _ctx(client))

        assert len(items) == 1
        assert items[0].full_text_status == "ok"
        assert "页面抽出的正文" in items[0].full_text
        assert client.calls == [FEED_URL, ARTICLE_URL]

    def test_page_fetch_failure_marks_failed_and_keeps_summary(self) -> None:
        client = _FakeClient({FEED_URL: RSS_NO_CONTENT.encode("utf-8")})
        items = rss.fetch_feed(_source(), _ctx(client))

        assert len(items) == 1
        assert items[0].full_text_status == "failed"
        assert items[0].full_text == ""
        assert items[0].summary == "只有很短的一段摘要"

    def test_long_summary_used_as_body_without_page_fetch(self) -> None:
        client = _FakeClient({FEED_URL: RSS_LONG_SUMMARY.encode("utf-8")})
        items = rss.fetch_feed(_source(), _ctx(client))

        assert len(items) == 1
        assert items[0].full_text_status == "ok"
        assert "足够长的摘要" in items[0].full_text
        assert client.calls == [FEED_URL]

    def test_allow_page_fetch_false_skips_article_page(self) -> None:
        client = _FakeClient({FEED_URL: RSS_NO_CONTENT.encode("utf-8")})
        items = rss.fetch_feed(
            _source(channel="c5_wechat2rss"), _ctx(client), allow_page_fetch=False
        )

        assert len(items) == 1
        assert items[0].full_text_status == "failed"
        assert client.calls == [FEED_URL]

    def test_all_feed_urls_failed_raises(self) -> None:
        client = _FakeClient()
        with pytest.raises(CollectError):
            rss.fetch_feed(_source(), _ctx(client))


class TestFetchFeedEdgeCases:
    def test_missing_link_entry_is_skipped(self) -> None:
        feed = (
            b"<rss version='2.0'><channel><item><title>no link</title>"
            b"<description>x</description></item></channel></rss>"
        )
        client = _FakeClient({FEED_URL: feed})
        assert rss.fetch_feed(_source(), _ctx(client)) == []

    def test_missing_date_item_kept_and_marked(self) -> None:
        client = _FakeClient({FEED_URL: RSS_NO_DATE.encode("utf-8")})
        items = rss.fetch_feed(_source(), _ctx(client, fetch_fulltext=False))

        assert len(items) == 1
        assert items[0].published_at == ""
        assert items[0].extra["no_published_at"] is True


class TestFinalizeItems:
    def test_dedupes_by_normalized_link(self) -> None:
        first = build_item(
            channel="c1_rss",
            source="s",
            source_label="S",
            category="news",
            title="标题一",
            link="https://example.com/a?utm_source=x",
            published_at="2026-11-09T08:00:00+00:00",
        )
        second = build_item(
            channel="c1_rss",
            source="s",
            source_label="S",
            category="news",
            title="标题一（重复）",
            link="https://example.com/a/",
            published_at="2026-11-09T09:00:00+00:00",
        )
        kept = finalize_items([first, second], _source(), _ctx(_FakeClient()))

        assert len(kept) == 1
        assert canonical_link(first.link) == canonical_link(second.link)
        assert first.item_id == second.item_id

    def test_window_filters_old_items_but_keeps_missing_dates(self) -> None:
        def make(title: str, published_at: str) -> Any:
            return build_item(
                channel="c1_rss",
                source="s",
                source_label="S",
                category="news",
                title=title,
                link=f"https://example.com/{title}",
                published_at=published_at,
            )

        recent = make("recent", (NOW - timedelta(hours=1)).isoformat())
        old = make("old", (NOW - timedelta(hours=100)).isoformat())
        missing = make("missing", "")
        kept = finalize_items([old, missing, recent], _source(), _ctx(_FakeClient()))

        assert kept == [recent, missing]

    def test_limit_keeps_newest(self) -> None:
        items = [
            build_item(
                channel="c1_rss",
                source="s",
                source_label="S",
                category="news",
                title=f"item-{hours}",
                link=f"https://example.com/{hours}",
                published_at=(NOW - timedelta(hours=hours)).isoformat(),
            )
            for hours in (5, 1, 3)
        ]
        kept = finalize_items(items, _source(limit=2), _ctx(_FakeClient()))

        assert [item.title for item in kept] == ["item-1", "item-3"]

    def test_skips_items_without_title_or_link(self) -> None:
        no_title = build_item(
            channel="c1_rss",
            source="s",
            source_label="S",
            category="news",
            title="",
            link="https://example.com/x",
        )
        no_link = build_item(
            channel="c1_rss",
            source="s",
            source_label="S",
            category="news",
            title="标题",
            link="",
        )
        assert finalize_items([no_title, no_link], _source(), _ctx(_FakeClient())) == []


class TestHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("<p>hello <b>world</b></p>", "hello world"),
            ("  multiple   spaces  ", "multiple spaces"),
            ("plain text", "plain text"),
            ("", ""),
        ],
    )
    def test_strip_html(self, raw: str, expected: str) -> None:
        assert strip_html(raw) == expected

    def test_detect_lang(self) -> None:
        assert detect_lang("这是一篇人工智能文章") == "zh"
        assert detect_lang("this is an ai article") == "en"

    def test_build_item_truncates_long_fields(self) -> None:
        item = build_item(
            channel="c1_rss",
            source="s",
            source_label="S",
            category="news",
            title="标题",
            link="https://example.com/x",
            summary="摘要" * 400,
            full_text="正文" * 10000,
        )
        assert len(item.summary) <= 500
        assert len(item.full_text) <= 12000
        assert item.full_text_status == "none"
