"""Stage 1 C3 RSSHub 通道：route → URL、容器就绪探测、通道抓取。

全部用例使用假的 httpx 客户端，不联网。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from ai_news_podcast.business.collect import rsshub
from ai_news_podcast.business.collect.base import (
    CollectError,
    ConfigError,
    FetchContext,
    SourceConfig,
)

NOW = datetime(2026, 11, 9, 16, 0, tzinfo=UTC)
BASE = "http://127.0.0.1:1200"

ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Anthropic</title>
  <entry>
    <title>Anthropic 发布新模型</title>
    <link rel="alternate" href="https://www.anthropic.com/news/model"/>
    <updated>2026-11-09T08:00:00Z</updated>
    <content type="html">&lt;p&gt;正文&lt;/p&gt;</content>
  </entry>
</feed>
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
        "rsshub_base": BASE,
        "now": NOW,
        "retry_wait": 0.0,
        "throttle_interval": 0.0,
    }
    params.update(overrides)
    return FetchContext(**params)


def _source(**overrides: Any) -> SourceConfig:
    params: dict[str, Any] = {
        "name": "anthropic_news",
        "label": "Anthropic News",
        "channel": "c3_rsshub",
        "rsshub_route": "/anthropic/news",
        "category": "official",
        "limit": 10,
    }
    params.update(overrides)
    return SourceConfig(**params)


class TestBuildFeedUrl:
    @pytest.mark.parametrize(
        ("base", "route", "expected"),
        [
            (BASE, "/anthropic/news", f"{BASE}/anthropic/news"),
            (f"{BASE}/", "/anthropic/news", f"{BASE}/anthropic/news"),
            (BASE, "anthropic/news", f"{BASE}/anthropic/news"),
            (BASE, "/ithome/tag/AI?limit=20", f"{BASE}/ithome/tag/AI?limit=20"),
            (
                BASE,
                "/qbitai/category/资讯",
                f"{BASE}/qbitai/category/%E8%B5%84%E8%AE%AF",
            ),
        ],
    )
    def test_cases(self, base: str, route: str, expected: str) -> None:
        assert rsshub.build_feed_url(base, route) == expected

    @pytest.mark.parametrize(("base", "route"), [("", "/x"), (BASE, "")])
    def test_invalid_raises(self, base: str, route: str) -> None:
        with pytest.raises(ConfigError):
            rsshub.build_feed_url(base, route)


class TestProbeRsshub:
    def test_healthz_ok(self) -> None:
        client = _FakeClient({f"{BASE}/healthz": _FakeResponse(text="OK")})
        assert rsshub.probe_rsshub(_ctx(client)) is True

    def test_unreachable_returns_false(self) -> None:
        assert rsshub.probe_rsshub(_ctx(_FakeClient())) is False

    def test_unhealthy_status_returns_false(self) -> None:
        client = _FakeClient({f"{BASE}/healthz": _FakeResponse(text="down", status_code=503)})
        assert rsshub.probe_rsshub(_ctx(client)) is False

    def test_no_base_returns_false(self) -> None:
        assert rsshub.probe_rsshub(_ctx(_FakeClient(), rsshub_base=None)) is False


class TestFetchRsshub:
    def test_uses_route_url_and_parses_feed(self) -> None:
        client = _FakeClient({f"{BASE}/anthropic/news": ATOM_FEED})
        items = rsshub.fetch_rsshub(_source(), _ctx(client))

        assert len(items) == 1
        assert items[0].channel == "c3_rsshub"
        assert items[0].title == "Anthropic 发布新模型"
        assert items[0].full_text_status == "ok"
        assert client.calls == [f"{BASE}/anthropic/news"]

    def test_requires_rsshub_base(self) -> None:
        with pytest.raises(CollectError):
            rsshub.fetch_rsshub(_source(), _ctx(_FakeClient(), rsshub_base=None))

    def test_requires_route(self) -> None:
        with pytest.raises(ConfigError):
            rsshub.fetch_rsshub(_source(rsshub_route=""), _ctx(_FakeClient()))

    def test_route_fetch_failure_raises(self) -> None:
        with pytest.raises(CollectError):
            rsshub.fetch_rsshub(_source(), _ctx(_FakeClient()))
