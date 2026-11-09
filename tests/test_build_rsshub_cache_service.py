"""Tests for scripts/build_rsshub_cache.py.

scripts/ 不是包，这里用 importlib 按路径加载脚本模块（同 test_prune_gh_pages.py）。
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_rsshub_cache.py"

_spec = importlib.util.spec_from_file_location("build_rsshub_cache", _SCRIPT)
cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache)


_RSS = """<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel>
  <item>
    <title>A &amp; B</title>
    <link>https://example.com/a</link>
    <pubDate>Thu, 06 Nov 2026 10:00:00 GMT</pubDate>
    <description>&lt;p&gt;Hello &amp;lt;world&amp;gt;&lt;/p&gt;</description>
  </item>
  <item>
    <title>Second</title>
    <link>https://example.com/b</link>
    <pubDate>Wed, 05 Nov 2026 10:00:00 GMT</pubDate>
    <content:encoded><![CDATA[<p>Full text here</p>]]></content:encoded>
  </item>
</channel>
</rss>
"""

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>v0.2.2-rc.1</title>
    <link rel="alternate" href="https://github.com/deepseek-ai/deepseek-harness/releases/tag/v0.2.2-rc.1"/>
    <updated>2026-11-04T13:07:14Z</updated>
    <content type="html">&lt;p&gt;Release notes&lt;/p&gt;</content>
  </entry>
</feed>
"""


class TestParseFeed:
    def test_rss_items(self) -> None:
        items = cache.parse_feed(_RSS)

        assert len(items) == 2
        assert items[0]["title"] == "A & B"
        assert items[0]["link"] == "https://example.com/a"
        assert items[0]["published_at"] == "2026-11-06T10:00:00+00:00"
        assert items[0]["summary"] == "Hello <world>"
        # content:encoded 优先于 description
        assert items[1]["summary"] == "Full text here"

    def test_atom_items(self) -> None:
        items = cache.parse_feed(_ATOM)

        assert len(items) == 1
        assert items[0]["title"] == "v0.2.2-rc.1"
        assert items[0]["link"].endswith("/releases/tag/v0.2.2-rc.1")
        assert items[0]["published_at"] == "2026-11-04T13:07:14+00:00"
        assert items[0]["summary"] == "Release notes"

    def test_skips_items_without_title_or_link(self) -> None:
        xml = "<rss><channel><item><title>only title</title></item></channel></rss>"
        assert cache.parse_feed(xml) == []

    def test_bad_xml_raises(self) -> None:
        import xml.etree.ElementTree as ET

        try:
            cache.parse_feed("<rss><channel>")
        except ET.ParseError:
            return
        raise AssertionError("expected ParseError")


class TestNormalizeItems:
    def _item(self, link: str, published: str | None) -> dict:
        return {
            "title": link,
            "link": link,
            "published_at": published,
            "summary": "",
        }

    def test_dedupes_filters_and_sorts(self) -> None:
        now = datetime(2026, 11, 7, tzinfo=UTC)
        items = [
            self._item("https://a", (now - timedelta(days=1)).isoformat()),
            self._item("https://b", (now - timedelta(days=30)).isoformat()),  # 太旧
            self._item("https://a", now.isoformat()),  # 重复
            self._item("https://c", None),  # 无日期，保留
        ]

        kept = cache.normalize_items(items, days=14, max_items=10, now=now)

        assert [item["link"] for item in kept] == ["https://a", "https://c"]

    def test_max_items(self) -> None:
        now = datetime(2026, 11, 7, tzinfo=UTC)
        items = [
            self._item(f"https://x/{i}", (now - timedelta(hours=i)).isoformat()) for i in range(5)
        ]
        assert len(cache.normalize_items(items, days=14, max_items=3, now=now)) == 3


class TestResolveUrl:
    def test_route_joined_with_base(self) -> None:
        source = {"route": "/anthropic/news"}
        assert cache.resolve_url(source, "http://localhost:1200/") == (
            "http://localhost:1200/anthropic/news"
        )

    def test_absolute_url_wins(self) -> None:
        source = {"route": "/ignored", "url": "https://example.com/feed.atom"}
        assert cache.resolve_url(source, "http://localhost:1200") == "https://example.com/feed.atom"

    def test_missing_both_returns_empty(self) -> None:
        assert cache.resolve_url({}, "http://localhost:1200") == ""


class TestBuildAndWrite:
    def test_build_snapshot(self, monkeypatch) -> None:
        monkeypatch.setattr(cache, "fetch_xml", lambda url, timeout=60: _RSS)
        snapshot = cache.build_snapshot(
            {"name": "demo", "label": "Demo", "route": "/demo", "category": "news"},
            base_url="http://localhost:1200",
            days=14,
            max_items=10,
        )
        assert snapshot["source"] == "demo"
        assert snapshot["label"] == "Demo"
        assert snapshot["item_count"] == 2
        assert snapshot["url"] == "http://localhost:1200/demo"
        assert snapshot["fetched_at"]

    def test_write_snapshot_sanitizes_name(self, tmp_path: Path) -> None:
        snapshot = {"source": "a/b c", "items": []}
        path = cache.write_snapshot(snapshot, tmp_path)
        assert path.name == "a_b_c.json"
        assert json.loads(path.read_text(encoding="utf-8"))["source"] == "a/b c"

    def test_sources_without_url_or_route_raise(self) -> None:
        try:
            cache.build_snapshot({}, base_url="http://localhost:1200", days=14, max_items=10)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


class TestMain:
    def _config(self, tmp_path: Path) -> Path:
        path = tmp_path / "routes.json"
        path.write_text(
            json.dumps(
                {
                    "sources": [
                        {"name": "one", "route": "/one", "category": "news"},
                        {"name": "two", "url": "https://example.com/two.atom"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_dry_run_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cache, "fetch_xml", lambda url, timeout=60: _RSS)
        out = tmp_path / "cache"
        code = cache.main(
            [
                "--config",
                str(self._config(tmp_path)),
                "--output-dir",
                str(out),
                "--dry-run",
            ]
        )
        assert code == 0
        assert not out.exists()

    def test_writes_snapshots(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cache, "fetch_xml", lambda url, timeout=60: _RSS)
        out = tmp_path / "cache"
        code = cache.main(["--config", str(self._config(tmp_path)), "--output-dir", str(out)])
        assert code == 0
        files = sorted(path.name for path in out.glob("*.json"))
        assert files == ["one.json", "two.json"]
        data = json.loads((out / "one.json").read_text(encoding="utf-8"))
        assert data["item_count"] == 2

    def test_partial_failure_keeps_old_snapshot(self, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "cache"
        out.mkdir()
        old = out / "one.json"
        old.write_text('{"source": "one", "items": []}', encoding="utf-8")

        def fetch(url, timeout=60):
            if url.endswith("/one"):
                raise urllib.error.URLError("boom")
            return _RSS

        monkeypatch.setattr(cache, "fetch_xml", fetch)
        code = cache.main(["--config", str(self._config(tmp_path)), "--output-dir", str(out)])

        assert code == 0
        assert old.read_text(encoding="utf-8") == '{"source": "one", "items": []}'

    def test_all_fail_returns_error(self, tmp_path: Path, monkeypatch) -> None:
        def fetch(url, timeout=60):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(cache, "fetch_xml", fetch)
        code = cache.main(
            ["--config", str(self._config(tmp_path)), "--output-dir", str(tmp_path / "cache")]
        )
        assert code == 1
