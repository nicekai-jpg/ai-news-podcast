"""Stage 1 runner：源清单装载、单源失败隔离、failover、artifact 落盘。

全部用例使用假的 httpx 客户端，不联网。
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from ai_news_podcast.business.collect import collect_all, collect_and_write
from ai_news_podcast.business.collect.base import ConfigError, load_sources
from ai_news_podcast.business.contracts.artifact import read_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 9, 16, 0, tzinfo=UTC)

GOOD_URL = "https://feed.example.com/good"
GOOD_URL_2 = "https://feed.example.com/good2"
BAD_URL = "https://feed.example.com/bad"
BAD_URL_2 = "https://feed.example.com/bad2"


def _atom_feed(
    *,
    title: str = "示例新闻",
    link: str = "https://example.com/a",
    updated: str = "2026-11-09T08:00:00Z",
) -> bytes:
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<feed xmlns='http://www.w3.org/2005/Atom'><title>Test</title>"
        f"<entry><title>{title}</title><link rel='alternate' href='{link}'/>"
        f"<updated>{updated}</updated><content type='html'>正文</content></entry>"
        "</feed>"
    ).encode()


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
        raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
        return _FakeResponse(text=raw.decode("utf-8", errors="replace"))


def _write_config(tmp_path: Path, sources: list[dict[str, Any]]) -> Path:
    path = tmp_path / "sources_test.yaml"
    path.write_text(
        yaml.safe_dump({"sources": sources}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _run(config: Path, client: Any, **overrides: Any) -> Any:
    params: dict[str, Any] = {
        "http_client": client,
        "now": NOW,
        "max_workers": 1,
        "request_interval": 0.0,
        "retry_wait": 0.0,
        "fetch_fulltext": False,
    }
    params.update(overrides)
    return collect_all(config, "2026-11-09", **params)


class TestShippedConfig:
    def test_source_counts(self) -> None:
        sources = load_sources(REPO_ROOT / "config" / "sources_v3.yaml")

        assert len(sources) == 132
        assert Counter(source.channel for source in sources) == Counter(
            {
                "c1_rss": 31,
                "c2_atom": 11,
                "c3_rsshub": 66,
                "c4_web": 7,
                "c5_wechat2rss": 17,
            }
        )

    def test_names_unique_and_channels_valid(self) -> None:
        sources = load_sources(REPO_ROOT / "config" / "sources_v3.yaml")

        assert len({source.name for source in sources}) == len(sources)
        assert {source.channel for source in sources} <= {
            "c1_rss",
            "c2_atom",
            "c3_rsshub",
            "c4_web",
            "c5_wechat2rss",
        }

    def test_all_sources_are_fetchable(self) -> None:
        sources = load_sources(REPO_ROOT / "config" / "sources_v3.yaml")

        for source in sources:
            assert source.page_urls() or source.rsshub_route, source.name
            assert source.limit > 0, source.name
            if source.channel == "c3_rsshub":
                assert source.rsshub_route.startswith("/"), source.name
            elif source.channel == "c4_web":
                assert source.parser in {"", "generic", "deepseek"}, source.name

    def test_disabled_source_is_36kr_native(self) -> None:
        sources = {
            source.name: source for source in load_sources(REPO_ROOT / "config" / "sources_v3.yaml")
        }

        disabled = [name for name, source in sources.items() if not source.enabled]
        assert disabled == ["36kr_native"]


class TestCollectAll:
    def test_isolates_failures_and_uses_failover(self, tmp_path: Path) -> None:
        config = _write_config(
            tmp_path,
            [
                {
                    "name": "good_feed",
                    "label": "Good",
                    "channel": "c1_rss",
                    "url": GOOD_URL,
                    "category": "news",
                    "limit": 5,
                },
                {
                    "name": "bad_feed",
                    "label": "Bad",
                    "channel": "c1_rss",
                    "url": BAD_URL,
                    "category": "news",
                },
                {
                    "name": "disabled_feed",
                    "label": "Disabled",
                    "channel": "c1_rss",
                    "url": GOOD_URL,
                    "enabled": False,
                },
                {
                    "name": "c3_failover",
                    "label": "C3 Failover",
                    "channel": "c3_rsshub",
                    "rsshub_route": "/demo/news",
                    "category": "news",
                    "failover": {"channel": "c1_rss", "url": GOOD_URL_2},
                },
            ],
        )
        client = _FakeClient(
            {
                GOOD_URL: _atom_feed(title="好消息"),
                GOOD_URL_2: _atom_feed(title="备用消息", link="https://example.com/b"),
            }
        )
        result = _run(config, client)

        stats = {stat["name"]: stat for stat in result.source_stats}
        assert stats["good_feed"]["status"] == "ok"
        assert stats["good_feed"]["count"] == 1
        assert stats["bad_feed"]["status"] == "failed"
        assert stats["bad_feed"]["error"]
        assert stats["disabled_feed"]["status"] == "disabled"
        assert stats["c3_failover"]["status"] == "ok"
        assert stats["c3_failover"]["failover"] is True
        assert len(result.items) == 2

        by_source = {item.source: item for item in result.items}
        assert by_source["good_feed"].channel == "c1_rss"
        # failover 实际走的是备用通道，item.channel 如实反映
        assert by_source["c3_failover"].channel == "c1_rss"
        assert by_source["c3_failover"].title == "备用消息"

    def test_c3_skipped_without_rsshub(self, tmp_path: Path) -> None:
        config = _write_config(
            tmp_path,
            [
                {
                    "name": "c3_only",
                    "label": "C3",
                    "channel": "c3_rsshub",
                    "rsshub_route": "/demo/news",
                    "category": "news",
                }
            ],
        )
        result = _run(config, _FakeClient())

        assert result.source_stats[0]["status"] == "skipped"
        assert result.items == []

    def test_empty_feed_status_empty(self, tmp_path: Path) -> None:
        empty_feed = b"<rss version='2.0'><channel><title>x</title></channel></rss>"
        config = _write_config(
            tmp_path,
            [{"name": "empty_feed", "label": "Empty", "channel": "c1_rss", "url": GOOD_URL}],
        )
        result = _run(config, _FakeClient({GOOD_URL: empty_feed}))

        assert result.source_stats[0]["status"] == "empty"
        assert result.items == []

    def test_window_filters_old_items(self, tmp_path: Path) -> None:
        config = _write_config(
            tmp_path,
            [{"name": "old_feed", "label": "Old", "channel": "c1_rss", "url": GOOD_URL}],
        )
        old_feed = _atom_feed(updated="2025-01-01T00:00:00Z")
        result = _run(config, _FakeClient({GOOD_URL: old_feed}))

        assert result.source_stats[0]["status"] == "empty"
        assert result.items == []

    def test_failover_failure_recorded_as_failed(self, tmp_path: Path) -> None:
        config = _write_config(
            tmp_path,
            [
                {
                    "name": "bad_then_bad",
                    "label": "Bad",
                    "channel": "c1_rss",
                    "url": BAD_URL,
                    "failover": {"channel": "c1_rss", "url": BAD_URL_2},
                }
            ],
        )
        result = _run(config, _FakeClient())

        assert result.source_stats[0]["status"] == "failed"
        assert result.items == []

    def test_parallel_collection_preserves_source_order(self, tmp_path: Path) -> None:
        sources = [
            {
                "name": f"feed_{index}",
                "label": f"Feed {index}",
                "channel": "c1_rss",
                "url": f"https://feed.example.com/{index}",
            }
            for index in range(3)
        ]
        config = _write_config(tmp_path, sources)
        client = _FakeClient(
            {
                f"https://feed.example.com/{index}": _atom_feed(
                    title=f"新闻{index}",
                    link=f"https://example.com/{index}",
                )
                for index in range(3)
            }
        )
        result = _run(config, client, max_workers=3)

        assert [stat["name"] for stat in result.source_stats] == [
            "feed_0",
            "feed_1",
            "feed_2",
        ]
        assert len(result.items) == 3


class TestCollectAndWrite:
    def _config(self, tmp_path: Path) -> Path:
        return _write_config(
            tmp_path,
            [
                {
                    "name": "good_feed",
                    "label": "Good",
                    "channel": "c1_rss",
                    "url": GOOD_URL,
                    "category": "news",
                }
            ],
        )

    def test_writes_artifact_into_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        path = collect_and_write(
            self._config(tmp_path),
            "2026-11-09",
            output_dir,
            http_client=_FakeClient({GOOD_URL: _atom_feed()}),
            now=NOW,
            max_workers=1,
            request_interval=0.0,
            retry_wait=0.0,
            fetch_fulltext=False,
        )

        assert path == output_dir / "raw-2026-11-09.json.gz"
        manifest, items = read_artifact(path)
        assert manifest["date"] == "2026-11-09"
        assert manifest["window_hours"] == 72
        assert manifest["item_count"] == 1
        assert manifest["source_stats"][0]["name"] == "good_feed"
        assert len(items) == 1
        assert items[0].channel == "c1_rss"
        assert items[0].title == "示例新闻"

    def test_writes_artifact_to_explicit_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "artifacts" / "custom.json.gz"
        path = collect_and_write(
            self._config(tmp_path),
            "2026-11-09",
            output_file,
            http_client=_FakeClient({GOOD_URL: _atom_feed()}),
            now=NOW,
            max_workers=1,
            request_interval=0.0,
            retry_wait=0.0,
            fetch_fulltext=False,
        )

        assert path == output_file
        manifest, items = read_artifact(path)
        assert manifest["item_count"] == 1
        assert len(items) == 1


class TestConfigValidation:
    @pytest.mark.parametrize(
        "sources",
        [
            [
                {"name": "dup", "label": "A", "channel": "c1_rss", "url": GOOD_URL},
                {"name": "dup", "label": "B", "channel": "c1_rss", "url": GOOD_URL},
            ],
            [{"name": "bad_channel", "label": "A", "channel": "c9_unknown", "url": GOOD_URL}],
            [{"name": "no_route", "label": "A", "channel": "c3_rsshub"}],
            [{"name": "no_url", "label": "A", "channel": "c1_rss"}],
        ],
    )
    def test_invalid_config_raises(self, tmp_path: Path, sources: list[dict[str, Any]]) -> None:
        with pytest.raises(ConfigError):
            load_sources(_write_config(tmp_path, sources))
