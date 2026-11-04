"""新首页生成契约：结构、数据块、DOM id、体积预算。"""

import gzip
import json
import re

from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.presentation.site_builder.html_gen_view import _voice_labels, build_index_html

_EPISODES = [
    {
        "id": "2026-10-15",
        "title": "AI 新闻快报 | 2026-10-15",
        "enclosure_url": "https://example.com/episodes/2026-10-15.mp3",
        "description": "<p>desc</p>",
    },
]
_MOCK_MARKERS = (
    "PLAYLIST_DATA",
    "SOURCES_DATA",
    "RADAR_DATA",
    "CANDIDATES_DATA",
    "Lorem ipsum",
    "第 286 期",
    "toggleRadarMockState",
    "simulateDateSwitch",
    "cdn.jsdelivr.net",
)


def _build(tmp_path, episodes=None, cfg=None) -> str:
    site = tmp_path / "site"
    build_index_html(
        site,
        "AI 每日先锋",
        episodes if episodes is not None else _EPISODES,
        "https://example.com",
        cfg,
    )
    return (site / "index.html").read_text(encoding="utf-8")


def _site_data(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S)
    assert match, "缺 site-data JSON 块"
    return json.loads(match.group(1))


class TestSkeleton:
    def test_four_sections_present(self, tmp_path) -> None:
        html = _build(tmp_path)
        for anchor in (
            'id="section-hero"',
            'id="section-daily"',
            'id="section-sources"',
            'id="section-radar"',
        ):
            assert html.count(anchor) == 1, anchor

    def test_docked_player_and_drawer(self, tmp_path) -> None:
        html = _build(tmp_path)
        for dom_id in ("docked-player", "transcript-drawer", "main-audio", "toast-pill"):
            assert html.count(f'id="{dom_id}"') == 1, dom_id
        assert 'href="#section-hero"' in html and "./feed.xml" in html

    def test_no_legacy_tabs_or_variants(self, tmp_path) -> None:
        html = _build(tmp_path)
        for legacy in (
            "panel-podcast",
            "btn-mode-",
            "variant-container",
            "ab-variant",
            "theme-variant",
            "mobile-modal",
        ):
            assert legacy not in html, legacy

    def test_no_apple_entry_in_v1(self, tmp_path) -> None:
        """D15：Apple Podcasts 入口 v1 不做，只留 RSS。"""
        html = _build(tmp_path)
        assert "apple-link" not in html
        assert "podcasts.apple.com" not in html
        assert "./feed.xml" in html


class TestSiteData:
    def test_parsable_and_scoped(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path))
        assert data["dates"] == ["2026-10-15"]
        assert data["episodes"]["2026-10-15"]["mp3"].endswith("2026-10-15.mp3")
        assert set(data["voices"]) == {"host_a", "host_b"}

    def test_empty_episodes(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path, episodes=[]))
        assert data["dates"] == []

    def test_desc_not_inlined(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert "<p>desc</p>" not in html, "desc 不应再内联（体积红线）"


class TestNoMockLeftovers:
    def test_no_prototype_constants(self, tmp_path) -> None:
        html = _build(tmp_path)
        for marker in _MOCK_MARKERS:
            assert marker not in html, f"原型 mock 残留: {marker}"


class TestVoiceLabelsContract:
    """音色标签收敛契约（原 test_html_gen_view.py::TestVoiceLabels 的 7 个用例）。"""

    def test_defaults_without_config(self) -> None:
        labels = _voice_labels({})
        assert set(labels["host_a"]) == {"professional", "lively"}

    def test_global_list_filters_both_hosts(self) -> None:
        cfg = {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"professional"}
        assert set(labels["host_b"]) == {"professional"}

    def test_per_host_dict_filters_independently(self) -> None:
        cfg = {
            "tts": {
                "cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}
            }
        }
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"lively"}
        assert set(labels["host_b"]) == {"professional"}

    def test_appconfig_cfg(self) -> None:
        cfg = AppConfig.from_dict(
            {
                "tts": {
                    "cosyvoice": {
                        "synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}
                    }
                }
            }
        )
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"lively"}
        assert set(labels["host_b"]) == {"professional"}

    def test_partial_custom_voice_names_does_not_crash(self) -> None:
        cfg = {
            "tts": {
                "voice_names": {"host_a": {"professional": "亲切女声"}},
                "cosyvoice": {"synth_variants": ["professional"]},
            },
        }
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"professional"}
        assert set(labels["host_b"]) == {"professional"}

    def test_scalar_variant_value_ignored(self) -> None:
        cfg = {
            "tts": {
                "cosyvoice": {"synth_variants": {"host_a": "lively", "host_b": ["professional"]}}
            }
        }
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"professional", "lively"}
        assert set(labels["host_b"]) == {"professional"}

    def test_none_cfg_returns_defaults(self) -> None:
        labels = _voice_labels(None)
        assert set(labels["host_a"]) == {"professional", "lively"}
        assert set(labels["host_b"]) == {"professional", "lively"}


class TestSizeBudget:
    def test_inline_budget(self, tmp_path) -> None:
        html = _build(
            tmp_path, episodes=[{"id": f"2026-09-{d:02d}", "title": "t"} for d in range(1, 31)]
        )
        raw = len(html.encode("utf-8"))
        assert raw < 1_500_000, raw
        assert len(gzip.compress(html.encode("utf-8"))) < 300_000
