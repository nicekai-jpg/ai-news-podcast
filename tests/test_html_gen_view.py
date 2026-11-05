"""首页生成契约：设计稿 v23（今日AI看点）骨架 + 内联资源 + 注入点。"""

import gzip
import json
import re

from ai_news_podcast.presentation.site_builder.html_gen_view import build_index_html

_EPISODES = [
    {
        "id": "2026-10-15",
        "title": "AI 新闻快报 | 2026-10-15",
        "enclosure_url": "https://example.com/episodes/2026-10-15.mp3",
        "description": "<p>desc</p>",
    },
    {
        "id": "2026-10-14",
        "title": "AI 新闻快报 | 2026-10-14",
        # 无 enclosure_url：应回退到 base_url 拼接
    },
]
_OLD_MARKERS = (
    "episodes.json.desc",
    "cdn.jsdelivr.net",
    "marked.js",
    "simulateDateSwitch",
    "blob-data",
    "panel-podcast",
    "Apple Podcasts",
)


def _build(tmp_path, episodes=None) -> str:
    site = tmp_path / "site"
    build_index_html(
        site,
        "今日AI看点",
        episodes if episodes is not None else _EPISODES,
        "https://example.com",
    )
    return (site / "index.html").read_text(encoding="utf-8")


def _site_data(html: str) -> dict:
    match = re.search(r'<script id="site-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "缺 site-data JSON 块"
    return json.loads(match.group(1))


class TestSkeleton:
    def test_four_sections_present(self, tmp_path) -> None:
        html = _build(tmp_path)
        for anchor in (
            'id="section-podcast"',
            'id="section-report"',
            'id="section-sources"',
            'id="section-radar"',
        ):
            assert html.count(anchor) == 1, anchor
        assert len(re.findall(r'class="section-number[^"]*">0[1-4]', html)) == 4

    def test_brand_and_theme_toggle(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert "今日AI看点" in html
        assert 'class="brand-badge">DAILY' in html
        assert 'id="theme-toggle"' in html
        assert 'id="theme-label"' in html
        assert "localStorage.getItem('theme')" in html

    def test_date_axis_both_sides(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'id="btn-prev-issue"' in html
        assert 'id="btn-next-issue"' in html
        assert 'id="date-pills"' in html

    def test_docked_player_and_drawer(self, tmp_path) -> None:
        html = _build(tmp_path)
        for dom_id in ("transcript-drawer", "drawer-overlay", "transcript-body", "player-current"):
            assert html.count(f'id="{dom_id}"') == 1, dom_id
        assert 'class="docked-player"' in html
        assert 'id="radarCanvas"' in html

    def test_no_legacy_or_mock_markers(self, tmp_path) -> None:
        html = _build(tmp_path)
        for legacy in _OLD_MARKERS:
            assert legacy not in html, legacy

    def test_logo_is_inline_svg_data_uri(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'src="data:image/svg+xml;base64,' in html
        assert 'src="logo.svg"' not in html
        # favicon 同样内联，避免每次打开页面 404 /favicon.ico
        assert '<link rel="icon" href="data:image/svg+xml;base64,' in html

    def test_inline_assets_only(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert "<style>" in html and "<script>" in html
        assert "<script src=" not in html
        assert 'rel="stylesheet"' not in html
        assert "hydrate" in html and "digestToData" in html

    def test_page_size_budget(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert len(html.encode()) < 200_000
        assert len(gzip.compress(html.encode())) < 60_000


class TestSiteData:
    def test_parsable_and_sorted(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path))
        assert data["dates"] == ["2026-10-15", "2026-10-14"]
        assert data["episodes"]["2026-10-15"]["mp3"].startswith("https://example.com/episodes/")
        assert (
            data["episodes"]["2026-10-14"]["mp3"] == "https://example.com/episodes/2026-10-14.mp3"
        )

    def test_empty_episodes(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path, episodes=[]))
        assert data == {"dates": [], "episodes": {}}

    def test_episode_title_fallback_uses_brand(self, tmp_path) -> None:
        data = _site_data(_build(tmp_path, episodes=[{"id": "2026-10-13"}]))
        assert data["episodes"]["2026-10-13"]["title"] == "今日AI看点 | 2026-10-13"


class TestAccessibility:
    def test_drawer_dialog_semantics(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'role="dialog"' in html
        assert 'aria-modal="true"' in html
        assert 'aria-labelledby="drawer-heading"' in html

    def test_interactive_controls_have_labels(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'aria-label="切换深浅主题"' in html
        assert 'aria-label="关闭"' in html
        assert 'aria-label="后退 15 秒"' in html
        assert 'aria-label="前进 15 秒"' in html

    def test_sections_and_nav_have_labels(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'aria-label="章节导航"' in html
        assert 'aria-hidden="true"' in html  # radar canvas 装饰


class TestFollowupRegressions:
    """v23 重做后补回的能力：30 期归档入口 + 雷达候选审计明细。"""

    def test_date_axis_has_archive_panel(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'id="date-archive-panel"' in html
        assert 'class="archive-panel"' in html

    def test_radar_candidate_audit_entry(self, tmp_path) -> None:
        html = _build(tmp_path)
        assert 'id="btn-radar-candidates"' in html
        assert 'id="radar-candidates-panel"' in html
        assert 'aria-controls="radar-candidates-panel"' in html
        assert 'id="radar-empty-actions"' in html
