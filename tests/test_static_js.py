"""站点静态资源质量门：JS 语法、无外链资源、注入点覆盖。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_STATIC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_news_podcast"
    / "presentation"
    / "site_builder"
    / "static"
)
_NODE = shutil.which("node")


def _read(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


def test_only_expected_assets() -> None:
    names = {path.name for path in _STATIC_DIR.iterdir() if path.is_file()}
    assert names == {"app.js", "style.css", "page_template.html", "logo.svg"}


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_app_js_syntax() -> None:
    result = subprocess.run(
        [str(_NODE), "--check", str(_STATIC_DIR / "app.js")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_no_external_script_or_style() -> None:
    app = _read("app.js")
    css = _read("style.css")
    assert "<script src=" not in app
    assert "@import" not in css
    assert "cdn." not in css


def test_template_placeholders_are_known() -> None:
    template = _read("page_template.html")
    keys = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", template))
    # 设计稿约定的注入字段：四章 × 六字段 + 全局字段
    chapter = {
        f"chapter_{n}_{name}"
        for n in range(1, 5)
        for name in (
            "tag",
            "topic",
            "title",
            "point_1_label",
            "point_1_text",
            "point_2_label",
            "point_2_text",
            "point_3_label",
            "point_3_text",
        )
    }
    global_keys = {
        "issue_number",
        "issue_date",
        "issue_tagline",
        "demo_note",
        "podcast_duration",
        "podcast_audio_url",
        "podcast_chapter_meta",
        "podcast_title",
        "podcast_subtitle",
        "player_subtitle",
        "host_a_name",
        "host_b_name",
        "lead_quote",
        "metric_sources_value",
        "metric_sources_unit",
        "metric_sources_label",
        "metric_hours_value",
        "metric_hours_unit",
        "metric_hours_label",
        "metric_speedup_value",
        "metric_speedup_unit",
        "metric_speedup_label",
        "metric_items_value",
        "metric_items_unit",
        "metric_items_label",
        "report_word_count",
        "report_read_minutes",
        "report_pill",
        "report_title",
        "editor_quote",
        "editor_name",
        "editor_role",
        "candidate_count",
        "excluded_recent",
        "radar_status_label",
        "radar_headline",
        "radar_paragraph",
        "radar_reason",
        "radar_pick_repo",
        "radar_pick_lang",
        "radar_pick_stars",
        "radar_pick_growth_text",
        "radar_pick_license",
        "radar_pick_desc",
        "radar_pick_url",
    }
    assert keys <= chapter | global_keys, keys - (chapter | global_keys)


def test_app_js_has_data_binding_contract() -> None:
    app = _read("app.js")
    for symbol in ("digestToData", "hydrate", "loadIssue", "applyConditionalBlocks", "toggleTheme"):
        assert symbol in app, symbol


def test_app_js_escapes_untrusted_html() -> None:
    """信源标题/摘要来自外部 RSS，逐字稿来自脚本：进 innerHTML 前必须转义。"""
    app = _read("app.js")
    assert "function escapeHtml" in app
    for expression in (
        "escapeHtml(title)",
        "escapeHtml(s.badge_label)",
        "escapeHtml(l.text)",
        "escapeHtml(h.name)",
        "escapeHtml(item.repo)",
    ):
        assert expression in app, expression


def test_app_js_only_allows_http_links() -> None:
    app = _read("app.js")
    assert "function safeHref" in app
    assert r"/^https?:\/\//i.test(value)" in app
    assert "SOURCE_BADGE_CLASSES" in app


def test_app_js_radar_color_is_defined() -> None:
    """回归：radar 十字线颜色曾被写成未定义变量 crossColor，导致动画首帧抛错。"""
    app = _read("app.js")
    assert "crossColor" not in app
    assert "rgba(212, 165, 116, 0.1)" in app


def test_radar_pick_description_is_cleaned() -> None:
    """仓库 README 首段是 Markdown/HTML 混排，直接渲染会出现 &nbsp;、**粗体**。"""
    app = _read("app.js")
    assert "function cleanReadmeText" in app
    assert "pick.description" in app


def test_app_js_date_archive_and_radar_candidates() -> None:
    app = _read("app.js")
    for symbol in (
        "DATE_PILL_LIMIT",
        "toggleArchive",
        "closeArchive",
        "toggleRadarCandidates",
        "renderRadarCandidates",
        "radar_candidates_preview",
    ):
        assert symbol in app, symbol


def test_hero_title_uses_short_design_style_title() -> None:
    """设计稿播客卡标题是「A 与 B」式短标题，不再是「品牌 | 日期」。"""
    app = _read("app.js")
    assert "const topTags = dives" in app
    assert "shortTitle || headline" in app
    assert "' 与 '" in app


def test_sources_summary_and_design_preview_count() -> None:
    app = _read("app.js")
    assert "summary: item.summary || ''" in app
    # 信源按设计稿展示 30 条精选（上限由命名常量控制，不再是魔法数字）
    assert "SOURCE_DISPLAY_LIMIT = 30" in app
    assert ".items || []).slice(0, 30)" not in app


def test_content_field_mapping_matches_design() -> None:
    """DESIGN §7：副标题是固定节目形态；信源 meta 用出刊日期；stamp 只显示日期。"""
    app = _read("app.js")
    assert "podcast_subtitle: '双主持对谈 · 今日焦点解读'" in app
    assert "meta: date," in app
    assert "stamp.textContent = DATA.issue_date" in app
    assert "radar_pick_growth_text" in app
    assert "dataReady ? '' : m" in app  # 数据就绪后不再残留 {{...}} 占位符


def test_report_title_keeps_masthead() -> None:
    """日报大标题保留报告报头（科技新闻日报 · 日期），避免与首章标题重复。"""
    app = _read("app.js")
    assert "report_title: cleanReportTitle(report.title || '')" in app
    assert "reportHeadline" not in app


def test_chapter_meta_falls_back_to_segment_count() -> None:
    """历史期没有真实 [CHAPTER] 章节时，01 板块头应显示「N 段对谈」而不是假章节数。"""
    app = _read("app.js")
    assert "podcast_chapter_meta" in app
    assert "段对谈" in app


def test_sources_capped_to_design_preview_count() -> None:
    app = _read("app.js")
    assert "SOURCE_DISPLAY_LIMIT = 30" in app


def test_sections_share_one_panel_style() -> None:
    """四个栏目外围统一：同样的底板，不再有 01/02 裸奔、03/04 冷暖底的差异。"""
    template = _read("page_template.html")
    assert template.count('class="content-section"') == 4
    assert "tint-cool" not in template and "tint-warm" not in template
    css = _read("style.css")
    assert "--surface-panel" in css
    assert ".content-section.tint-" not in css
