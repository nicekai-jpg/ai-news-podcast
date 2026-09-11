"""Tests for material.py — build_material_text with different strategies."""

from __future__ import annotations

import pytest

from ai_news_podcast.pipeline.material import build_material_text, build_radar_text


class TestBuildMaterialText:
    def _make_brief(self, stories: list[dict]) -> dict:
        return {"stories": stories, "thesis": "test", "metadata": {}}

    def test_pure_score_strategy(self) -> None:
        """pure_score strategy should return top N by total_score."""
        stories = [
            {
                "representative_title": "Story A",
                "role": "main",
                "total_score": 15,
                "context": {
                    "factual_summary": ["Summary A"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
            {
                "representative_title": "Story B",
                "role": "supporting",
                "total_score": 10,
                "context": {
                    "factual_summary": ["Summary B"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
            {
                "representative_title": "Story C",
                "role": "quick",
                "total_score": 5,
                "context": {
                    "factual_summary": ["Summary C"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
        ]
        brief = self._make_brief(stories)
        result = build_material_text(brief, max_stories=2, strategy="pure_score")

        assert "Story A" in result
        assert "Story B" in result
        assert "Story C" not in result  # Only top 2

    def test_score_diversity_strategy(self) -> None:
        """score_diversity strategy should apply entity penalty."""
        stories = [
            {
                "representative_title": "OpenAI releases GPT-5",
                "role": "main",
                "total_score": 15,
                "context": {
                    "factual_summary": ["Summary 1"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
            {
                "representative_title": "OpenAI announces new feature",
                "role": "supporting",
                "total_score": 12,
                "context": {
                    "factual_summary": ["Summary 2"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
            {
                "representative_title": "Google launches Bard 2",
                "role": "quick",
                "total_score": 10,
                "context": {
                    "factual_summary": ["Summary 3"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
        ]
        brief = self._make_brief(stories)
        result = build_material_text(brief, max_stories=2, strategy="score_diversity")

        # With diversity penalty, Google (score 10) should beat OpenAI (score 12)
        # because OpenAI is already represented by the first story
        assert "OpenAI releases GPT-5" in result
        assert "Google launches Bard 2" in result
        assert "OpenAI announces new feature" not in result

    def test_skip_stories_are_excluded(self) -> None:
        """Stories with role='skip' should be excluded."""
        stories = [
            {
                "representative_title": "Included Story",
                "role": "main",
                "total_score": 15,
                "context": {
                    "factual_summary": ["Summary"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
            {
                "representative_title": "Skipped Story",
                "role": "skip",
                "total_score": 3,
                "context": {
                    "factual_summary": ["Skip"],
                    "historical_background": "",
                    "sources_ranked": [],
                },
            },
        ]
        brief = self._make_brief(stories)
        result = build_material_text(brief, max_stories=5, strategy="pure_score")

        assert "Included Story" in result
        assert "Skipped Story" not in result

    def test_empty_brief(self) -> None:
        """Empty brief should return empty string."""
        brief = self._make_brief([])
        result = build_material_text(brief, max_stories=5, strategy="pure_score")
        assert result == ""

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy should raise ValueError."""
        brief = self._make_brief([])
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_material_text(brief, max_stories=5, strategy="unknown")

    def test_material_format(self) -> None:
        """Material should be formatted with markers and summaries."""
        stories = [
            {
                "representative_title": "Test Title",
                "role": "main",
                "total_score": 15,
                "context": {
                    "factual_summary": ["Fact 1", "Fact 2"],
                    "historical_background": "Background info",
                    "sources_ranked": [{"name": "TechCrunch"}],
                },
            },
        ]
        brief = self._make_brief(stories)
        result = build_material_text(brief, max_stories=1, strategy="pure_score")

        assert "【素材1】" in result
        assert "Test Title" in result
        assert "Fact 1" in result
        assert "Fact 2" in result
        assert "Background info" in result
        assert "TechCrunch" in result


class TestBuildRadarText:
    def test_empty_radar_returns_empty(self) -> None:
        assert build_radar_text(None) == ""
        assert build_radar_text({"projects": []}) == ""

    def test_formats_pick_with_excerpt_and_numbers(self) -> None:
        radar = {
            "projects": [
                {
                    "repo": "owner/hot",
                    "url": "https://github.com/owner/hot",
                    "stars": 1500,
                    "delta_stars": 1000,
                    "language": "Python",
                    "license": "MIT",
                    "description": "Fast LLM harness",
                    "readme_excerpt": "pip install hot",
                },
                {
                    "repo": "owner/next",
                    "url": "u2",
                    "stars": 800,
                    "delta_stars": None,
                    "language": "Rust",
                    "license": "Apache-2.0",
                    "description": "Agent runtime",
                    "readme_excerpt": "",
                },
            ],
            "meta": {"pick_repo": "owner/hot"},
        }
        text = build_radar_text(radar)
        assert "[主推]" in text and "owner/hot" in text
        assert "⭐1500" in text and "较昨日 +1000" in text
        assert "pip install hot" in text and "逐字" in text
        assert "(摘录结束)" in text
        assert "其余候选" in text and "owner/next ⭐800" in text
        assert "禁止编造" in text

    def test_pick_with_none_delta_shows_first_day(self) -> None:
        """唯一项目即主推且 delta_stars 为 None 时,应显示"首日无对比数据"。"""
        radar = {
            "projects": [
                {
                    "repo": "owner/solo",
                    "url": "https://github.com/owner/solo",
                    "stars": 42,
                    "delta_stars": None,
                    "language": "Go",
                    "license": None,
                    "description": "Brand new tool",
                    "readme_excerpt": "",
                }
            ],
            "meta": {"pick_repo": "owner/solo"},
        }
        text = build_radar_text(radar)
        assert "首日无对比数据" in text
        assert "[主推]" in text

    def test_unmatched_pick_repo_degrades_safely(self) -> None:
        """pick_repo 匹配不到任何项目时,降级为仅其余候选+使用规则,不出现主推段。"""
        radar = {
            "projects": [
                {
                    "repo": "owner/next",
                    "url": "u2",
                    "stars": 800,
                    "delta_stars": None,
                    "language": "Rust",
                    "license": "Apache-2.0",
                    "description": "Agent runtime",
                    "readme_excerpt": "",
                }
            ],
            "meta": {"pick_repo": "owner/ghost"},
        }
        text = build_radar_text(radar)
        assert "[主推]" not in text
        assert "其余候选" in text
        assert (
            "使用规则:仓库名、数字、安装命令必须原样引用,禁止编造或修改;只讲主推项目,不要展开其余候选;禁止把项目与新闻混在同一栏目。"
            in text
        )
