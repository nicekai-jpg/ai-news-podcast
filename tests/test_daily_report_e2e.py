"""End-to-end test for daily_report main() with mocked backends."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

import ai_news_podcast.cli.daily_report as report_module


@pytest.fixture
def report_config(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "briefs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "reports").mkdir(parents=True, exist_ok=True)

    config_yaml = """
fetch:
  timeout_seconds: 5
  user_agent: "test"
  max_items_per_feed: 5

processing:
  dedup:
    rapidfuzz_threshold: 92
    jieba_keyword_overlap: 0.35
    title_sim_threshold: 85
    dedup_window_hours: 48
  clustering:
    ngram_range: [2, 4]
    eps: 0.35
    min_samples: 2
  scoring:
    dimensions:
      - "impact_scope"
      - "novelty"
      - "explainability"
      - "listener_relevance"
      - "source_richness"
    role_thresholds:
      main: [12, 15]
      supporting: [8, 11]
      quick: [5, 7]
      skip_below: 5
  source_authority:
    official: 1
    research: 2
    news: 3
  story_memory_days: 90
  max_pages_per_episode: 80
"""
    (tmp_path / "config" / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (tmp_path / "config" / "sources.yaml").write_text(
        "sources:\n  - name: X\n    url: https://x.com\n    enabled: true\n",
        encoding="utf-8",
    )
    return tmp_path


FAKE_BRIEF = {
    "thesis": "Test",
    "stories": [
        {
            "cluster_id": 0,
            "representative_title": "Test News",
            "role": "main",
            "role_emoji": "🔴",
            "total_score": 13,
            "scores": {
                "impact_scope": 3,
                "novelty": 2,
                "explainability": 3,
                "listener_relevance": 3,
                "source_richness": 2,
            },
            "context": {
                "factual_summary": ["Summary."],
                "historical_background": "",
                "sources_ranked": [],
            },
            "items": [],
        }
    ],
    "metadata": {},
}


@pytest.mark.asyncio
async def test_main_success(report_config: Path, monkeypatch) -> None:
    root = report_config
    today = datetime.now().strftime("%Y-%m-%d")
    brief_path = root / "data" / "briefs" / f"brief_{today}.json"
    brief_path.write_text(json.dumps(FAKE_BRIEF, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py")
    )
    monkeypatch.setattr(sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")])

    def _fake_llm(prompt: str, llm_cfg: dict) -> str:
        return "# Report\n\nGenerated content."

    monkeypatch.setattr(report_module, "call_llm", _fake_llm)

    rc = await report_module.main()
    assert rc == 0

    reports_dir = root / "data" / "reports"
    reports = list(reports_dir.glob("daily_report_*.md"))
    assert len(reports) == 1
    content = reports[0].read_text(encoding="utf-8")
    assert "# Report" in content


@pytest.mark.asyncio
async def test_main_fallback_when_llm_fails(report_config: Path, monkeypatch) -> None:
    root = report_config
    today = datetime.now().strftime("%Y-%m-%d")
    brief_path = root / "data" / "briefs" / f"brief_{today}.json"
    brief_path.write_text(json.dumps(FAKE_BRIEF, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py")
    )
    monkeypatch.setattr(sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")])

    monkeypatch.setattr(report_module, "call_llm", lambda p, cfg: None)

    rc = await report_module.main()
    assert rc == 0

    reports = list((root / "data" / "reports").glob("daily_report_*.md"))
    content = reports[0].read_text(encoding="utf-8")
    assert "由于大模型服务暂时不可用" in content
    assert "Test News" in content


@pytest.mark.asyncio
async def test_main_no_brief_returns_1(report_config: Path, monkeypatch) -> None:
    root = report_config
    monkeypatch.setattr(
        report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py")
    )
    monkeypatch.setattr(sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")])

    rc = await report_module.main()
    assert rc == 1


@pytest.mark.asyncio
async def test_main_no_stories_returns_1(report_config: Path, monkeypatch) -> None:
    root = report_config
    today = datetime.now().strftime("%Y-%m-%d")
    brief_path = root / "data" / "briefs" / f"brief_{today}.json"
    brief_path.write_text(
        json.dumps({"stories": [], "thesis": "", "metadata": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py")
    )
    monkeypatch.setattr(sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")])

    rc = await report_module.main()
    assert rc == 1
