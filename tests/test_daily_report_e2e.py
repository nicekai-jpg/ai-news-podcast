"""End-to-end test for daily_report main() with mocked backends."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

import ai_news_podcast.cli.daily_report as report_module


@pytest.fixture
def report_config(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
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


@pytest.mark.asyncio
async def test_main_success(report_config: Path, monkeypatch) -> None:
    root = report_config
    monkeypatch.setattr(report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py"))
    monkeypatch.setattr(
        sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")]
    )

    from ai_news_podcast.pipeline.fetcher import RawItem

    raw_item = RawItem(
        id="abc",
        title="Test News",
        link="https://a.com",
        normalized_link="https://a.com",
        source_name="S",
        source_category="news",
        published_at=datetime.now().isoformat(),
        summary="Summary.",
        full_text_snippet="Text." * 100,
        category="product",
        language="zh",
    )

    async def _fake_fetch_all(*args, **kwargs):
        return [raw_item]

    monkeypatch.setattr(report_module, "fetch_all", _fake_fetch_all)

    def _fake_ollama(prompt: str, model: str) -> str:
        return "# Report\n\nGenerated content."

    monkeypatch.setattr(report_module, "_call_llm_ollama_direct", _fake_ollama)

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
    monkeypatch.setattr(report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py"))
    monkeypatch.setattr(
        sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")]
    )

    from ai_news_podcast.pipeline.fetcher import RawItem

    raw_item = RawItem(
        id="abc",
        title="Test News",
        link="https://a.com",
        normalized_link="https://a.com",
        source_name="S",
        source_category="news",
        published_at=datetime.now().isoformat(),
        summary="Summary.",
        full_text_snippet="Text." * 100,
        category="product",
        language="zh",
    )

    async def _fake_fetch_all(*args, **kwargs):
        return [raw_item]

    monkeypatch.setattr(report_module, "fetch_all", _fake_fetch_all)
    monkeypatch.setattr(report_module, "_call_llm_ollama_direct", lambda p, m: None)

    rc = await report_module.main()
    assert rc == 0

    reports = list((root / "data" / "reports").glob("daily_report_*.md"))
    content = reports[0].read_text(encoding="utf-8")
    assert "由于大模型服务暂时不可用" in content
    assert "Test News" in content


@pytest.mark.asyncio
async def test_main_no_items_returns_1(report_config: Path, monkeypatch) -> None:
    root = report_config
    monkeypatch.setattr(report_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "daily_report.py"))
    monkeypatch.setattr(
        sys, "argv", ["daily_report", "--outdir", str(root / "data" / "reports")]
    )

    async def _fake_fetch_all(*args, **kwargs):
        return []

    monkeypatch.setattr(report_module, "fetch_all", _fake_fetch_all)

    rc = await report_module.main()
    assert rc == 1
