"""Branch coverage tests for run_daily main()."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_no_raw_items_returns_1(monkeypatch, tmp_path: Path) -> None:
    """If fetch_all returns empty, main() should return 1."""
    import ai_news_podcast.cli.run_daily as m

    fake_file = tmp_path / "src" / "ai_news_podcast" / "cli" / "run_daily.py"
    monkeypatch.setattr(m, "__file__", str(fake_file))

    # Create minimal config tree
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "site" / "episodes").mkdir(parents=True, exist_ok=True)

    config_yaml = """
podcast:
  title: "T"
  description: "D"
  language: "zh-cn"
  author: "A"
  category: "Technology"
  explicit: false
  keep_last: 30

tts:
  backend: "edge-tts"
  voice: "zh-CN-XiaoxiaoNeural"

fetch:
  timeout_seconds: 5
  user_agent: "test"
  max_items_per_feed: 5

selection:
  prefer_recent_hours: 36
  fallback_recent_hours: 96
  per_feed_cap: 6
  include_keywords: ["AI"]
  exclude_keywords: []

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

script:
  mode_a:
    name: "连点成线"
  mode_b:
    name: "工具优先"
  style:
    banned_words: []

llm:
  provider: "openai_compatible"
  api_key_env: "FAKE_KEY"
  model: "fake"
  base_url: "http://localhost/v1"
  temperature: 0.7
  max_output_tokens: 2048
  timeout: 5

build:
  site_dir: "site"
  episodes_dir: "site/episodes"
  episodes_index: "data/episodes.json"
"""
    (tmp_path / "config" / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (tmp_path / "config" / "sources.yaml").write_text("sources:\n  - name: X\n    url: https://x.com\n    enabled: true\n", encoding="utf-8")
    (tmp_path / "data" / "episodes.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["run_daily", "--no-audio", "--date", "2024-03-15"],
    )

    async def _fake_fetch_all(*args, **kwargs):
        return []

    monkeypatch.setattr(m, "fetch_all", _fake_fetch_all)

    rc = await m.main()
    assert rc == 1
