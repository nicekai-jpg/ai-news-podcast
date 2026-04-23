"""Lightweight end-to-end test: brief → script → site files."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Create a minimal config tree in tmp_path and return its root."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    episodes_dir = site_dir / "episodes"
    episodes_dir.mkdir()

    config_yaml = """
podcast:
  title: "Test Podcast"
  description: "Test desc"
  language: "zh-cn"
  author: "Tester"
  category: "Technology"
  explicit: false
  keep_last: 30
  max_stories: 1

tts:
  backend: "edge-tts"
  voice: "zh-CN-XiaoxiaoNeural"

fetch:
  timeout_seconds: 5
  user_agent: "test/0.1"
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
    product: 3
    analysis: 3
    tools: 4
    events: 4
  story_memory_days: 90
  max_pages_per_episode: 80

script:
  mode_a:
    name: "连点成线"
    hook_chars: [150, 180]
    thesis_chars: [120, 160]
    main_chars: [1200, 1500]
    supporting_chars: [450, 550]
    quick_chars: [300, 450]
    closing_chars: [150, 220]
  mode_b:
    name: "工具优先"
  style:
    sentence_length: [12, 28]
    banned_words: []
    total_chars: [800, 2500]
    target_duration_minutes: [5, 10]

llm:
  provider: "openai_compatible"
  api_key_env: "LLM_API_KEY"
  model: "test-model"
  base_url: "http://localhost/v1"
  temperature: 0.7
  max_output_tokens: 2048
  timeout: 5

build:
  site_dir: "site"
  episodes_dir: "site/episodes"
  episodes_index: "data/episodes.json"
"""
    (config_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")

    sources_yaml = """
sources:
  - name: "Test Feed"
    url: "https://example.com/feed.xml"
    category: "news"
    enabled: true
"""
    (config_dir / "sources.yaml").write_text(sources_yaml, encoding="utf-8")

    # Placeholder episodes.json
    (data_dir / "episodes.json").write_text("[]", encoding="utf-8")

    return tmp_path


@pytest.mark.asyncio
async def test_no_audio_run_creates_files(minimal_config: Path, monkeypatch) -> None:
    """Run daily pipeline with --no-audio and verify output files."""
    root = minimal_config
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_daily",
            "--config", str(root / "config" / "config.yaml"),
            "--sources", str(root / "config" / "sources.yaml"),
            "--no-audio",
            "--date", "2024-03-15",
            "--base-url", "https://test.example.com",
        ],
    )

    # main() computes root = Path(__file__).resolve().parents[3]
    # Patch __file__ so it resolves to our temp tree.
    import ai_news_podcast.cli.run_daily as run_daily_module
    fake_file = root / "src" / "ai_news_podcast" / "cli" / "run_daily.py"
    monkeypatch.setattr(run_daily_module, "__file__", str(fake_file))

    # Mock fetch_all to return one raw item so process() has data
    from ai_news_podcast.pipeline.fetcher import RawItem

    raw_item = RawItem(
        id="abc123",
        title="Test AI News",
        link="https://example.com/1",
        normalized_link="https://example.com/1",
        source_name="Test Feed",
        source_category="news",
        published_at=datetime.now(tz=timezone.utc).isoformat(),
        summary="Summary text here.",
        full_text_snippet="Full text here." * 100,
        category="product",
        language="zh",
    )

    async def _fake_fetch_all(*args, **kwargs):
        return [raw_item]

    monkeypatch.setattr(run_daily_module, "fetch_all", _fake_fetch_all)

    # Mock generate_script to avoid calling LLM
    fake_script = "[Host A] 欢迎收听今日 AI 新闻。\n[Host B] 今天的主要内容如下。"
    monkeypatch.setattr(run_daily_module, "generate_script", lambda *a, **kw: (fake_script, []))

    # Mock synthesize to avoid TTS
    async def _fake_synthesize(*args, **kwargs) -> None:
        pass
    monkeypatch.setattr(run_daily_module, "synthesize", _fake_synthesize)

    # Run
    rc = await run_daily_module.main()
    assert rc == 0

    # Verify outputs
    episode_id = "2024-03-15"
    episodes_dir = root / "site" / "episodes"
    assert (episodes_dir / f"{episode_id}.txt").exists()
    assert (episodes_dir / f"{episode_id}.html").exists()
    # MP3 should NOT exist because --no-audio
    assert not (episodes_dir / f"{episode_id}.mp3").exists()

    # --no-audio mode skips feed.xml / index.html / episodes.json update
    assert not (root / "site" / "feed.xml").exists()
    assert not (root / "site" / "index.html").exists()
