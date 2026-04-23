"""Full branch test for run_daily with audio generation and publishing."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import ai_news_podcast.cli.run_daily as run_daily_module


@pytest.fixture
def full_config(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "site" / "episodes").mkdir(parents=True, exist_ok=True)

    config_yaml = """
podcast:
  title: "Test Podcast"
  description: "Desc"
  language: "zh-cn"
  author: "A"
  category: "Technology"
  explicit: false
  keep_last: 30

tts:
  backend: "edge-tts"
  voice: "zh-CN-XiaoxiaoNeural"
  rate: "+0%"
  volume: "+0%"
  pitch: "+0Hz"

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
    (tmp_path / "config" / "sources.yaml").write_text(
        "sources:\n  - name: X\n    url: https://x.com\n    enabled: true\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "episodes.json").write_text("[]", encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_full_run_with_audio_and_publish(full_config: Path, monkeypatch) -> None:
    """Run full pipeline including audio synthesis and site publishing."""
    root = full_config
    monkeypatch.setattr(
        run_daily_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "run_daily.py")
    )
    monkeypatch.setattr(
        sys, "argv", ["run_daily", "--date", "2024-03-15", "--base-url", "https://test.example.com"]
    )

    from ai_news_podcast.pipeline.fetcher import RawItem

    raw_item = RawItem(
        id="abc",
        title="Test News",
        link="https://a.com",
        normalized_link="https://a.com",
        source_name="S",
        source_category="news",
        published_at=datetime.now(tz=timezone.utc).isoformat(),
        summary="Summary.",
        full_text_snippet="Text." * 100,
        category="product",
        language="zh",
    )

    async def _fake_fetch_all(*args, **kwargs):
        return [raw_item]

    monkeypatch.setattr(run_daily_module, "fetch_all", _fake_fetch_all)

    fake_script = "[Host A] Hello\n[Host B] World"
    monkeypatch.setattr(run_daily_module, "generate_script", lambda *a, **kw: (fake_script, []))

    async def _fake_synthesize(*args, **kwargs) -> None:
        output = kwargs.get("output_path")
        if output:
            Path(output).write_text("fake mp3", encoding="utf-8")

    monkeypatch.setattr(run_daily_module, "synthesize", _fake_synthesize)

    rc = await run_daily_module.main()
    assert rc == 0

    episode_id = "2024-03-15"
    episodes_dir = root / "site" / "episodes"
    assert (episodes_dir / f"{episode_id}.txt").exists()
    assert (episodes_dir / f"{episode_id}.html").exists()
    assert (episodes_dir / f"{episode_id}.mp3").exists()

    # Verify site files
    assert (root / "site" / "feed.xml").exists()
    assert (root / "site" / "index.html").exists()

    # Verify episodes index
    import json

    eps = json.loads((root / "data" / "episodes.json").read_text(encoding="utf-8"))
    assert any(ep["id"] == episode_id for ep in eps)
    assert all(ep["enclosure_url"].startswith("https://test.example.com") for ep in eps)


@pytest.mark.asyncio
async def test_transcript_cleaning(full_config: Path, monkeypatch) -> None:
    """Verify transcript cleaning removes mood tags and literal backslash-n."""
    root = full_config
    monkeypatch.setattr(
        run_daily_module, "__file__", str(root / "src" / "ai_news_podcast" / "cli" / "run_daily.py")
    )
    monkeypatch.setattr(
        sys, "argv", ["run_daily", "--date", "2024-03-16", "--no-audio"]
    )

    from ai_news_podcast.pipeline.fetcher import RawItem

    raw_item = RawItem(
        id="abc",
        title="T",
        link="https://a.com",
        normalized_link="https://a.com",
        source_name="S",
        source_category="news",
        published_at=datetime.now(tz=timezone.utc).isoformat(),
        summary="S.",
        full_text_snippet="T." * 100,
        category="product",
        language="zh",
    )

    async def _fake_fetch(*args, **kwargs):
        return [raw_item]
    monkeypatch.setattr(run_daily_module, "fetch_all", _fake_fetch)

    # Script with mood tags, fact tags, and literal \n
    fake_script = "[mood:excited] [Host A] Hello\\n[FACT] world"
    monkeypatch.setattr(
        run_daily_module, "generate_script", lambda *a, **kw: (fake_script, ["test warning"])
    )
    async def _fake_synth(*args, **kwargs) -> None:
        pass
    monkeypatch.setattr(run_daily_module, "synthesize", _fake_synth)

    rc = await run_daily_module.main()
    assert rc == 0

    transcript = (root / "site" / "episodes" / "2024-03-16.txt").read_text(encoding="utf-8")
    assert "[mood:excited]" not in transcript
    assert "[FACT]" not in transcript
    assert "\\n" not in transcript
    assert "\n" in transcript  # literal backslash-n was converted to real newline
