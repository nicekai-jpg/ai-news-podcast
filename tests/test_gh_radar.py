"""Tests for ai_news_podcast.pipeline.gh_radar 与相关配置。"""

from __future__ import annotations

from ai_news_podcast.config.models import AppConfig


class TestGhRadarConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig.from_dict({})
        assert cfg.gh_radar.enabled is True
        assert cfg.gh_radar.min_stars == 500
        assert cfg.gh_radar.pick_count == 1
        assert cfg.gh_radar.runner_up_count == 2
        assert cfg.gh_radar.repeat_window_days == 30
        assert cfg.gh_radar.snapshot_dir == "gh_snapshots"

    def test_overrides(self) -> None:
        cfg = AppConfig.from_dict(
            {
                "gh_radar": {
                    "min_stars": 100,
                    "preferred_topics": ["rag"],
                    "pick_count": 2,
                }
            }
        )
        assert cfg.gh_radar.min_stars == 100
        assert cfg.gh_radar.preferred_topics == ["rag"]
        assert cfg.gh_radar.pick_count == 2
