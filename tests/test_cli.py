"""Tests for ai_news_podcast.cli.run_daily helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ai_news_podcast.cli.run_daily import (
    _coerce_episode_list,
    _episode_id,
    _get_base_url,
    _prune_episodes,
)


class TestGetBaseUrl:
    def test_cli_flag_wins(self, monkeypatch) -> None:
        monkeypatch.delenv("PODCAST_BASE_URL", raising=False)
        assert _get_base_url({}, "https://cli.example.com") == "https://cli.example.com"

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("PODCAST_BASE_URL", "https://env.example.com")
        assert _get_base_url({}, None) == "https://env.example.com"

    def test_config_value(self, monkeypatch) -> None:
        monkeypatch.delenv("PODCAST_BASE_URL", raising=False)
        cfg = {"podcast": {"base_url": "https://cfg.example.com"}}
        assert _get_base_url(cfg, None) == "https://cfg.example.com"

    def test_github_pages_user_repo(self, monkeypatch) -> None:
        monkeypatch.delenv("PODCAST_BASE_URL", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "alice")
        monkeypatch.setenv("GITHUB_REPOSITORY", "alice/ai-news-podcast")
        assert _get_base_url({}, None) == "https://alice.github.io/ai-news-podcast"

    def test_github_pages_user_site(self, monkeypatch) -> None:
        monkeypatch.delenv("PODCAST_BASE_URL", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "alice")
        monkeypatch.setenv("GITHUB_REPOSITORY", "alice/alice.github.io")
        assert _get_base_url({}, None) == "https://alice.github.io"

    def test_fallback_localhost(self, monkeypatch) -> None:
        monkeypatch.delenv("PODCAST_BASE_URL", raising=False)
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_REPOSITORY_OWNER", raising=False)
        assert _get_base_url({}, None) == "http://localhost"

    def test_strips_trailing_slash(self, monkeypatch) -> None:
        monkeypatch.setenv("PODCAST_BASE_URL", "https://example.com/")
        assert _get_base_url({}, None) == "https://example.com"


class TestEpisodeId:
    def test_format(self) -> None:
        dt = datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)
        assert _episode_id(dt) == "2024-03-15"


class TestCoerceEpisodeList:
    def test_list_of_dicts(self) -> None:
        assert _coerce_episode_list([{"id": "1"}, {"id": "2"}]) == [{"id": "1"}, {"id": "2"}]

    def test_filters_non_dicts(self) -> None:
        assert _coerce_episode_list([{"id": "1"}, "bad", 123, None]) == [{"id": "1"}]

    def test_non_list_returns_empty(self) -> None:
        assert _coerce_episode_list(None) == []
        assert _coerce_episode_list("string") == []
        assert _coerce_episode_list({}) == []


class TestPruneEpisodes:
    def test_keeps_last_n(self, tmp_path: Path) -> None:
        episodes = [
            {"id": "2024-03-01", "published_at_iso": "2024-03-01T00:00:00+00:00"},
            {"id": "2024-03-02", "published_at_iso": "2024-03-02T00:00:00+00:00"},
            {"id": "2024-03-03", "published_at_iso": "2024-03-03T00:00:00+00:00"},
        ]
        result = _prune_episodes(episodes, keep_last=2, episodes_dir=tmp_path)
        assert len(result) == 2
        assert result[0]["id"] == "2024-03-03"
        assert result[1]["id"] == "2024-03-02"

    def test_removes_old_files(self, tmp_path: Path) -> None:
        episodes = [
            {"id": "2024-03-01", "published_at_iso": "2024-03-01T00:00:00+00:00"},
            {"id": "2024-03-02", "published_at_iso": "2024-03-02T00:00:00+00:00"},
        ]
        # Create files for the older episode
        (tmp_path / "2024-03-01.mp3").write_text("mp3")
        (tmp_path / "2024-03-01.html").write_text("html")
        (tmp_path / "2024-03-01.txt").write_text("txt")

        result = _prune_episodes(episodes, keep_last=1, episodes_dir=tmp_path)

        assert len(result) == 1
        assert not (tmp_path / "2024-03-01.mp3").exists()
        assert not (tmp_path / "2024-03-01.html").exists()
        assert not (tmp_path / "2024-03-01.txt").exists()

    def test_empty_list(self, tmp_path: Path) -> None:
        assert _prune_episodes([], keep_last=5, episodes_dir=tmp_path) == []
