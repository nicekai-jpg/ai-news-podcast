"""publish 控制器纯函数测试：重发同一期不能重置 pubDate。"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_news_podcast.presentation.cli.episode_utils_controller import coerce_episode_list
from ai_news_podcast.presentation.cli.podcast_publish_controller import (
    _resolve_episode_timestamps,
)


def test_republish_keeps_previous_timestamps() -> None:
    previous = {
        "id": "2026-10-01",
        "pubDate": "Wed, 01 Oct 2026 00:00:00 +0000",
        "published_at_iso": "2026-10-01T00:00:00+00:00",
    }
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    published_at_iso, pub_date_rfc = _resolve_episode_timestamps(previous, now)
    assert published_at_iso == "2026-10-01T00:00:00+00:00"
    assert pub_date_rfc == "Wed, 01 Oct 2026 00:00:00 +0000"


def test_new_episode_uses_now() -> None:
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    published_at_iso, pub_date_rfc = _resolve_episode_timestamps(None, now)
    assert published_at_iso == now.isoformat()
    assert "05 Nov 2026" in pub_date_rfc


def test_coerce_episode_list_accepts_dict_wrapper() -> None:
    """episodes.json 若被写成 {"episodes": [...]}，publish 不能把它当成空列表清空历史。"""
    payload = {"episodes": [{"id": "2026-11-05"}, {"id": "2026-11-04"}]}
    assert [ep["id"] for ep in coerce_episode_list(payload)] == ["2026-11-05", "2026-11-04"]
