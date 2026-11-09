"""Material selection strategy protocol — Strategy Pattern.

All material selection strategies must implement this protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

# 时效分档：24 小时内不罚分，1-2 天扣 1 分，超过 2 天扣 3 分。
# 分数上限 15，扣 3 分足以让「三天前的旧闻」低于任何新鲜素材，同时不至于让重大
# 事件(14-15 分)直接沉底。
_FRESH_HOURS = 24.0
_RECENT_HOURS = 48.0


class MaterialSelectionStrategy(Protocol):
    """Protocol for material selection strategies."""

    def select(
        self,
        stories: list[dict],
        max_stories: int,
        reference_time: datetime | None = None,
    ) -> list[dict]:
        """Select stories from the brief.

        Args:
            stories: List of story dicts from episode_brief.
            max_stories: Maximum number of stories to select.
            reference_time: 计算素材时效的基准时间(通常是 brief 生成时间)。

        Returns:
            Selected stories in priority order.
        """
        ...


def story_published_at(story: dict) -> datetime | None:
    """返回一条素材中最新一条原始新闻的发布时间(UTC)；无发布时间返回 None。"""
    latest: datetime | None = None
    for item in story.get("items") or []:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("published_at") or "").strip()
        if not raw:
            continue
        try:
            published = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        if latest is None or published > latest:
            latest = published
    return latest


def story_age_hours(story: dict, reference_time: datetime | None) -> float | None:
    """素材距今小时数；无法确定时返回 None。"""
    published = story_published_at(story)
    if published is None or reference_time is None:
        return None
    reference = (
        reference_time if reference_time.tzinfo is not None else reference_time.replace(tzinfo=UTC)
    )
    return max(0.0, (reference - published).total_seconds() / 3600.0)


def freshness_penalty(story: dict, reference_time: datetime | None) -> int:
    """按素材时效返回扣分(0/1/3)，让旧闻在素材选择时让位给新事件。"""
    age = story_age_hours(story, reference_time)
    if age is None:
        return 0
    if age <= _FRESH_HOURS:
        return 0
    if age <= _RECENT_HOURS:
        return 1
    return 3


def effective_score(story: dict, reference_time: datetime | None) -> int:
    """素材排序分 = 五维总分 - 时效扣分。"""
    try:
        base = int(story.get("total_score", 0) or 0)
    except (TypeError, ValueError):
        base = 0
    return base - freshness_penalty(story, reference_time)
