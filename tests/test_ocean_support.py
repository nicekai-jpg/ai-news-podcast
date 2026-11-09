"""Stage 2 ocean 测试的共享构造器（不联网）。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from ai_news_podcast.business.contracts import Event, Item, Labels, make_item_id
from ai_news_podcast.business.contracts.artifact import write_artifact
from ai_news_podcast.business.ocean.config import load_ocean_config
from ai_news_podcast.business.ocean.labels import Lexicon, load_lexicon

ROOT = Path(__file__).resolve().parents[1]
OCEAN_CONFIG_PATH = ROOT / "config" / "ocean_v3.yaml"
NLP_DIR = ROOT / "config" / "nlp"
DEFAULT_PUBLISHED = "2026-11-09T08:00:00+00:00"


def ocean_config() -> dict:
    return load_ocean_config(OCEAN_CONFIG_PATH)


def ocean_lexicon() -> Lexicon:
    return load_lexicon(NLP_DIR)


def make_item(
    *,
    title: str,
    item_id: str | None = None,
    link: str | None = None,
    source: str = "source_a",
    source_label: str | None = None,
    category: str = "news",
    summary: str = "",
    full_text: str = "",
    published_at: str = DEFAULT_PUBLISHED,
    channel: str = "c1_rss",
    source_group: str = "",
    full_text_status: str = "ok",
) -> Item:
    link = link if link is not None else f"https://example.com/{source}/{abs(hash(title)) % 100000}"
    return Item(
        item_id=item_id or make_item_id(link, title, source),
        channel=channel,
        source=source,
        source_label=source_label or source,
        category=category,
        title=title,
        link=link,
        summary=summary,
        full_text=full_text,
        full_text_status=full_text_status,
        published_at=published_at,
        fetched_at=published_at,
        extra={"source_group": source_group} if source_group else {},
    )


def make_labels(
    *,
    subjects: tuple[tuple[str, float], ...] = (("openai", 1.0),),
    actions: tuple[str, ...] = (),
    numbers: tuple[dict, ...] = (),
    lifecycle: str = "unknown",
    evidence_tier: str = "unknown",
) -> Labels:
    return Labels(
        subjects=[
            {
                "id": subject_id,
                "name": subject_id.upper(),
                "type": "company",
                "weight": weight,
                "confidence": 1.0,
            }
            for subject_id, weight in subjects
        ],
        actions=list(actions),
        primary_action=actions[0] if actions else "",
        lifecycle=lifecycle,
        evidence_tier=evidence_tier,
        novelty_numbers=[dict(number) for number in numbers],
    )


def make_event(
    title: str,
    *,
    summary: str = "",
    labels: Labels | None = None,
    event_id: str | None = None,
    sequence: int = 1,
    continuity_id: str = "",
    covered_before: bool = False,
    source_count: int = 1,
    first_seen_at: str = DEFAULT_PUBLISHED,
    last_update_at: str | None = None,
    item_ids: list[str] | None = None,
) -> Event:
    return Event(
        event_id=event_id or "e_" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:16],
        representative_title=title,
        summary=summary,
        item_ids=item_ids or [],
        source_count=source_count,
        independent_sources=source_count,
        first_seen_at=first_seen_at,
        last_update_at=last_update_at or first_seen_at,
        continuity_id=continuity_id,
        sequence=sequence,
        covered_before=covered_before,
        labels=labels or make_labels(),
    )


def utc_iso(date_str: str, hour: int = 8) -> str:
    moment = datetime.fromisoformat(f"{date_str}T00:00:00").replace(hour=hour, tzinfo=UTC)
    return moment.isoformat()


def write_raw_artifact(path: Path, items: list[Item], date_str: str) -> Path:
    return write_artifact(path, items, date_str=date_str)
