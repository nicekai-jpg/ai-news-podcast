"""Stage 2「新闻潮汐」：去重 / 聚类 / 连续性 / 标注 / 证据 → `data/ocean/{date}/`。"""

from ai_news_podcast.business.ocean.build import OceanSnapshot, build_ocean
from ai_news_podcast.business.ocean.config import load_ocean_config
from ai_news_podcast.business.ocean.dedupe import DedupResult, dedupe_items
from ai_news_podcast.business.ocean.evidence import EvidenceResult, score_event
from ai_news_podcast.business.ocean.labels import (
    LabeledItem,
    Lexicon,
    label_item,
    load_lexicon,
    merge_labels,
)

__all__ = [
    "DedupResult",
    "EvidenceResult",
    "LabeledItem",
    "Lexicon",
    "OceanSnapshot",
    "build_ocean",
    "dedupe_items",
    "label_item",
    "load_lexicon",
    "load_ocean_config",
    "merge_labels",
    "score_event",
]
