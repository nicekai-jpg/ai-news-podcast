"""Stage 3 选题推荐：多路召回 → 规则粗排 → 精排（规则 + LLM rubric）→ MMR 重排。"""

from __future__ import annotations

from ai_news_podcast.business.recommend.coarse import (
    CoarseCandidate,
    CoarseElimination,
    CoarseOutcome,
    coarse_rank,
)
from ai_news_podcast.business.recommend.config import RecommendConfig
from ai_news_podcast.business.recommend.fine import (
    FineScore,
    LLMJudge,
    apply_judgements,
    score_events,
)
from ai_news_podcast.business.recommend.prompts import SYSTEM_PROMPT, build_rubric_prompt
from ai_news_podcast.business.recommend.recall import (
    RECALL_CHANNELS,
    RecallCandidate,
    recall_events,
)
from ai_news_podcast.business.recommend.rerank import RerankedItem, rerank
from ai_news_podcast.business.recommend.service import (
    RecommendationResult,
    load_events,
    recommend,
    recommend_with_audit,
)

__all__ = [
    "RECALL_CHANNELS",
    "SYSTEM_PROMPT",
    "CoarseCandidate",
    "CoarseElimination",
    "CoarseOutcome",
    "FineScore",
    "LLMJudge",
    "RecallCandidate",
    "RecommendConfig",
    "RecommendationResult",
    "RerankedItem",
    "apply_judgements",
    "build_rubric_prompt",
    "coarse_rank",
    "load_events",
    "recall_events",
    "recommend",
    "recommend_with_audit",
    "rerank",
    "score_events",
]
