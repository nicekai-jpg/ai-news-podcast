"""Stage 2 — 信息加工模块

职责：RawItem 列表 → 三层去重 → 聚类 → context_block → 五维打分 →
      角色分配 → Thesis 提炼 → 输出 episode_brief dict。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from fetcher import RawItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 输出数据结构
# ---------------------------------------------------------------------------


@dataclass
class ContextBlock:
    factual_summary: list[str]  # 3 句事实摘要
    historical_background: str
    sources_ranked: list[dict[str, Any]]  # [{name, authority, link}]


@dataclass
class ScoredStory:
    """加工后的故事，用于输出 episode_brief。"""

    cluster_id: int
    representative_title: str
    items: list[dict[str, Any]]  # 原始 RawItem dicts
    context: ContextBlock
    scores: dict[str, int]  # 五维得分
    total_score: int
    role: str  # main / supporting / quick / skip
    role_emoji: str


# ---------------------------------------------------------------------------
# 第一层去重：URL 硬去重
# ---------------------------------------------------------------------------


def _dedup_url(items: list[RawItem]) -> list[RawItem]:
    """基于 normalized_link 去重。"""
    seen: set[str] = set()
    out: list[RawItem] = []
    for item in items:
        key = item.normalized_link
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    logger.info("URL dedup: %d → %d", len(items), len(out))
    return out


# ---------------------------------------------------------------------------
# 第二层去重：rapidfuzz 标题软去重
# ---------------------------------------------------------------------------


def _dedup_title_fuzzy(
    items: list[RawItem],
    *,
    threshold: int = 92,
    window_hours: int = 48,
) -> list[RawItem]:
    """token_set_ratio ≥ threshold 且时间差 ≤ window_hours 时视为重复。"""
    out: list[RawItem] = []
    for item in items:
        is_dup = False
        item_dt = datetime.fromisoformat(item.published_at)
        for kept in out:
            kept_dt = datetime.fromisoformat(kept.published_at)
            if abs((item_dt - kept_dt).total_seconds()) > window_hours * 3600:
                continue
            ratio = fuzz.token_set_ratio(item.title, kept.title)
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(item)
    logger.info("Fuzzy title dedup: %d → %d", len(items), len(out))
    return out


# ---------------------------------------------------------------------------
# 第三层去重：jieba 关键词重叠
# ---------------------------------------------------------------------------


def _extract_keywords(text: str, topk: int = 10) -> set[str]:
    """提取 jieba 关键词集合。"""
    try:
        import jieba.analyse

        keywords = jieba.analyse.extract_tags(text, topK=topk)
        return set(keywords)
    except Exception:
        words = set(jieba.cut(text))
        return {w for w in words if len(w) >= 2}


def _dedup_keyword_overlap(
    items: list[RawItem],
    *,
    overlap_threshold: float = 0.35,
    title_sim_threshold: int = 85,
) -> list[RawItem]:
    """jieba 关键词重叠 ≥ overlap_threshold 且标题相似度 ≥ title_sim 视为重复。"""
    out: list[RawItem] = []
    kw_cache: list[set[str]] = []
    for item in items:
        text = f"{item.title} {item.summary}"
        kws = _extract_keywords(text)
        is_dup = False
        for i, kept in enumerate(out):
            # 标题相似度门槛
            title_ratio = fuzz.token_set_ratio(item.title, kept.title)
            if title_ratio < title_sim_threshold:
                continue
            # 关键词重叠
            kept_kws = kw_cache[i]
            if not kws or not kept_kws:
                continue
            overlap = len(kws & kept_kws) / min(len(kws), len(kept_kws))
            if overlap >= overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(item)
            kw_cache.append(kws)
    logger.info("Keyword overlap dedup: %d → %d", len(items), len(out))
    return out


def dedup_pipeline(
    items: list[RawItem],
    *,
    rapidfuzz_threshold: int = 92,
    jieba_overlap: float = 0.35,
    title_sim: int = 85,
    window_hours: int = 48,
) -> list[RawItem]:
    """三层去重管线。"""
    items = _dedup_url(items)
    items = _dedup_title_fuzzy(
        items, threshold=rapidfuzz_threshold, window_hours=window_hours
    )
    items = _dedup_keyword_overlap(
        items, overlap_threshold=jieba_overlap, title_sim_threshold=title_sim
    )
    return items


# ---------------------------------------------------------------------------
# 聚类：TF-IDF + DBSCAN
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    cluster_id: int
    items: list[RawItem]
    representative: RawItem  # 标题最长 / 全文最丰富的代表


def cluster_stories(
    items: list[RawItem],
    *,
    ngram_range: tuple[int, int] = (2, 4),
    eps: float = 0.35,
    min_samples: int = 2,
) -> list[Cluster]:
    """TF-IDF char n-gram + DBSCAN(cosine) 聚类。"""
    if len(items) < 2:
        return (
            [Cluster(cluster_id=0, items=items, representative=items[0])]
            if items
            else []
        )

    texts = [f"{it.title} {it.summary}" for it in items]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=10000,
    )
    tfidf = vectorizer.fit_transform(texts)
    dist = cosine_distances(tfidf)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist)

    clusters_map: dict[int, list[int]] = {}
    noise_id = max(labels) + 1 if len(labels) > 0 else 0
    for idx, label in enumerate(labels):
        if label == -1:
            # 噪声点各成一簇
            clusters_map[noise_id] = [idx]
            noise_id += 1
        else:
            clusters_map.setdefault(label, []).append(idx)

    clusters: list[Cluster] = []
    for cid, indices in sorted(clusters_map.items()):
        cluster_items = [items[i] for i in indices]
        # 选全文最长的作为代表
        representative = max(cluster_items, key=lambda x: len(x.full_text_snippet))
        clusters.append(
            Cluster(cluster_id=cid, items=cluster_items, representative=representative)
        )

    logger.info("Clustered %d items → %d clusters", len(items), len(clusters))
    return clusters


# ---------------------------------------------------------------------------
# context_block 增补
# ---------------------------------------------------------------------------

_AUTHORITY_ORDER = {
    "official": 1,
    "research": 2,
    "news": 3,
    "product": 3,
    "analysis": 3,
    "tools": 4,
    "events": 4,
    "other": 5,
}


def _build_context(
    cluster: Cluster,
    *,
    story_memory: dict[str, Any] | None = None,
    source_authority: dict[str, int] | None = None,
) -> ContextBlock:
    """构建 context_block。"""
    auth_map = source_authority or _AUTHORITY_ORDER
    rep = cluster.representative

    # 3 句事实摘要 — 取前 3 个 item 的 summary 首句
    summaries: list[str] = []
    for item in cluster.items[:3]:
        text = item.summary or item.full_text_snippet
        if text:
            first_sentence = re.split(r"[。.！!？?\n]", text)[0].strip()
            if first_sentence:
                summaries.append(first_sentence)
    if not summaries:
        summaries = [rep.title]

    # 历史背景 — 简单占位（后续可从 story_memory 拉取）
    background = ""
    if story_memory:
        # 搜索 story_memory 中相关条目
        for mem_key, mem_val in story_memory.items():
            if any(kw in rep.title for kw in mem_key.split()):
                background = str(mem_val)
                break

    # 来源排序
    sources_ranked: list[dict[str, Any]] = []
    for item in cluster.items:
        auth = auth_map.get(item.source_category, 5)
        sources_ranked.append(
            {
                "name": item.source_name,
                "authority": auth,
                "link": item.link,
            }
        )
    sources_ranked.sort(key=lambda x: x["authority"])

    return ContextBlock(
        factual_summary=summaries,
        historical_background=background,
        sources_ranked=sources_ranked,
    )


# ---------------------------------------------------------------------------
# 五维打分 (PLAN §2.4)
# ---------------------------------------------------------------------------


def _score_cluster(cluster: Cluster) -> dict[str, int]:
    """
    五维评分，每维 1-3 分，总分 5-15。
    - impact_scope: 多源报道 → 3，2源 → 2，单源 → 1
    - novelty: 含新产品/论文关键词 → 3
    - explainability: 全文丰富度
    - listener_relevance: 中文内容 + AI 强相关 → 3
    - source_richness: 来源权威性
    """
    items = cluster.items
    rep = cluster.representative
    text = f"{rep.title} {rep.summary} {rep.full_text_snippet}".lower()

    # impact_scope
    unique_sources = len({it.source_name for it in items})
    impact = 3 if unique_sources >= 3 else (2 if unique_sources >= 2 else 1)

    # novelty
    novel_kws = (
        "首次",
        "突破",
        "创新",
        "first",
        "novel",
        "breakthrough",
        "state-of-the-art",
        "新发布",
        "新模型",
        "launch",
        "release",
        "开源",
    )
    novelty = (
        3
        if any(k in text for k in novel_kws)
        else (2 if any(k in text for k in ("更新", "update", "升级")) else 1)
    )

    # explainability
    fulltext_len = len(rep.full_text_snippet)
    explain = 3 if fulltext_len >= 1200 else (2 if fulltext_len >= 600 else 1)

    # listener_relevance
    zh_count = sum(1 for it in items if it.language == "zh")
    relevance_kws = ("大模型", "llm", "chatgpt", "claude", "gpt", "agent", "智能体")
    has_relevance = any(k in text for k in relevance_kws)
    relevance = (
        3
        if (zh_count > 0 and has_relevance)
        else (2 if has_relevance or zh_count > 0 else 1)
    )

    # source_richness
    auth_scores = [_AUTHORITY_ORDER.get(it.source_category, 5) for it in items]
    min_auth = min(auth_scores) if auth_scores else 5
    richness = 3 if min_auth <= 1 else (2 if min_auth <= 2 else 1)

    return {
        "impact_scope": impact,
        "novelty": novelty,
        "explainability": explain,
        "listener_relevance": relevance,
        "source_richness": richness,
    }


# ---------------------------------------------------------------------------
# 角色分配 (PLAN §2.5)
# ---------------------------------------------------------------------------


def _assign_role(total: int, thresholds: dict[str, Any]) -> tuple[str, str]:
    """
    12-15 → 🔴main  |  8-11 → 🟡supporting  |  5-7 → 🟢quick  |  <5 → ⚪skip
    """
    main_range = thresholds.get("main", [12, 15])
    supporting_range = thresholds.get("supporting", [8, 11])
    quick_range = thresholds.get("quick", [5, 7])
    skip_below = thresholds.get("skip_below", 5)

    if total >= main_range[0]:
        return "main", "🔴"
    elif total >= supporting_range[0]:
        return "supporting", "🟡"
    elif total >= quick_range[0]:
        return "quick", "🟢"
    else:
        return "skip", "⚪"


# ---------------------------------------------------------------------------
# Thesis 提炼 (PLAN §2.6)
# ---------------------------------------------------------------------------

_THESIS_TEMPLATES = [
    "今天的AI领域，{main_topic}正在重塑行业格局",
    "从{main_topic}到{sub_topic}，AI技术持续加速演进",
    "{main_topic}——这可能是本周最值得关注的AI趋势",
    "当{main_topic}遇上实际落地，我们看到了什么",
    "围绕{main_topic}，多方力量正在汇聚",
]


def _extract_thesis(scored_stories: list[ScoredStory]) -> str:
    """从主故事和支撑故事中提炼一句主线论点。"""
    main_stories = [s for s in scored_stories if s.role == "main"]
    supporting = [s for s in scored_stories if s.role == "supporting"]

    if not main_stories:
        return "今天的AI领域动态丰富，多个方向齐头并进"

    main_topic = main_stories[0].representative_title
    # 简化：取前 15 字作为话题
    if len(main_topic) > 15:
        main_topic = main_topic[:15] + "..."

    sub_topic = ""
    if supporting:
        sub_topic = supporting[0].representative_title
        if len(sub_topic) > 12:
            sub_topic = sub_topic[:12] + "..."

    import random

    template = random.choice(_THESIS_TEMPLATES)
    return template.format(main_topic=main_topic, sub_topic=sub_topic or "产业应用")


# ---------------------------------------------------------------------------
# 主入口 — 加工管线
# ---------------------------------------------------------------------------


def process(
    raw_items: list[RawItem],
    *,
    processing_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    完整加工管线。返回 episode_brief dict。

    Parameters
    ----------
    raw_items : 来自 fetcher 的原始条目
    processing_cfg : config.yaml 中的 processing 段
    """
    cfg = processing_cfg or {}
    dedup_cfg = cfg.get("dedup", {})
    cluster_cfg = cfg.get("clustering", {})
    scoring_cfg = cfg.get("scoring", {})
    authority_cfg = cfg.get("source_authority", _AUTHORITY_ORDER)

    # 1) 三层去重
    deduped = dedup_pipeline(
        raw_items,
        rapidfuzz_threshold=int(dedup_cfg.get("rapidfuzz_threshold", 92)),
        jieba_overlap=float(dedup_cfg.get("jieba_keyword_overlap", 0.35)),
        title_sim=int(dedup_cfg.get("title_sim_threshold", 85)),
        window_hours=int(dedup_cfg.get("dedup_window_hours", 48)),
    )

    if not deduped:
        logger.warning("No items after dedup")
        return {
            "stories": [],
            "thesis": "",
            "metadata": {"total_raw": len(raw_items), "total_deduped": 0},
        }

    # 2) 聚类
    ngram = tuple(cluster_cfg.get("ngram_range", [2, 4]))
    clusters = cluster_stories(
        deduped,
        ngram_range=ngram,  # type: ignore[arg-type]
        eps=float(cluster_cfg.get("eps", 0.35)),
        min_samples=int(cluster_cfg.get("min_samples", 2)),
    )

    # 3) 为每个 cluster 构建 context + 打分 + 角色分配
    role_thresholds = scoring_cfg.get(
        "role_thresholds",
        {
            "main": [12, 15],
            "supporting": [8, 11],
            "quick": [5, 7],
            "skip_below": 5,
        },
    )

    scored_stories: list[ScoredStory] = []
    for cluster in clusters:
        context = _build_context(cluster, source_authority=authority_cfg)
        scores = _score_cluster(cluster)
        total = sum(scores.values())
        role, emoji = _assign_role(total, role_thresholds)

        scored_stories.append(
            ScoredStory(
                cluster_id=cluster.cluster_id,
                representative_title=cluster.representative.title,
                items=[item.to_dict() for item in cluster.items],
                context=context,
                scores=scores,
                total_score=total,
                role=role,
                role_emoji=emoji,
            )
        )

    # 按总分降序排列
    scored_stories.sort(key=lambda x: x.total_score, reverse=True)

    # 4) Thesis 提炼
    thesis = _extract_thesis(scored_stories)

    # 5) 组装 episode_brief
    brief: dict[str, Any] = {
        "thesis": thesis,
        "stories": [],
        "metadata": {
            "total_raw": len(raw_items),
            "total_deduped": len(deduped),
            "total_clusters": len(clusters),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    }

    for ss in scored_stories:
        brief["stories"].append(
            {
                "cluster_id": ss.cluster_id,
                "representative_title": ss.representative_title,
                "role": ss.role,
                "role_emoji": ss.role_emoji,
                "total_score": ss.total_score,
                "scores": ss.scores,
                "context": {
                    "factual_summary": ss.context.factual_summary,
                    "historical_background": ss.context.historical_background,
                    "sources_ranked": ss.context.sources_ranked,
                },
                "items": ss.items,
            }
        )

    return brief


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy int/float types that stdlib json cannot serialize."""

    def default(self, o: Any) -> Any:
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return super().default(o)


def save_brief(brief: dict[str, Any], path: Path) -> None:
    """保存 episode_brief.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(brief, f, ensure_ascii=False, indent=2, cls=_NumpySafeEncoder)
        f.write("\n")
    logger.info("Saved episode_brief to %s", path)
