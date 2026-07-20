"""Cross-episode deduplication — URL-based and semantic similarity filtering.

Provides functions to extract previously broadcasted URLs and texts from
historical episodes, compute semantic similarity (SentenceTransformer with
TF-IDF fallback), and filter out duplicates from new items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_news_podcast.utils import read_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HistoricalRecord
# ---------------------------------------------------------------------------


@dataclass
class HistoricalRecord:
    text: str
    story_title: str
    episode_id: str


# ---------------------------------------------------------------------------
# URL dedup
# ---------------------------------------------------------------------------


def get_recent_broadcasted_urls(
    episodes_json_path: Path,
    limit: int = 14,
    current_episode_id: str | None = None,
) -> set[str]:
    """Extract previously broadcasted news URLs from recent episodes for cross-episode dedup."""
    import re

    from ai_news_podcast.pipeline.fetcher import normalize_url

    urls: set[str] = set()
    if not episodes_json_path.exists():
        return urls

    try:
        episodes = read_json(episodes_json_path)
        if not isinstance(episodes, list):
            return urls

        for ep in episodes[:limit]:
            ep_id = ep.get("id")
            if current_episode_id and str(ep_id) == str(current_episode_id):
                continue
            desc = ep.get("description", "")
            if not desc:
                continue
            found = re.findall(r'href="([^"]+)"', desc)
            for url in found:
                url_str = str(url).strip()
                if url_str.endswith((".mp3", ".xml", ".html")) or "/feed.xml" in url_str:
                    continue
                norm = normalize_url(url_str)
                if norm:
                    urls.add(norm)
    except (OSError, ValueError):
        log.exception("Failed to parse historical broadcasted URLs")
    return urls


# ---------------------------------------------------------------------------
# Semantic dedup helpers
# ---------------------------------------------------------------------------


def extract_semantic_keywords(title: str, summary: str, full_text: str, top_k: int = 15) -> str:
    """Extract core keywords from a news item and combine with title/summary for semantic comparison."""
    import jieba.analyse

    raw_text = f"{title} {summary} {full_text}".strip()
    if not raw_text:
        return ""

    try:
        keywords = jieba.analyse.extract_tags(raw_text, topK=top_k)
        return f"{title} {summary} {' '.join(keywords)}".strip()
    except (ValueError, TypeError, OSError) as e:
        log.warning("Failed to extract keywords using jieba: %s", e)
        return f"{title} {summary}".strip()


# ---------------------------------------------------------------------------
# Text extraction from historical episodes
# ---------------------------------------------------------------------------


def _extract_story_text(story: dict[str, Any]) -> str:
    """Extract combined text from a single story for semantic comparison."""
    title = story.get("representative_title", "")
    context = story.get("context", {})
    summaries = context.get("factual_summary", [])

    item_texts = []
    for item in story.get("items", []):
        if item.get("title"):
            item_texts.append(item["title"])
        if item.get("summary"):
            item_texts.append(item["summary"])
        if item.get("full_text_snippet"):
            item_texts.append(item["full_text_snippet"])

    combined_items_text = " ".join(item_texts)
    keywords_str = ""
    if combined_items_text:
        import jieba.analyse

        try:
            kws = jieba.analyse.extract_tags(combined_items_text, topK=15)
            keywords_str = " ".join(kws)
        except (ValueError, TypeError, OSError):
            pass

    return f"{title} {' '.join(summaries)} {keywords_str}".strip()


def _extract_from_brief(brief: dict[str, Any], ep_id: str) -> list[HistoricalRecord]:
    """Extract HistoricalRecords from a brief dict."""
    records: list[HistoricalRecord] = []
    if not isinstance(brief, dict) or not brief.get("stories"):
        return records

    for story in brief["stories"]:
        full_story_text = _extract_story_text(story)
        if full_story_text:
            title = story.get("representative_title", "")
            records.append(
                HistoricalRecord(
                    text=full_story_text,
                    story_title=title,
                    episode_id=ep_id,
                )
            )
    return records


def _extract_from_description(desc: str, ep_id: str) -> list[HistoricalRecord]:
    """Extract HistoricalRecords from episode description HTML."""
    import re

    records: list[HistoricalRecord] = []
    matches = re.findall(r'<a\s+[^>]*href="[^"]+"[^>]*>([^<]+)</a>', desc)
    for raw_title in matches:
        title_str = raw_title.strip()
        if title_str and len(title_str) > 3:
            records.append(
                HistoricalRecord(
                    text=title_str,
                    story_title=title_str,
                    episode_id=ep_id,
                )
            )
    return records


def get_recent_broadcasted_texts(
    episodes_json_path: Path,
    limit: int = 14,
    current_episode_id: str | None = None,
) -> list[HistoricalRecord]:
    """Extract previously broadcasted news texts (with keyword extraction) for semantic dedup."""
    records: list[HistoricalRecord] = []
    if not episodes_json_path.exists():
        return records

    try:
        episodes = read_json(episodes_json_path)
        if not isinstance(episodes, list):
            return records

        data_dir = episodes_json_path.parent
        for ep in episodes[:limit]:
            ep_id = ep.get("id")
            if current_episode_id and str(ep_id) == str(current_episode_id):
                continue

            brief_loaded = False
            brief_path = data_dir / "briefs" / f"brief_{ep_id}.json"
            if brief_path.exists():
                try:
                    brief = read_json(brief_path)
                    records.extend(_extract_from_brief(brief, str(ep_id)))
                    brief_loaded = True
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    log.warning("Failed to load brief %s: %s", brief_path, e)

            if not brief_loaded:
                desc = ep.get("description", "")
                if desc:
                    records.extend(_extract_from_description(desc, str(ep_id)))

    except (json.JSONDecodeError, KeyError, ValueError):
        log.exception("Failed to parse historical broadcasted texts")
    return records


# ---------------------------------------------------------------------------
# Semantic dedup core
# ---------------------------------------------------------------------------


def _extract_model_versions(text: str) -> set[tuple[str, str]]:
    """Extract (model_family, version) pairs from text, e.g., ('qwen', '3.8'), ('kimi', 'k3')."""
    import re

    text_lower = text.lower()

    family_map = {
        "千问": "qwen",
        "通义": "qwen",
        "qwen": "qwen",
        "kimi": "kimi",
        "月之暗面": "kimi",
        "deepseek": "deepseek",
        "claude": "claude",
        "gpt": "gpt",
        "gemini": "gemini",
        "llama": "llama",
        "glm": "glm",
        "智谱": "glm",
        "minimax": "minimax",
        "hunyuan": "hunyuan",
        "混元": "hunyuan",
        "doubao": "doubao",
        "豆包": "doubao",
        "gemma": "gemma",
        "smollm": "smollm",
        "mistral": "mistral",
    }

    found_pairs: set[tuple[str, str]] = set()
    for raw_key, family in family_map.items():
        if raw_key in text_lower:
            pattern = rf"(?:{raw_key})[\s\-_]*(?:v|k|r)?(\d+(?:\.\d+)*|k\d+(?:\.\d+)*|v\d+(?:\.\d+)*|r\d+(?:\.\d+)*)"
            matches = re.findall(pattern, text_lower)
            for m in matches:
                if m and len(m) >= 1:
                    found_pairs.add((family, m.strip("vk-_")))

    for raw_key, family in family_map.items():
        if raw_key in text_lower:
            gen_matches = re.findall(
                r"\b(?:v|k|r)?(\d+\.\d+|[2-9]b|\d+t|k[2-5](?:\.\d+)?|v[2-6](?:\.\d+)?)\b",
                text_lower,
            )
            for gm in gen_matches:
                clean_v = gm.strip("vk-_")
                if clean_v and clean_v not in ("2024", "2025", "2026"):
                    found_pairs.add((family, clean_v))

    return found_pairs


def _are_distinct_model_versions(new_text: str, hist_text: str) -> bool:
    """Check if new_text specifies a model version distinct from or missing in hist_text."""
    new_versions = _extract_model_versions(new_text)
    if not new_versions:
        return False

    hist_versions = _extract_model_versions(hist_text)
    hist_text_lower = hist_text.lower()

    for family, new_v in new_versions:
        fam_hist_versions = {v for f, v in hist_versions if f == family}
        if fam_hist_versions and new_v not in fam_hist_versions:
            log.info(
                "Model version divergence detected for '%s': new '%s' vs historical %s. Keeping item.",
                family,
                new_v,
                fam_hist_versions,
            )
            return True

        if new_v not in hist_text_lower:
            log.info(
                "Model version '%s' (%s) not found in historical match. Keeping item.",
                new_v,
                family,
            )
            return True

    return False


def _build_dedup_detail(
    item: Any,
    matched_rec: HistoricalRecord,
    max_sim: float,
    sim_threshold: float,
    method: str,
) -> dict[str, Any]:
    """Build a dedup detail dict for a removed item."""
    return {
        "title": item.title,
        "link": item.link,
        "source": item.source_name,
        "reason": "cross_episode_semantic",
        "similarity": round(max_sim, 4),
        "threshold": sim_threshold,
        "matched_story_title": matched_rec.story_title,
        "matched_episode_id": matched_rec.episode_id,
        "detail": (
            f"{method} similarity ({max_sim:.2f}) >= threshold ({sim_threshold:.2f}) "
            f"with historical story '{matched_rec.story_title}' from {matched_rec.episode_id}."
        ),
    }


def _dedup_with_sentence_transformers(
    raw_items: list[Any],
    recent_records: list[HistoricalRecord],
    dedup_cfg: dict[str, Any],
    model: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Deduplicate using SentenceTransformer embeddings."""
    from sentence_transformers import util

    sim_threshold = float(dedup_cfg.get("embedding_sim_threshold", 0.72))
    segmented_hist_texts = [r.text for r in recent_records]
    new_texts = [
        extract_semantic_keywords(it.title, it.summary, it.full_text_snippet, top_k=15)
        for it in raw_items
    ]

    hist_embeddings = model.encode(
        segmented_hist_texts, convert_to_tensor=True, show_progress_bar=False
    )
    new_embeddings = model.encode(new_texts, convert_to_tensor=True, show_progress_bar=False)

    cos_scores = util.cos_sim(new_embeddings, hist_embeddings)

    filtered_items: list[Any] = []
    dedup_details: list[dict[str, Any]] = []
    skipped_count = 0
    for idx, item in enumerate(raw_items):
        max_sim = float(cos_scores[idx].max())
        if max_sim >= sim_threshold:
            max_hist_idx = int(cos_scores[idx].argmax())
            matched_rec = recent_records[max_hist_idx]

            item_text = f"{item.title} {item.summary} {item.full_text_snippet}"
            if _are_distinct_model_versions(item_text, matched_rec.text):
                filtered_items.append(item)
                continue

            log.info(
                "Semantic dedup (ST): filtered out '%s' (similarity %.2f >= %.2f with historical story '%s' from %s)",
                item.title,
                max_sim,
                sim_threshold,
                matched_rec.story_title,
                matched_rec.episode_id,
            )

            dedup_details.append(
                _build_dedup_detail(
                    item, matched_rec, max_sim, sim_threshold, "SentenceTransformer"
                )
            )
            skipped_count += 1
        else:
            filtered_items.append(item)

    if skipped_count > 0:
        log.info(
            "Sentence-Transformers semantic dedup: filtered out %d items by threshold %.2f, %d items remaining",
            skipped_count,
            sim_threshold,
            len(filtered_items),
        )
    return filtered_items, dedup_details


def _dedup_with_tfidf(
    raw_items: list[Any],
    recent_records: list[HistoricalRecord],
    dedup_cfg: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Deduplicate using TF-IDF + cosine similarity fallback."""
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sim_threshold = float(dedup_cfg.get("semantic_sim_threshold", 0.20))

    def tokenize_text(text: str) -> str:
        return " ".join(jieba.cut(text))

    segmented_hist = [tokenize_text(r.text) for r in recent_records]
    new_texts = [
        extract_semantic_keywords(it.title, it.summary, it.full_text_snippet, top_k=15)
        for it in raw_items
    ]
    segmented_new = [tokenize_text(t) for t in new_texts]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=True,
    )
    all_texts = segmented_hist + segmented_new
    vectorizer.fit(all_texts)

    hist_tfidf = vectorizer.transform(segmented_hist)
    new_tfidf = vectorizer.transform(segmented_new)

    sim_matrix = cosine_similarity(new_tfidf, hist_tfidf)

    filtered_items: list[Any] = []
    dedup_details: list[dict[str, Any]] = []
    skipped_count = 0
    for idx, item in enumerate(raw_items):
        max_sim = float(sim_matrix[idx].max())
        if max_sim >= sim_threshold:
            max_hist_idx = int(sim_matrix[idx].argmax())
            matched_rec = recent_records[max_hist_idx]

            item_text = f"{item.title} {item.summary} {item.full_text_snippet}"
            if _are_distinct_model_versions(item_text, matched_rec.text):
                filtered_items.append(item)
                continue

            log.info(
                "Semantic dedup (TF-IDF): filtered out '%s' (similarity %.2f >= %.2f with historical story '%s' from %s)",
                item.title,
                max_sim,
                sim_threshold,
                matched_rec.story_title,
                matched_rec.episode_id,
            )

            dedup_details.append(
                _build_dedup_detail(item, matched_rec, max_sim, sim_threshold, "TF-IDF")
            )
            skipped_count += 1
        else:
            filtered_items.append(item)

    if skipped_count > 0:
        log.info(
            "TF-IDF semantic similarity dedup: filtered out %d items by similarity threshold %.2f, %d items remaining",
            skipped_count,
            sim_threshold,
            len(filtered_items),
        )
    return filtered_items, dedup_details


def semantic_dedup(
    raw_items: list[Any],
    recent_records: list[HistoricalRecord],
    dedup_cfg: dict[str, Any],
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Perform cross-episode semantic deduplication on new items.

    Tries SentenceTransformer first; falls back to TF-IDF if unavailable.

    Parameters
    ----------
    raw_items:
        List of RawItem objects to filter.
    recent_records:
        Historical broadcasted records for comparison.
    dedup_cfg:
        Dedup configuration dict (from config.yaml processing.dedup).
    semantic_model:
        SentenceTransformer model name.

    Returns
    -------
    tuple[list[Any], list[dict[str, Any]]]
        (filtered_items, dedup_details) — items that survived dedup, and
        details about each removed item.
    """
    model = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(semantic_model)
        log.info(
            "Successfully loaded SentenceTransformer model '%s' for semantic similarity.",
            semantic_model,
        )
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("Sentence-Transformers loading failed (falling back to TF-IDF): %s", e)

    if model is not None:
        try:
            return _dedup_with_sentence_transformers(raw_items, recent_records, dedup_cfg, model)
        except (ImportError, OSError, ValueError, RuntimeError):
            log.exception(
                "Failed to perform Sentence-Transformers deduplication (falling back to TF-IDF)",
            )

    try:
        return _dedup_with_tfidf(raw_items, recent_records, dedup_cfg)
    except (ImportError, OSError, ValueError, RuntimeError):
        log.exception("Failed to perform TF-IDF semantic similarity deduplication")
        return raw_items, []
