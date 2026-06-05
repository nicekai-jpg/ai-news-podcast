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
                if (
                    url_str.endswith(".mp3")
                    or url_str.endswith(".xml")
                    or url_str.endswith(".html")
                    or "/feed.xml" in url_str
                ):
                    continue
                norm = normalize_url(url_str)
                if norm:
                    urls.add(norm)
    except (OSError, ValueError) as e:
        log.error("Failed to parse historical broadcasted URLs: %s", e)
    return urls


@dataclass
class HistoricalRecord:
    text: str
    story_title: str
    episode_id: str


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


def get_recent_broadcasted_texts(
    episodes_json_path: Path,
    limit: int = 14,
    current_episode_id: str | None = None,
) -> list[HistoricalRecord]:
    """Extract previously broadcasted news texts (with keyword extraction) for semantic dedup."""
    import re

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
            ep_id = ep.get("id")
            brief_path = data_dir / "briefs" / f"brief_{ep_id}.json"
            brief_loaded = False
            if brief_path.exists():
                try:
                    brief = read_json(brief_path)
                    if brief and isinstance(brief, dict) and brief.get("stories"):
                        for story in brief["stories"]:
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

                            full_story_text = f"{title} {' '.join(summaries)} {keywords_str}".strip()
                            if full_story_text:
                                records.append(
                                    HistoricalRecord(
                                        text=full_story_text,
                                        story_title=title,
                                        episode_id=str(ep_id),
                                    )
                                )
                        brief_loaded = True
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    log.warning("Failed to load brief %s: %s", brief_path, e)

            if not brief_loaded:
                desc = ep.get("description", "")
                if desc:
                    matches = re.findall(r'<a\s+[^>]*href="[^"]+"[^>]*>([^<]+)</a>', desc)
                    for title_str in matches:
                        title_str = title_str.strip()
                        if title_str and len(title_str) > 3:
                            records.append(
                                HistoricalRecord(
                                    text=title_str,
                                    story_title=title_str,
                                    episode_id=str(ep_id),
                                )
                            )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.error("Failed to parse historical broadcasted texts: %s", e)
    return records


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
    dedup_details: list[dict[str, Any]] = []

    use_sentence_transformers = False
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer(semantic_model)
        use_sentence_transformers = True
        log.info("Successfully loaded SentenceTransformer model '%s' for semantic similarity.", semantic_model)
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("Sentence-Transformers loading failed (falling back to TF-IDF): %s", e)

    if use_sentence_transformers:
        sim_threshold = float(dedup_cfg.get("embedding_sim_threshold", 0.72))
        try:
            segmented_hist_texts = [r.text for r in recent_records]
            new_texts = [
                extract_semantic_keywords(it.title, it.summary, it.full_text_snippet, top_k=15)
                for it in raw_items
            ]

            hist_embeddings = model.encode(segmented_hist_texts, convert_to_tensor=True, show_progress_bar=False)
            new_embeddings = model.encode(new_texts, convert_to_tensor=True, show_progress_bar=False)

            cos_scores = util.cos_sim(new_embeddings, hist_embeddings)

            filtered_items = []
            skipped_count = 0
            for idx, item in enumerate(raw_items):
                max_sim = float(cos_scores[idx].max())
                if max_sim >= sim_threshold:
                    max_hist_idx = int(cos_scores[idx].argmax())
                    matched_rec = recent_records[max_hist_idx]

                    log.info(
                        "Semantic dedup (ST): filtered out '%s' (similarity %.2f >= %.2f with historical story '%s' from %s)",
                        item.title,
                        max_sim,
                        sim_threshold,
                        matched_rec.story_title,
                        matched_rec.episode_id,
                    )

                    dedup_details.append({
                        "title": item.title,
                        "link": item.link,
                        "source": item.source_name,
                        "reason": "cross_episode_semantic",
                        "similarity": round(max_sim, 4),
                        "threshold": sim_threshold,
                        "matched_story_title": matched_rec.story_title,
                        "matched_episode_id": matched_rec.episode_id,
                        "detail": f"SentenceTransformer similarity ({max_sim:.2f}) >= threshold ({sim_threshold:.2f}) with historical story '{matched_rec.story_title}' from {matched_rec.episode_id}."
                    })
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
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            log.error("Failed to perform Sentence-Transformers deduplication: %s (falling back to TF-IDF)", e)
            use_sentence_transformers = False

    # ── TF-IDF fallback ─────────────────────────────────────────────────
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sim_threshold = float(dedup_cfg.get("semantic_sim_threshold", 0.20))
    try:
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

        filtered_items = []
        skipped_count = 0
        for idx, item in enumerate(raw_items):
            max_sim = float(sim_matrix[idx].max())
            if max_sim >= sim_threshold:
                max_hist_idx = int(sim_matrix[idx].argmax())
                matched_rec = recent_records[max_hist_idx]

                log.info(
                    "Semantic dedup (TF-IDF): filtered out '%s' (similarity %.2f >= %.2f with historical story '%s' from %s)",
                    item.title,
                    max_sim,
                    sim_threshold,
                    matched_rec.story_title,
                    matched_rec.episode_id,
                )

                dedup_details.append({
                    "title": item.title,
                    "link": item.link,
                    "source": item.source_name,
                    "reason": "cross_episode_semantic",
                    "similarity": round(max_sim, 4),
                    "threshold": sim_threshold,
                    "matched_story_title": matched_rec.story_title,
                    "matched_episode_id": matched_rec.episode_id,
                    "detail": f"TF-IDF similarity ({max_sim:.2f}) >= threshold ({sim_threshold:.2f}) with historical story '{matched_rec.story_title}' from {matched_rec.episode_id}."
                })
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
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        log.error("Failed to perform TF-IDF semantic similarity deduplication: %s", e)
        return raw_items, dedup_details
