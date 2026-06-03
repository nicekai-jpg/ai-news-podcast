"""Pipeline runner — 数据基础层。

职责：抓取 → 三层去重 → DBSCAN 聚类 → 五维打分 → 保存 brief JSON。

上层业务（播客、科技日报等）均通过 run_pipeline() 获取当日的 episode_brief，
不允许直接调用 fetch_all() 或 process()。

使用示例::

    brief = await run_pipeline(cfg, sources, date_str="2026-06-02", data_dir=Path("data"))
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process, save_brief
from ai_news_podcast.utils import read_json

log = logging.getLogger(__name__)


def get_recent_broadcasted_urls(episodes_json_path: Path, limit: int = 14) -> set[str]:
    """提取最近若干期节目中已经播报过的新闻 URL，用于跨期去重。"""
    import re
    from ai_news_podcast.pipeline.fetcher import normalize_url

    urls: set[str] = set()
    if not episodes_json_path.exists():
        return urls

    try:
        episodes = read_json(episodes_json_path)
        if not isinstance(episodes, list):
            return urls

        # 只遍历最近的 limit 期节目，避免过度过滤历史主题
        for ep in episodes[:limit]:
            desc = ep.get("description", "")
            if not desc:
                continue
            # 匹配 HTML 中的 href 链接
            found = re.findall(r'href="([^"]+)"', desc)
            for url in found:
                url_str = str(url).strip()
                # 过滤掉 mp3 音频下载链接、以及 feed 订阅和主页链接
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
    except Exception as e:
        log.error("Failed to parse historical broadcasted URLs: %s", e)
    return urls


@dataclass
class HistoricalRecord:
    text: str
    story_title: str
    episode_id: str


def extract_semantic_keywords(title: str, summary: str, full_text: str, top_k: int = 15) -> str:
    """提取新闻的核心关键词并与标题、摘要组合，形成精简但覆盖全面的语义特征文本。"""
    import jieba.analyse

    raw_text = f"{title} {summary} {full_text}".strip()
    if not raw_text:
        return ""

    try:
        keywords = jieba.analyse.extract_tags(raw_text, topK=top_k)
        return f"{title} {summary} {' '.join(keywords)}".strip()
    except Exception as e:
        log.warning("Failed to extract keywords using jieba: %s", e)
        return f"{title} {summary}".strip()


def get_recent_broadcasted_texts(episodes_json_path: Path, limit: int = 14) -> list[HistoricalRecord]:
    """提取最近若干期节目中已播报的新闻文本（使用关键词提取进行精简），用于更全面的跨期语义去重。"""
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
            # 优先从 data/briefs/brief_{ep_id}.json 中读取更完整的“标题+摘要+精简关键词”作为语义特征
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
                            
                            # 提取该 Story 下所有新闻 item 的 title, summary 以及 full_text_snippet
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
                                except Exception:
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
                except Exception as e:
                    log.warning("Failed to load brief %s: %s", brief_path, e)

            # 兜底方案：若无 brief 缓存，则解析 episodes.json 里的描述 HTML 锚文本（仅标题）
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
    except Exception as e:
        log.error("Failed to parse historical broadcasted texts: %s", e)
    return records


async def run_pipeline(
    cfg: dict[str, Any],
    sources: list[Any],
    date_str: str,
    data_dir: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """执行数据基础管线，返回当日 episode_brief。

    Parameters
    ----------
    cfg:
        完整的 config.yaml 内容（dict）。
    sources:
        由 load_sources() 读取的信源列表。
    date_str:
        日期字符串，格式 ``YYYY-MM-DD``，用于命名 brief 文件。
    data_dir:
        数据目录（例如 ``<root>/data``），brief JSON 保存于此。
    force_refresh:
        若为 True，则忽略已有 brief 文件，强制重新抓取并处理。

    Returns
    -------
    dict
        episode_brief 字典，包含 ``thesis``、``stories``、``metadata`` 等字段。
    """
    brief_path = data_dir / "briefs" / f"brief_{date_str}.json"

    # ── 复用已有 brief ──────────────────────────────────────────────────────
    if brief_path.exists() and not force_refresh:
        log.info("Found existing brief at %s — skipping fetch and process", brief_path)
        brief = read_json(brief_path)
        if brief and isinstance(brief, dict) and brief.get("stories") is not None:
            log.info(
                "Loaded brief: %d stories (main=%d, supporting=%d, quick=%d)",
                len(brief.get("stories", [])),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
                sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
            )
            return brief
        log.warning("Existing brief at %s appears invalid, re-running pipeline", brief_path)

    # ── Stage 1: 抓取 ───────────────────────────────────────────────────────
    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(processing_cfg.get("max_pages", 80))

    log.info("Stage 1 [pipeline]: fetching RSS feeds …")
    raw_items = await fetch_all(
        sources,
        timeout_seconds=timeout_seconds,
        connect_timeout=connect_timeout,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
        max_pages=max_pages,
    )

    if not raw_items:
        log.warning("No items fetched — pipeline aborted")
        return {"stories": [], "thesis": "", "metadata": {"total_raw": 0, "error": "no_items"}}

    log.info("Stage 1 [pipeline]: fetched %d raw items", len(raw_items))

    # ── 跨期历史去重 ──────────────────────────────────────────────────────────
    dedup_details: list[dict[str, Any]] = []
    episodes_index = data_dir / "episodes.json"
    recent_urls = get_recent_broadcasted_urls(episodes_index, limit=14)
    if recent_urls:
        filtered_items = []
        for item in raw_items:
            if item.normalized_link in recent_urls:
                dedup_details.append({
                    "title": item.title,
                    "link": item.link,
                    "source": item.source_name,
                    "reason": "cross_episode_url",
                    "detail": "URL matches a recently broadcasted article."
                })
            else:
                filtered_items.append(item)
        
        filtered_count = len(raw_items) - len(filtered_items)
        raw_items = filtered_items
        if filtered_count > 0:
            log.info(
                "Cross-episode dedup: filtered out %d already broadcasted articles, %d items remaining",
                filtered_count,
                len(raw_items),
            )

    # ── 跨期语义相似度去重 ──────────────────────────────────────────────────────
    recent_records = get_recent_broadcasted_texts(episodes_index, limit=14)
    if recent_records and raw_items:
        dedup_cfg = processing_cfg.get("dedup", {})
        
        use_sentence_transformers = False
        try:
            from sentence_transformers import SentenceTransformer, util
            # 使用轻量级多语言模型 paraphrase-multilingual-MiniLM-L12-v2 进行嵌入向量计算
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            use_sentence_transformers = True
            log.info("Successfully loaded SentenceTransformer model 'paraphrase-multilingual-MiniLM-L12-v2' for semantic similarity.")
        except Exception as e:
            log.warning("Sentence-Transformers loading failed (falling back to TF-IDF): %s", e)

        if use_sentence_transformers:
            sim_threshold = float(dedup_cfg.get("embedding_sim_threshold", 0.72))
            try:
                # 提取精简后的对比文本
                segmented_hist_texts = [r.text for r in recent_records]
                new_texts = [
                    extract_semantic_keywords(it.title, it.summary, it.full_text_snippet, top_k=15)
                    for it in raw_items
                ]
                
                # 计算 Embeddings 向量
                hist_embeddings = model.encode(segmented_hist_texts, convert_to_tensor=True, show_progress_bar=False)
                new_embeddings = model.encode(new_texts, convert_to_tensor=True, show_progress_bar=False)
                
                # 计算 Cosine Similarity 相似度矩阵
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
                
                raw_items = filtered_items
                if skipped_count > 0:
                    log.info(
                        "Sentence-Transformers semantic dedup: filtered out %d items by threshold %.2f, %d items remaining",
                        skipped_count,
                        sim_threshold,
                        len(raw_items),
                    )
            except Exception as e:
                log.error("Failed to perform Sentence-Transformers deduplication: %s (falling back to TF-IDF)", e)
                use_sentence_transformers = False

        if not use_sentence_transformers:
            # ── TF-IDF 兜底去重 ─────────────────────────────────────────────────
            import jieba
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            sim_threshold = float(dedup_cfg.get("semantic_sim_threshold", 0.20))
            try:
                # 使用 jieba 进行中文分词，以便正确进行词级别相似度计算
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

                raw_items = filtered_items
                if skipped_count > 0:
                    log.info(
                        "TF-IDF semantic similarity dedup: filtered out %d items by similarity threshold %.2f, %d items remaining",
                        skipped_count,
                        sim_threshold,
                        len(raw_items),
                    )
            except Exception as e:
                log.error("Failed to perform TF-IDF semantic similarity deduplication: %s", e)

    # ── Stage 2: 处理（三层去重 → DBSCAN 聚类 → 五维打分） ──────────────────
    log.info("Stage 2 [pipeline]: dedup → cluster → score …")
    brief = process(raw_items, processing_cfg=processing_cfg)

    # ── 注入去重详细计算信息 ────────────────────────────────────────────────────
    brief.setdefault("metadata", {})["dedup_details"] = dedup_details

    # ── 持久化 brief ─────────────────────────────────────────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    save_brief(brief, brief_path)

    log.info(
        "Pipeline complete: %d stories (main=%d, supporting=%d, quick=%d) → %s",
        len(brief.get("stories", [])),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "main"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "supporting"),
        sum(1 for s in brief.get("stories", []) if s.get("role") == "quick"),
        brief_path,
    )

    return brief
