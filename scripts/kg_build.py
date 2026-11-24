"""知识图谱抽取 CLI 原型：抓文章 → build_kg → 落盘 kg.json。

用法：

    uv run python scripts/kg_build.py --feed https://www.qbitai.com/feed --limit 10 \
        --out data/kg

装 spaCy + 中文模型时自动做依存确认（置信度 0.9/0.65/0.4），否则降级模板（0.55）。
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from readability import Document

from ai_news_podcast.business.kg import build_kg
from ai_news_podcast.business.kg.lexicon import KGlexicon

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def _fetch_articles(feed_url: str, limit: int) -> list[dict[str, str]]:
    import feedparser

    resp = httpx.get(
        feed_url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    parsed = feedparser.parse(resp.text)
    articles: list[dict[str, str]] = []
    for entry in parsed.entries[:limit]:
        link = str(entry.get("link") or "")
        full_text = ""
        if link:
            try:
                page = httpx.get(
                    link, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
                )
                if page.status_code == 200:
                    raw = Document(page.text).summary()
                    decoded = re.sub(r"<[^>]+>", " ", html.unescape(raw))
                    full_text = _clean(html.unescape(decoded))
            except Exception:
                full_text = ""
        articles.append(
            {
                "title": _clean(entry.get("title") or ""),
                "link": link,
                "summary": _clean(entry.get("summary") or ""),
                "full_text": full_text[:6000],
            }
        )
    return articles


def _load_nlp() -> Any | None:
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("zh_core_web_sm")
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="data/kg")
    parser.add_argument("--nlp-dir", default="config/nlp")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument(
        "--strict", action="store_true", help="严格模式：主语只认词表实体、尾巴必须是短名词短语"
    )
    args = parser.parse_args()

    lexicon = KGlexicon.load(args.nlp_dir)
    nlp = _load_nlp()
    print(f"依存确认：{'开（spaCy 已加载）' if nlp is not None else '关（未装 spaCy/模型）'}")

    articles = _fetch_articles(args.feed, args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    for article in articles:
        doc = build_kg(
            article["full_text"] or article["summary"],
            article["title"],
            lexicon=lexicon,
            nlp=nlp,
            min_confidence=args.min_confidence,
            strict=args.strict,
        )
        documents.append({**doc.to_dict(), "link": article["link"]})
        for edge in doc.edges:
            relation_counts[edge.relation] += 1
        print(
            f"  {len(doc.nodes):3} 节点 {len(doc.edges):3} 边 "
            f"高置信 {doc.stats['high_confidence']:3}  {article['title'][:32]}"
        )

    (out_dir / "kg_batch.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "articles": len(documents),
        "nodes": sum(len(row["nodes"]) for row in documents),
        "edges": sum(len(row["edges"]) for row in documents),
        "high_confidence": sum(row["stats"]["high_confidence"] for row in documents),
        "relations": relation_counts.most_common(),
    }
    (out_dir / "kg_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
