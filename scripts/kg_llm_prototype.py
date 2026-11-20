"""LLM 知识图谱抽取原型（只测质量，不接流水线）。

用法：
    uv run python scripts/kg_llm_prototype.py --feed https://www.qbitai.com/feed --limit 5 \
        --out data/kg_llm

校验（不合格直接丢）：
1. 节点名必须出现在原文；
2. evidence 必须出现在原文；
3. 头、尾必须出现在 evidence 里；
4. 关系长度 2–10 字。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from readability import Document

_WS = re.compile(r"\s+")

_PROMPT = """你是知识图谱抽取器。从这篇新闻里抽取关键事实三元组。

规则：
1. 节点必须是具体对象：公司、机构、人物、产品/模型、政府、标准组织。
   不要抽：泛词（AI、大模型、市场、行业）、媒体名、作者、时间、纯数字。
2. 关系用原文动词或动词短语（2–8 字），例如：发布、开源、上线、融资、收购、
   合作、起诉、监管、裁员、辞职、出任、超过、搭载、接入、举办。
3. 每条边必须给 evidence：包含头和尾的那句原文（原样摘录，不改写）。
4. 只抽文中明确写了的事实，不许推断、不许补充背景知识。
5. 最多 20 条，优先最关键的事实。
6. 不要输出思考过程，不要使用 <think> 标签，直接输出 JSON。

只输出 JSON：
{{"nodes": [{{"name": "原文写法", "type": "company|person|org|product|tech|other"}}],
 "edges": [{{"head": "节点名", "relation": "动词", "tail": "节点名", "evidence": "原文句子"}}]}}

标题：{title}

正文：
{body}
"""


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
                "full_text": full_text[:4000],
            }
        )
    return articles


def _llm(prompt: str, cfg: dict[str, Any]) -> tuple[str, int]:
    client = OpenAI(
        api_key=os.environ[cfg.get("api_key_env", "MINIMAX_API_KEY")],
        base_url=cfg.get("base_url"),
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=cfg.get("model", "MiniMax-M2.7-highspeed"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=16384,
    )
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return response.choices[0].message.content or "", tokens


def _parse_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", raw, flags=re.S)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "edges" in value:
            return value
    return {}


def _ground(article: dict[str, str], data: dict[str, Any]) -> tuple[list[dict], list[dict], dict]:
    haystack = _clean(f"{article['title']} {article['summary']} {article['full_text']}")
    nodes = [
        {"name": _clean(node.get("name") or ""), "type": _clean(node.get("type") or "other")}
        for node in data.get("nodes") or []
        if isinstance(node, dict)
    ]
    nodes = [node for node in nodes if node["name"] and node["name"] in haystack]
    kept: list[dict[str, str]] = []
    dropped = {"node_missing": 0, "evidence_missing": 0, "endpoint_missing": 0, "bad_relation": 0}
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        head = _clean(edge.get("head") or "")
        tail = _clean(edge.get("tail") or "")
        relation = _clean(edge.get("relation") or "")
        evidence = _clean(edge.get("evidence") or "")
        if not head or not tail or head not in haystack or tail not in haystack:
            dropped["endpoint_missing"] += 1
            continue
        if not evidence or evidence not in haystack:
            dropped["evidence_missing"] += 1
            continue
        if head not in evidence or tail not in evidence:
            dropped["endpoint_missing"] += 1
            continue
        if not 2 <= len(relation) <= 10:
            dropped["bad_relation"] += 1
            continue
        kept.append({"head": head, "relation": relation, "tail": tail, "evidence": evidence})
    return nodes, kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default="data/kg_llm")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="MiniMax-M2.7-highspeed")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(".env")
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = dict(raw_cfg.get("llm") or {})
    cfg["model"] = args.model

    articles = _fetch_articles(args.feed, args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    def _run(article: dict[str, str]) -> dict[str, Any]:
        body = article["full_text"] or article["summary"]
        prompt = _PROMPT.format(title=article["title"], body=body[:4000])
        raw, tokens = _llm(prompt, cfg)
        data = _parse_json(raw)
        nodes, edges, dropped = _ground(article, data)
        return {
            "title": article["title"],
            "link": article["link"],
            "nodes": nodes,
            "edges": edges,
            "dropped": dropped,
            "tokens": tokens,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(_run, articles))

    for row in rows:
        print(f"\n【{row['title'][:34]}】节点 {len(row['nodes'])} 边 {len(row['edges'])}")
        for edge in row["edges"]:
            print(f"  ({edge['head']}) --{edge['relation']}--> ({edge['tail']})")
    (out_dir / "kg_llm.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "articles": len(rows),
        "nodes": sum(len(row["nodes"]) for row in rows),
        "edges": sum(len(row["edges"]) for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
    }
    print("\n", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
