#!/usr/bin/env python3
"""标注金标：用 LLM 对事件做 subjects/tech_stack/topics 初标，产出 tagging_eval 格式。

为什么需要：S3 的所有特征都来自 S2 标注，但标注改词表是变好还是变坏无法判断
（只有空白率，没有 P/R）。这里产出可评测的金标，供 `eval_tags.py` 计算 P/R。

口径：
- 只标 subjects / tech_stack / topics 三个字段（它们直接喂 S3 特征）；
- LLM 只允许从事件正文/标题中出现过的**原文片段**里抽取（避免编造）；
- 输出 `{event_id, fields:{...}, annotator:"llm_prelabel", reviewed:false}`，
  人工抽检后把 reviewed 改 true。

用法：
    uv run python scripts/label_gold_llm.py --days 6 --per-day 35 --out data/eval/label_gold_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OCEAN = ROOT / "data" / "ocean"

PROMPT = """你是 AI 新闻的标注员。请从下面这条新闻里抽出三类标签。

只输出 JSON 对象，格式：
{{"subjects": ["..."], "tech_stack": ["..."], "topics": ["..."]}}

规则：
1. subjects = 新闻的主体（公司/机构/人物/模型产品名），必须**逐字出现在标题或正文里**；
2. tech_stack = 涉及的技术（如 LLM、多模态、强化学习、世界模型、RAG、Agent），
   用通用术语，不要写公司名；
3. topics = 新闻类别（如 模型发布、融资、监管政策、安全问题、人事变动、研究成果）；
4. 每条最多 5 个，重复即去掉；抽不出就给空数组，不要编造；
5. 中文新闻用中文标签，英文新闻用英文标签，不要翻译专名。

新闻标题：{title}
新闻摘要：{summary}
新闻正文（节选）：{body}
"""


def _llm_cfg(model: str) -> dict:
    return {
        "provider": "openai_compatible",
        "api_key_env": "MINIMAX_API_KEY",
        "model": model,
        "base_url": "https://api.minimaxi.com/v1",
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "timeout": 180,
    }


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key in ("subjects", "tech_stack", "topics"):
        value = data.get(key)
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            out[key] = list(dict.fromkeys(items))[:5]
    return out


def pick_events(days: list[str], per_day: int) -> list[dict]:
    """按事件分层抽样：每天取前 N 个事件，**跨天按 event_id 去重**。

    同一条新闻会连续多天出现在不同日期的 events.json 里，不去重会重复标注
    （实测 210 行里只有 145 个唯一 event_id）。
    """
    picked: list[dict] = []
    seen: set[str] = set()
    for day in days:
        path = OCEAN / day / "events.json"
        if not path.is_file():
            continue
        events = json.loads(path.read_text(encoding="utf-8"))
        events = sorted(events, key=lambda e: str(e.get("event_id")))
        taken = 0
        for event in events:
            if taken >= per_day:
                break
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in seen:
                continue
            if not str(event.get("representative_title") or "").strip():
                continue
            seen.add(event_id)
            picked.append({"date": day, **event})
            taken += 1
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注金标 LLM 初标")
    parser.add_argument("--days", type=int, default=6, help="取最近 N 天")
    parser.add_argument("--per-day", type=int, default=35)
    parser.add_argument("--out", default="data/eval/label_gold_llm.jsonl")
    parser.add_argument("--model", default="MiniMax-M3.1-Flash")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    from ai_news_podcast.business.pipeline.llm_client_service import call_llm

    all_days = sorted(p.name for p in OCEAN.iterdir() if p.is_dir())
    days = all_days[-args.days :]
    events = pick_events(days, args.per_day)
    print(f"抽样 {len(events)} 个事件（{len(days)} 天 × {args.per_day}）")

    cfg = _llm_cfg(args.model)
    items = {i["item_id"]: i for d in days for i in _load_items(d)}

    def one(event: dict) -> dict | None:
        body = ""
        for item_id in (event.get("item_ids") or [])[:2]:
            item = items.get(str(item_id)) or {}
            text = str(item.get("full_text") or item.get("summary") or "")
            if text:
                body = text[:1200]
                break
        prompt = PROMPT.format(
            title=str(event.get("representative_title") or "")[:200],
            summary=str(event.get("summary") or "")[:400],
            body=body,
        )
        try:
            fields = _parse(call_llm(prompt, cfg))
        except Exception:
            return None
        if not fields:
            return None
        return {
            "event_id": str(event.get("event_id")),
            "date": event.get("date"),
            "title": str(event.get("representative_title") or ""),
            "fields": fields,
            "annotator": "llm_prelabel",
            "reviewed": False,
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, e) for e in events]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row:
                results.append(row)
            if index % 40 == 0:
                print(f"  {index}/{len(events)}", flush=True)

    results.sort(key=lambda r: (str(r.get("date")), r["event_id"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8"
    )
    empty = sum(1 for r in results if not any(r["fields"].values()))
    print(f"\n完成：{len(results)} 条（空标注 {empty}）-> {out}")
    return 0


def _load_items(day: str) -> list[dict]:
    path = OCEAN / day / "items.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


if __name__ == "__main__":
    raise SystemExit(main())
