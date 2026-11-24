#!/usr/bin/env python3
"""用 LLM 给推荐金标做初标，供人工抽检校准。

为什么不用「人工标 1000 条」：需要判断的候选有 950+，人工成本不现实。
这里用 LLM 按**从实际日报反推出来的标准**做初标，人工只抽检校准。

判定标准（来自 data/reports/daily_report_*.md 的实际选题）：
0. **arXiv 论文一律不要**（2026-11-24 决定：该源已停用，选题不看论文）
1. AI 相关（核心或密切相关）
2. 有具体的新事件（发布/融资/收购/政策/研究结果/事故）
3. 信息量足以展开成「背景 + 突破 + 影响」
4. 排除：招聘课程广告榜单、纯评测软文、观点散文、旧闻回顾、教程

用法：
    python scripts/kg_recommend_gold.py --limit 10          # 先验证 prompt 质量
    python scripts/kg_recommend_gold.py --workers 6         # 全量
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_QUEUE = "data/eval/recommend/review_queue_2026-11_1000_prelabel.jsonl"
DEFAULT_OUT = "data/eval/recommend/gold_v1_llm.jsonl"

PROMPT = """你是 AI 播客的选题编辑，判断下面这条新闻**值不值得做成当天的一期节目**。

值得做（true）：
- 与 AI 直接相关；
- 有具体的新事件：发布/开源/融资/收购/合作/政策监管/诉讼/事故/研究成果/人事变动；
- 有足够信息量，能展开成「背景 + 核心突破 + 行业影响」三段。

不值得做（false）：
- **arXiv / 学术论文预印本**（无论多重要，一律不做：本项目选题只看行业事件，不看论文）；
- 招聘、课程、训练营、广告、榜单、促销；
- 纯评测/上手/软文，没有独立事件；
- 观点散文、经验分享、教程、方法论；
- 旧闻回顾、周报汇总、纯转述且无新进展；
- 与 AI 无关的汽车/消费/社会新闻；
- 只有标题没有实质内容。

只输出 JSON 数组，每条一个对象：
[{{"i": 序号, "worth": true/false, "why": "不超过 20 字的理由"}}]

待判断：
{items}
"""


def _llm_cfg(model: str) -> dict:
    return {
        "provider": "openai_compatible",
        "api_key_env": "MINIMAX_API_KEY",
        "model": model,
        "base_url": "https://api.minimaxi.com/v1",
        "temperature": 0.0,
        "max_output_tokens": 8192,
        "timeout": 240,
    }


def _parse(raw: str | None) -> list[dict]:
    if not raw:
        return []
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [row for row in data if isinstance(row, dict)]


def _batch_prompt(rows: list[dict]) -> str:
    lines = []
    for index, row in enumerate(rows):
        title = str(row.get("title") or "")[:120]
        summary = str(row.get("summary") or "")[:220]
        lines.append(
            f'{{"i": {index}, "title": {json.dumps(title, ensure_ascii=False)}, '
            f'"summary": {json.dumps(summary, ensure_ascii=False)}}}'
        )
    return PROMPT.format(items="\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="推荐金标 LLM 初标")
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default="MiniMax-M3.1-Flash")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)

    from ai_news_podcast.business.pipeline.llm_client_service import call_llm

    rows = [
        json.loads(line)
        for line in Path(args.queue).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    batches = [rows[i : i + args.batch] for i in range(0, len(rows), args.batch)]
    print(f"待标注 {len(rows)} 条，分 {len(batches)} 批，并发 {args.workers}")

    cfg = _llm_cfg(args.model)
    results: dict[int, dict] = {}
    started = time.monotonic()

    def one(start: int) -> tuple[int, list[dict], str | None]:
        chunk = batches[start]
        raw = call_llm(_batch_prompt(chunk), cfg)
        return start, _parse(raw), raw

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one, i): i for i in range(len(batches))}
        for future in as_completed(futures):
            start, parsed, _raw = future.result()
            chunk = batches[start]
            by_index = {int(row.get("i", -1)): row for row in parsed}
            for index, row in enumerate(chunk):
                verdict = by_index.get(index)
                global_index = args.offset + start * args.batch + index
                results[global_index] = {
                    "date": row.get("date"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "worth_publishing": (verdict or {}).get("worth"),
                    "why": (verdict or {}).get("why", ""),
                    "prelabel_worth": row.get("suggested_worth_publishing"),
                    "prelabel_relevance": row.get("suggested_ai_relevance"),
                    "label_source": "llm_prelabel",
                    "reviewed": False,
                }
            done += len(chunk)
            if done % 100 < args.batch:
                print(f"  {done}/{len(rows)} ({time.monotonic() - started:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = [results[i] for i in sorted(results)]
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8"
    )

    judged = [r for r in ordered if r["worth_publishing"] is not None]
    true_n = sum(1 for r in judged if r["worth_publishing"])
    disagree = [
        r
        for r in judged
        if r["prelabel_worth"] is not None
        and bool(r["prelabel_worth"]) != bool(r["worth_publishing"])
    ]
    print(f"\n标注完成：{len(ordered)} 条，有效判定 {len(judged)}")
    print(f"  worth=true: {true_n} ({true_n / max(1, len(judged)):.0%})")
    print(f"  与规则预标分歧: {len(disagree)} ({len(disagree) / max(1, len(judged)):.0%})")
    print(f"  输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
