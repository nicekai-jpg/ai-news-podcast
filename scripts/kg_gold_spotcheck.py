#!/usr/bin/env python3
"""把 LLM 初标的推荐金标整理成人工抽检清单。

输出一份 markdown：分层抽样（判 true / false 各若干），每行带勾选框。
人工只需核对，发现与 LLM 不一致的就标 `x`，然后用 --apply 回写。

用法：
    python scripts/kg_gold_spotcheck.py                       # 生成清单
    python scripts/kg_gold_spotcheck.py --apply --review-md ...  # 回写人工判定
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = "data/eval/recommend/gold_v1_llm.jsonl"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def render(rows: list[dict], *, per_class: int, seed: int) -> str:
    rng = random.Random(seed)
    judged = [r for r in rows if r.get("worth_publishing") is not None]
    true_rows = [r for r in judged if r["worth_publishing"]]
    false_rows = [r for r in judged if not r["worth_publishing"]]
    picked = rng.sample(true_rows, min(per_class, len(true_rows)))
    picked += rng.sample(false_rows, min(per_class, len(false_rows)))
    rng.shuffle(picked)

    lines = [
        f"# 推荐金标抽检（{len(picked)} 条）",
        "",
        "> LLM 初标结果。核对方式：**认同就不动；不认同把「改判」列填相反值**。",
        "> 全部核对完告诉我，我把修改回写并冻结金标。",
        "",
        "| # | LLM 判定 | 改判 | 标题 | LLM 理由 |",
        "|---|---|---|---|---|",
    ]
    for index, row in enumerate(picked, 1):
        verdict = "要" if row["worth_publishing"] else "不要"
        title = str(row.get("title") or "").replace("|", "丨")[:70]
        why = str(row.get("why") or "").replace("|", "丨")[:28]
        lines.append(f"| {index} | {verdict} |  | {title} | {why} |")
    lines += ["", "---", "", "## 补充信息：全部 1000 条的判定分布", ""]
    dist = Counter(
        ("要" if r["worth_publishing"] else "不要")
        if r.get("worth_publishing") is not None
        else "未判定"
        for r in rows
    )
    lines += ["| 判定 | 数量 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in dist.most_common()]
    return "\n".join(lines)


def apply_corrections(gold: Path, review_md: Path, *, seed: int) -> int:
    """按抽检清单里的「改判」列回写 worth_publishing，并标记 reviewed。"""
    lines = review_md.read_text(encoding="utf-8").splitlines()
    # 清单里出现的每一行都算「已核对」：没填改判即认同 LLM 判定。
    corrections: dict[str, bool] = {}
    for line in lines:
        if not line.startswith(("| ", "| #", "|---")) or line.startswith(("| #", "|---")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        title, verdict, correction = cells[3], cells[1], cells[2]
        if verdict not in ("要", "不要") and correction not in ("要", "不要"):
            continue
        if correction in ("要", "不要"):
            corrections[title] = correction == "要"
        else:
            corrections[title] = verdict == "要"
    rows = _load(gold)
    changed = 0
    for row in rows:
        title = str(row.get("title") or "").replace("|", "丨")[:70]
        if title not in corrections:
            continue
        if bool(row["worth_publishing"]) != corrections[title]:
            row["worth_publishing"] = corrections[title]
            row["label_source"] = "human_corrected"
            changed += 1
        row["reviewed"] = True
    gold.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"回写完成：修正 {changed} 条，标记 reviewed {len(corrections)} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="推荐金标抽检")
    parser.add_argument("--gold", default=str(ROOT / DEFAULT_GOLD))
    parser.add_argument("--out-md", default=str(ROOT / "data/eval/recommend/gold_spotcheck.md"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    gold = Path(args.gold)
    if not gold.is_file():
        print(f"金标不存在: {gold}", file=__import__("sys").stderr)
        return 2
    if args.apply:
        return apply_corrections(gold, Path(args.out_md), seed=args.seed)

    rows = _load(gold)
    text = render(rows, per_class=args.per_class, seed=args.seed)
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text[:1500])
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
