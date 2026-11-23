#!/usr/bin/env python3
"""列出词典未覆盖的产品名 / 机构名，供人工增补 config/nlp/*.yaml。

用法：
    python scripts/kg_lexicon_gaps.py --days 3
    python scripts/kg_lexicon_gaps.py --days 8 --out-md data/facts/lexicon_gaps.md

背景：A 档要求两端都在词典里（`subject_open` / `object_weak` 会把开放集合降到 B）。
所以「提召回」的真实瓶颈是词典覆盖，不是抽取代码。这个脚本把缺口按频次排出来，
人只需从高到低挑真正是专名的词条加进词典。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.kg.lexicon import KGlexicon  # noqa: E402

ACTION_WORDS = (
    "发布",
    "推出",
    "开源",
    "上线",
    "收购",
    "合作",
    "支持",
    "部署",
    "宣布",
    "量产",
    "投资",
    "融资",
    "首发",
    "亮相",
)
_ACTION_RE = re.compile(
    "(?:" + "|".join(ACTION_WORDS) + r")([^\u4e00-\u9fff，。；：、！？“”「」（）()]{2,40})"
)
# 注意：小数点不能当分隔符，否则「Seedance 2.5 flash」会被截成「Seedance 2」。
_SPLIT_RE = re.compile(r"[,;:!?，。；：！？()\[\]/|｜]|\.(?!\d)")

# 明显不是专名的片段：纯缩写、通用词、单位。
_NOISE = {
    "ai",
    "api",
    "ceo",
    "cto",
    "cfo",
    "lp",
    "gpu",
    "cpu",
    "tpu",
    "the",
    "and",
    "for",
    "with",
    "new",
    "pro",
    "max",
    "app",
    "ios",
    "macos",
    "windows",
    "linux",
}


@dataclass(frozen=True)
class Gap:
    name: str
    count: int
    days: int


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "will",
    "can",
    "has",
    "have",
}


def _clean_fragment(fragment: str) -> str:
    """去掉尾部粘连的普通英文词（正文紧跟在专名后面时会被一起吞进来）。"""
    tokens = fragment.split()
    while tokens and (
        (len(tokens[-1]) == 1 and tokens[-1].isalpha())
        or tokens[-1].lower() in _STOPWORDS
        or len(tokens) > 4
    ):
        if len(tokens) > 4:
            tokens = tokens[:4]
            break
        tokens.pop()
    return " ".join(tokens).strip()


def extract_candidates(text: str) -> list[str]:
    """从「动词 + 拉丁片段」里切出可能是专名的部分。"""
    out: list[str] = []
    for match in _ACTION_RE.finditer(text or ""):
        fragment = _SPLIT_RE.split(match.group(1).strip())[0].strip()
        fragment = _clean_fragment(fragment.strip(" ：:·—-\"'"))
        if len(fragment) < 2 or not re.match(r"^[A-Za-z]", fragment):
            continue
        out.append(fragment)
    return out


def looks_like_noise(name: str) -> bool:
    lowered = name.lower()
    if lowered in _NOISE:
        return True
    # 纯小写英文且不含数字，多半是普通名词（如 skills、agents）。
    return bool(re.fullmatch(r"[a-z ]+", lowered))


def collect_gaps(
    data_dir: Path,
    dates: list[str],
    lexicon: KGlexicon,
) -> list[Gap]:
    counter: Counter[str] = Counter()
    per_name_days: dict[str, set[str]] = {}
    for date_str in dates:
        path = data_dir / "ocean" / date_str / "items.json"
        if not path.is_file():
            continue
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in items:
            text = f"{item.get('title') or ''} {item.get('full_text') or ''}"
            for name in extract_candidates(text):
                if looks_like_noise(name):
                    continue
                if lexicon.find_spans(name):
                    continue
                counter[name] += 1
                per_name_days.setdefault(name, set()).add(date_str)
    return [
        Gap(name=name, count=count, days=len(per_name_days.get(name, ())))
        for name, count in counter.most_common()
    ]


def _recent_dates(data_dir: Path, days: int) -> list[str]:
    ocean = data_dir / "ocean"
    if not ocean.is_dir():
        return []
    names = sorted(path.name for path in ocean.iterdir() if path.is_dir())
    return names[-days:] if days > 0 else names


def render(gaps: list[Gap], dates: list[str], limit: int) -> str:
    today = date.today().isoformat()
    lines = [
        "# 词典缺口（候选专名）",
        "",
        f"生成日期：{today}　扫描：{'、'.join(dates) or '无'}",
        "",
        "> 这些词出现在「发布/推出/开源…」后面，但不在 `config/nlp/{objects,entities}.yaml` 里。",
        "> 词典命中的两端才能进 A 档，所以补词条是提升可用事实量的主要手段。",
        "> 加之前先确认它确实是**专名**，不是普通名词或缩写。",
        "",
        "| 次数 | 出现天数 | 片段 |",
        "|---|---|---|",
    ]
    lines.extend(f"| {gap.count} | {gap.days} | `{gap.name}` |" for gap in gaps[:limit])
    if not gaps:
        lines.append("| - | - | （无缺口） |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="词典缺口报告")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    dates = _recent_dates(data_dir, args.days)
    if not dates:
        print("没有 ocean 数据", file=sys.stderr)
        return 2
    lexicon = KGlexicon.load(Path(args.config_dir) / "nlp")
    gaps = collect_gaps(data_dir, dates, lexicon)
    report = render(gaps, dates, args.limit)
    print(report)
    if args.out_md:
        target = Path(args.out_md)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"\nwritten -> {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
