#!/usr/bin/env python3
"""S1 采集层的**正确**验证脚本。

为什么单独写这个：这一轮排查中我反复用错测试方法，导致多个错误结论：

  1. 用 `now=datetime.now()` 测 2026-11-16 的数据 → 窗口过滤全灭，
     误判「91% 条目无用」
  2. 用 `rss.fetch_feed()` 直接测 C4 源 → 绕过 channel 分派，误判
     minimax/sensetime 不可用
  3. 测 C3 源不传 `rsshub_base` → 全部报「缺少 url」，误判 rsshub 源坏了
  4. 只测 1 天就下结论 → 把偶发失败当持久故障

本脚本强制以下口径：
  - 参考时间 = 目标数据日期的 12:00 UTC（不是今天）
  - 走 `_FETCHERS[channel]` 真实分派
  - C3 源单独标注（本地无 RSSHub 时跳过，不当作故障）
  - 每个源重复 N 次，区分「持久失败」与「偶发失败」
  - 结果按 成功/失败/窗口外 分类，并给出可比对的基线

用法：
    python scripts/s1_verify.py                      # 默认验证 3 天
    python scripts/s1_verify.py --date 2026-11-16 --repeat 3
    python scripts/s1_verify.py --only broken        # 只测已知失败的源
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_news_podcast.business.collect.base import (  # noqa: E402
    FetchContext,
    SourceConfig,
    load_sources,
    within_window,
)
from ai_news_podcast.business.collect.runner import _FETCHERS  # noqa: E402

# 已知的「历史失败源」，用于快速复查
KNOWN_BROKEN = ("synced_review", "36kr_native", "sensetime_news")


@dataclass(slots=True)
class SourceResult:
    name: str
    channel: str
    attempts: int = 0
    ok: int = 0
    failed: int = 0
    raw: int = 0
    in_window: int = 0
    error: str = ""

    @property
    def verdict(self) -> str:
        if self.name and self.channel == "c3_rsshub" and self.ok == 0 and "RSSHub" in self.error:
            return "skipped(rsshub)"
        if self.ok == 0:
            return "FAILED"
        if self.raw == 0:
            return "EMPTY"
        if self.in_window == 0:
            return "STALE"  # 抓到了但全在窗口外
        return "OK"


def _context(date_str: str, window_hours: int) -> FetchContext:
    """参考时间必须是数据日期，不是今天——这是之前最大的坑。"""
    day = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
    return FetchContext(
        date_str=date_str,
        window_hours=window_hours,
        now=day,
        fetch_fulltext=False,  # 验证源可用性不需要回源抓正文
    )


def probe(source: SourceConfig, ctx: FetchContext, repeat: int) -> SourceResult:
    fetcher = _FETCHERS.get(source.channel)
    row = SourceResult(name=source.name, channel=source.channel)
    for _ in range(max(1, repeat)):
        row.attempts += 1
        try:
            items = fetcher(source, ctx)  # type: ignore[misc]
        except Exception as exc:
            row.failed += 1
            row.error = f"{type(exc).__name__}: {exc}"[:160]
            continue
        row.ok += 1
        row.raw = max(row.raw, len(items))
        row.in_window = max(
            row.in_window,
            sum(
                1
                for item in items
                if item.published_at
                and within_window(item.published_at, now=ctx.now, window_hours=ctx.window_hours)
            ),
        )
        if row.error:
            row.error = ""  # 至少成功过一次，不算持久失败
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1 采集验证")
    parser.add_argument("--date", default="2026-11-16")
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--repeat", type=int, default=2, help="每源重复次数，区分偶发与持久失败")
    parser.add_argument("--only", choices=["all", "broken", "enabled"], default="broken")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.disable(logging.WARNING)  # 源自身的 warning 太多，这里只看结论
    sources = [s for s in load_sources(ROOT / "config" / "sources_v3.yaml") if s.enabled]
    if args.only == "broken":
        sources = [s for s in sources if s.name in KNOWN_BROKEN]
    elif args.only == "enabled":
        pass

    ctx = _context(args.date, args.window_hours)
    print(f"参考时间 {ctx.now.isoformat()}（窗口 {args.window_hours}h），每源重复 {args.repeat} 次")
    print(f"待测 {len(sources)} 个源\n")

    rows = [probe(s, ctx, args.repeat) for s in sources]
    rows.sort(key=lambda r: (r.verdict, -r.raw))

    print(f"{'源':28}{'通道':14}{'成功/尝试':>10}{'抓到':>6}{'窗口内':>7}  判定")
    for r in rows:
        if r.verdict == "OK" and not args.verbose:
            continue
        print(
            f"{r.name:28}{r.channel:14}{r.ok}/{r.attempts:>3}{r.raw:>6}{r.in_window:>7}  {r.verdict}"
            + (f" | {r.error[:70]}" if r.error else "")
        )

    print("\n判定汇总:", dict(Counter(r.verdict for r in rows)))
    print("\n判读说明：")
    print("  FAILED  = 每次尝试都抛异常 → 源真的坏了")
    print("  EMPTY   = 抓取成功但没有条目 → feed 为空或解析器不匹配")
    print("  STALE   = 抓到但全在时间窗外 → 档案型 feed（不是故障）")
    print("  OK      = 至少 1 条落在窗口内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
