"""采集源健康度：把多天 raw artifact 的 source_stats 聚合成可行动报告。

输入是 `data/raw/raw-*.json.gz` 里的 (manifest, items)，输出每个源的：
- 出现天数、各状态次数（ok/empty/failed/skipped/disabled）；
- 条目总数、全文抓取成功率、平均耗时；
- 健康标记 ok / warn / bad 与原因。

阈值故意保守：只把“长期失败/长期空/全文率过低”标成 bad，避免单日网络抖动误伤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_news_podcast.business.contracts import Item

FULLTEXT_OK = "ok"
FULLTEXT_PARTIAL = ("truncated",)


@dataclass
class SourceHealth:
    name: str
    label: str = ""
    channel: str = ""
    days: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    items_total: int = 0
    fulltext_ok: int = 0
    fulltext_partial: int = 0
    fulltext_total: int = 0
    duration_ms_total: int = 0
    failures: int = 0
    empty_days: int = 0
    skipped_days: int = 0
    disabled_days: int = 0
    health: str = "ok"
    reasons: list[str] = field(default_factory=list)

    @property
    def fulltext_ok_rate(self) -> float | None:
        return round(self.fulltext_ok / self.fulltext_total, 4) if self.fulltext_total else None

    @property
    def avg_duration_ms(self) -> int:
        return int(self.duration_ms_total / self.days) if self.days else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "channel": self.channel,
            "days": self.days,
            "status_counts": self.status_counts,
            "items_total": self.items_total,
            "fulltext_ok": self.fulltext_ok,
            "fulltext_partial": self.fulltext_partial,
            "fulltext_total": self.fulltext_total,
            "fulltext_ok_rate": self.fulltext_ok_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "failures": self.failures,
            "empty_days": self.empty_days,
            "skipped_days": self.skipped_days,
            "disabled_days": self.disabled_days,
            "health": self.health,
            "reasons": self.reasons,
        }


def aggregate_source_health(
    snapshots: list[tuple[dict[str, Any], list[Item]]],
) -> list[SourceHealth]:
    """按源聚合多天采集结果，返回健康度列表（bad/warn 优先）。"""
    by_source: dict[str, SourceHealth] = {}
    for manifest, items in snapshots:
        stats = manifest.get("source_stats") or []
        items_by_source: dict[str, list[Item]] = {}
        for item in items:
            items_by_source.setdefault(item.source, []).append(item)
        for stat in stats:
            name = str(stat.get("name") or "")
            if not name:
                continue
            health = by_source.setdefault(
                name,
                SourceHealth(
                    name=name,
                    label=str(stat.get("label") or ""),
                    channel=str(stat.get("channel") or ""),
                ),
            )
            health.days += 1
            status = str(stat.get("status") or "unknown")
            health.status_counts[status] = health.status_counts.get(status, 0) + 1
            health.items_total += int(stat.get("count") or 0)
            health.duration_ms_total += int(stat.get("duration_ms") or 0)
            if status == "failed":
                health.failures += 1
            elif status == "empty":
                health.empty_days += 1
            elif status == "skipped":
                health.skipped_days += 1
            elif status == "disabled":
                health.disabled_days += 1
            for item in items_by_source.get(name, []):
                health.fulltext_total += 1
                if item.full_text_status == FULLTEXT_OK:
                    health.fulltext_ok += 1
                elif item.full_text_status in FULLTEXT_PARTIAL:
                    health.fulltext_partial += 1
    for health in by_source.values():
        _mark_health(health)
    return sorted(by_source.values(), key=lambda h: (-_health_rank(h.health), h.name))


def _health_rank(health: str) -> int:
    return {"bad": 2, "warn": 1, "ok": 0}.get(health, 0)


def _mark_health(health: SourceHealth) -> None:
    reasons: list[str] = []
    if health.days == 0:
        health.health = "warn"
        reasons.append("没有采集记录")
    if health.days and health.failures >= max(1, health.days // 2 + health.days % 2):
        reasons.append(f"失败 {health.failures}/{health.days} 天")
    if health.days >= 2 and health.items_total == 0 and health.disabled_days < health.days:
        reasons.append("连续多天 0 条")
    if health.fulltext_ok_rate is not None and health.fulltext_ok_rate < 0.5:
        reasons.append(f"全文成功率 {health.fulltext_ok_rate:.0%}")
    if health.empty_days and health.days and health.empty_days >= max(1, health.days // 2):
        reasons.append(f"空源 {health.empty_days}/{health.days} 天")
    if any("失败" in reason for reason in reasons) or any("0 条" in reason for reason in reasons):
        health.health = "bad"
    elif reasons:
        health.health = "warn"
    else:
        health.health = "ok"
    health.reasons = reasons


def render_markdown(healths: list[SourceHealth], *, limit: int = 40) -> str:
    lines = [
        "# 采集源健康度",
        "",
        "| 源 | 名称 | 天数 | 条目 | 失败天数 | 空源天数 | 全文成功率 | 平均耗时 | 健康 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for health in healths[:limit]:
        rate = "n/a" if health.fulltext_ok_rate is None else f"{health.fulltext_ok_rate:.0%}"
        lines.append(
            f"| {health.name} | {health.label} | {health.days} | {health.items_total} | "
            f"{health.failures} | {health.empty_days} | {rate} | {health.avg_duration_ms}ms | "
            f"{health.health} |"
        )
    attention = [h for h in healths if h.health != "ok"]
    if attention:
        lines += ["", "## 需要处理", ""]
        lines += [f"- `{h.name}`：{'；'.join(h.reasons) or h.health}" for h in attention]
    return "\n".join(lines) + "\n"
