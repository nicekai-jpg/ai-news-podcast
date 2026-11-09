"""Stage 1 采集入海编排（runner）。

按通道调度各 fetcher，单源失败隔离（记录 status + 原因，可启用 failover），
最后输出统一的 ``contracts.Item`` 列表；``collect_and_write`` 落盘为
``raw-{date}.json.gz`` artifact（含 manifest + source_stats）。

对外接口：

```python
collect_all(config_path, date_str, *, rsshub_base=None, window_hours=72,
            http_client=None, ...) -> CollectionResult
collect_and_write(config_path, date_str, output_path, *, ...) -> Path
```
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Item
from ai_news_podcast.business.contracts.artifact import write_artifact

from . import rss, rsshub, web, wechat2rss
from .base import (
    ConfigError,
    FetchContext,
    SourceConfig,
    finalize_items,
    load_sources,
)

logger = logging.getLogger(__name__)

_FETCHERS: dict[str, Callable[[SourceConfig, FetchContext], list[Item]]] = {
    "c1_rss": rss.fetch_feed,
    "c2_atom": rss.fetch_feed,
    "c3_rsshub": rsshub.fetch_rsshub,
    "c4_web": web.fetch_web,
    "c5_wechat2rss": wechat2rss.fetch_wechat2rss,
}


@dataclass
class SourceStat:
    """单源采集统计（写入 artifact manifest，便于审计/调参）。"""

    name: str
    label: str
    channel: str
    status: str = "ok"  # ok / empty / failed / skipped / disabled
    count: int = 0
    error: str = ""
    failover: bool = False
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollectionResult:
    """一次采集的完整结果。"""

    date: str
    items: list[Item] = field(default_factory=list)
    source_stats: list[dict[str, Any]] = field(default_factory=list)
    window_hours: int = 72

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "window_hours": self.window_hours,
            "item_count": len(self.items),
            "source_stats": self.source_stats,
        }


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _dispatch(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    fetcher = _FETCHERS.get(source.channel)
    if fetcher is None:
        raise ConfigError(f"源 {source.name} 的 channel 未知: {source.channel!r}")
    return fetcher(source, ctx)


def _try_failover(
    source: SourceConfig,
    ctx: FetchContext,
) -> tuple[list[Item], SourceStat] | None:
    """主通道失败时尝试一次 failover；仍失败返回 ``None``（不再级联）。"""
    spec = source.failover
    if spec is None:
        return None
    if spec.channel == "c3_rsshub" and not ctx.rsshub_base:
        return None
    fallback = replace(
        source,
        channel=spec.channel,
        url=spec.url or source.url,
        urls=[],
        rsshub_route=spec.route or source.rsshub_route,
        parser=spec.parser or source.parser,
        failover=None,
    )
    started = time.monotonic()
    try:
        items = finalize_items(_dispatch(fallback, ctx), fallback, ctx)
    except Exception as exc:  # failover 失败只能放弃，不影响其它源
        logger.warning("源 %s 的 failover(%s) 也失败: %s", source.name, spec.channel, exc)
        return None
    stat = SourceStat(
        name=source.name,
        label=source.label,
        channel=source.channel,
        status="ok" if items else "empty",
        count=len(items),
        failover=True,
        duration_ms=_duration_ms(started),
    )
    logger.info("源 %s 经 failover(%s) 取到 %d 条", source.name, spec.channel, len(items))
    return items, stat


def _process_source(
    source: SourceConfig,
    ctx: FetchContext,
    *,
    rsshub_ready: bool,
) -> tuple[list[Item], SourceStat]:
    """抓取单个源：失败隔离 + 可选 failover，永远返回统计而不抛异常。"""
    if not source.enabled:
        return [], SourceStat(
            name=source.name,
            label=source.label,
            channel=source.channel,
            status="disabled",
        )
    started = time.monotonic()
    if source.channel == "c3_rsshub" and not rsshub_ready:
        fallback = _try_failover(source, ctx)
        if fallback is not None:
            return fallback
        return [], SourceStat(
            name=source.name,
            label=source.label,
            channel=source.channel,
            status="skipped",
            error="RSSHub 容器未就绪，C3 源跳过",
            duration_ms=_duration_ms(started),
        )
    try:
        items = finalize_items(_dispatch(source, ctx), source, ctx)
    except Exception as exc:  # 单源失败必须被隔离，不能中断整轮采集
        logger.warning("源 %s(%s) 采集失败: %s", source.name, source.channel, exc)
        fallback = _try_failover(source, ctx)
        if fallback is not None:
            return fallback
        return [], SourceStat(
            name=source.name,
            label=source.label,
            channel=source.channel,
            status="failed",
            error=str(exc),
            duration_ms=_duration_ms(started),
        )
    stat = SourceStat(
        name=source.name,
        label=source.label,
        channel=source.channel,
        status="ok" if items else "empty",
        count=len(items),
        duration_ms=_duration_ms(started),
    )
    return items, stat


def collect_all(
    config_path: str | Path,
    date_str: str,
    *,
    rsshub_base: str | None = None,
    window_hours: int = 72,
    http_client: Any | None = None,
    max_workers: int = 4,
    fetch_fulltext: bool = True,
    request_interval: float = 1.0,
    retry_wait: float = 1.0,
    now: datetime | None = None,
) -> CollectionResult:
    """按 ``config/sources_v3.yaml`` 采集全部启用源，返回统一 item 列表。

    - ``rsshub_base``：临时 RSSHub 实例地址；未提供或未就绪时 C3 源标记 skipped
      （若有 failover 则改走备用通道），其它通道照常；
    - ``http_client``：可注入的 httpx 客户端（测试/复用连接池）；
    - ``max_workers``：源级并发数（``1`` 为串行）。
    """
    sources = load_sources(config_path)
    context = FetchContext(
        date_str=date_str,
        window_hours=window_hours,
        http_client=http_client,
        rsshub_base=(rsshub_base or "").strip() or None,
        now=now or datetime.now(tz=UTC),
        fetch_fulltext=fetch_fulltext,
        throttle_interval=request_interval,
        retry_wait=retry_wait,
    )
    try:
        rsshub_ready = _ensure_rsshub(context, sources)
        if max_workers > 1 and len(sources) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_source, source, context, rsshub_ready=rsshub_ready)
                    for source in sources
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                _process_source(source, context, rsshub_ready=rsshub_ready) for source in sources
            ]
    finally:
        context.close()

    all_items: list[Item] = []
    stats: list[dict[str, Any]] = []
    for items, stat in results:
        all_items.extend(items)
        stats.append(stat.to_dict())
    logger.info(
        "采集完成 date=%s: %d 源, %d 条 item",
        date_str,
        len(sources),
        len(all_items),
    )
    return CollectionResult(
        date=date_str,
        items=all_items,
        source_stats=stats,
        window_hours=window_hours,
    )


def _ensure_rsshub(context: FetchContext, sources: list[SourceConfig]) -> bool:
    """只在存在启用中的 C3 源时探测 RSSHub 容器。"""
    needs_rsshub = any(source.enabled and source.channel == "c3_rsshub" for source in sources)
    if not needs_rsshub:
        return False
    if not context.rsshub_base:
        logger.warning("未提供 rsshub_base，C3 通道整体跳过")
        return False
    return rsshub.probe_rsshub(context)


def _resolve_artifact_path(output_path: str | Path, date_str: str) -> Path:
    path = Path(output_path)
    if path.name.endswith(".json.gz"):
        return path
    return path / f"raw-{date_str}.json.gz"


def collect_and_write(
    config_path: str | Path,
    date_str: str,
    output_path: str | Path,
    *,
    rsshub_base: str | None = None,
    window_hours: int = 72,
    http_client: Any | None = None,
    **kwargs: Any,
) -> Path:
    """采集并写出 ``raw-{date}.json.gz`` artifact，返回实际路径。

    ``output_path`` 以 ``.json.gz`` 结尾时视为文件路径，否则视为目录
    （自动拼 ``raw-{date}.json.gz``）。
    """
    result = collect_all(
        config_path,
        date_str,
        rsshub_base=rsshub_base,
        window_hours=window_hours,
        http_client=http_client,
        **kwargs,
    )
    path = _resolve_artifact_path(output_path, date_str)
    return write_artifact(
        path,
        result.items,
        date_str=date_str,
        window_hours=result.window_hours,
        source_stats=result.source_stats,
    )
