#!/usr/bin/env python3
"""RSSHub 快照采集脚本（供 GitHub Actions 的 rsshub_cache workflow 使用）。

思路：CI 里先起一个**临时的** RSSHub 实例（service container），本脚本请求配置里的
RSSHub 路由 / 直连 RSS 地址，把结果归一化成小 JSON 落到 `data/cache/`，提交回仓库。
日报 Stage 1 之后优先消费这些快照，从而：

- 不依赖任何第三方公共 RSSHub 实例（公共实例随时 403 / 停服）；
- 抓取失败时保留上一次快照（不会用空数据覆盖）；
- 快照只存 title/link/published_at/summary，体积小、可版本化、可审计。

只依赖 Python 标准库（不装包），单个源失败不影响其他源；退出码恒为 0（缓存链路必须可失败）。
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger("build_rsshub_cache")
USER_AGENT = "ai-news-podcast-cache/0.1 (+https://github.com/)"
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
}
_DEFAULT_TIMEOUT = 60


def _text(node: ET.Element | None) -> str:
    """取 Element 的完整文本（含子节点文本）。"""
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _strip_html(text: str) -> str:
    """去标签 + 反转义 + 压缩空白，截断到 1200 字。"""
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()[:1200]


def _parse_date(value: str) -> str | None:
    """把 RFC822 / ISO 日期解析成 UTC ISO 字符串；无法解析返回 None。"""
    value = (value or "").strip()
    if not value:
        return None
    parsed: datetime | None = None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _atom_link(node: ET.Element) -> str:
    """Atom entry 的 alternate 链接（没有 rel 时取第一个 href）。"""
    fallback = ""
    for link in node.findall("atom:link", _ATOM_NS):
        href = str(link.get("href") or "").strip()
        if not href:
            continue
        rel = link.get("rel")
        if rel in (None, "alternate"):
            return href
        fallback = fallback or href
    return fallback


def parse_feed(xml_text: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """解析 RSS 2.0 / Atom，返回归一化条目（title/link/published_at/summary）。"""
    root = ET.fromstring(xml_text)
    items: list[dict[str, Any]] = []

    if root.find("channel") is not None:
        channel = root.find("channel")
        nodes = channel.findall("item") if channel is not None else []
        for node in nodes[:limit]:
            summary = _text(node.find("content:encoded", _ATOM_NS)) or _text(
                node.find("description")
            )
            items.append(
                {
                    "title": _text(node.find("title")),
                    "link": _text(node.find("link")),
                    "published_at": _parse_date(_text(node.find("pubDate"))),
                    "summary": _strip_html(summary),
                }
            )
    else:
        for node in root.findall("atom:entry", _ATOM_NS)[:limit]:
            content = node.find("atom:content", _ATOM_NS)
            summary_node = node.find("atom:summary", _ATOM_NS)
            body = _text(content) if content is not None else _text(summary_node)
            items.append(
                {
                    "title": _text(node.find("atom:title", _ATOM_NS)),
                    "link": _atom_link(node),
                    "published_at": _parse_date(
                        _text(node.find("atom:published", _ATOM_NS))
                        or _text(node.find("atom:updated", _ATOM_NS))
                    ),
                    "summary": _strip_html(body),
                }
            )

    return [item for item in items if item["title"] and item["link"]]


def normalize_items(
    items: list[dict[str, Any]],
    *,
    days: int,
    max_items: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """按链接去重、丢弃过旧条目、按发布时间倒序、截断。"""
    reference = now or datetime.now(tz=UTC)
    cutoff = reference - timedelta(days=days)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for item in items:
        link = str(item.get("link") or "").strip()
        if not link or link in seen:
            continue
        published = item.get("published_at")
        if published:
            try:
                if datetime.fromisoformat(str(published)) < cutoff:
                    continue
            except ValueError:
                pass
        seen.add(link)
        kept.append(item)
    kept.sort(key=lambda entry: str(entry.get("published_at") or ""), reverse=True)
    return kept[:max_items]


def resolve_url(source: dict[str, Any], base_url: str) -> str:
    """配置里给绝对 url 就用它，否则拼 RSSHub 实例地址 + route。"""
    url = str(source.get("url") or "").strip()
    if url:
        return url
    route = str(source.get("route") or "").strip()
    if not route:
        return ""
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def fetch_xml(url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """抓取 RSS/Atom 文本，带 UA 与超时。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def build_snapshot(
    source: dict[str, Any],
    *,
    base_url: str,
    days: int,
    max_items: int,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """抓取并生成单个源的快照数据。"""
    url = resolve_url(source, base_url)
    if not url:
        raise ValueError(f"source {source.get('name')!r} 缺少 url/route")
    started = time.monotonic()
    xml_text = fetch_xml(url, timeout=timeout)
    raw_items = parse_feed(xml_text, limit=max(120, max_items * 4))
    items = normalize_items(raw_items, days=days, max_items=max_items)
    elapsed = time.monotonic() - started
    return {
        "source": str(source.get("name") or ""),
        "label": str(source.get("label") or source.get("name") or ""),
        "category": str(source.get("category") or "news"),
        "url": url,
        "fetched_at": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "item_count": len(items),
        "items": items,
    }


def load_config(path: Path) -> list[dict[str, Any]]:
    """读取路由配置（JSON：{"sources": [...]}）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = data.get("sources") if isinstance(data, dict) else data
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"配置文件 {path} 里没有有效的 sources 列表")
    return [source for source in sources if isinstance(source, dict)]


def write_snapshot(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """写单个源的快照文件（带换行、缩进，方便 diff）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(snapshot.get("source") or "source"))
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 RSSHub 路由/直连 RSS 并落盘快照")
    parser.add_argument("--config", default="config/rsshub_routes.json", help="路由配置文件")
    parser.add_argument("--base-url", default="http://localhost:1200", help="RSSHub 实例地址")
    parser.add_argument("--output-dir", default="data/cache", help="快照输出目录")
    parser.add_argument("--days", type=int, default=14, help="只保留最近 N 天的条目")
    parser.add_argument("--max-items", type=int, default=30, help="每个源最多保留条数")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT, help="单请求超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只抓取并打印，不写文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    try:
        sources = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.exception("读取配置失败: %s", config_path)
        return 1

    ok = 0
    failed = 0
    for source in sources:
        name = str(source.get("name") or "?")
        try:
            snapshot = build_snapshot(
                source,
                base_url=args.base_url,
                days=args.days,
                max_items=args.max_items,
                timeout=args.timeout,
            )
        except (urllib.error.URLError, OSError, ValueError, ET.ParseError) as exc:
            failed += 1
            LOG.warning("源 %s 抓取失败，保留旧快照: %s", name, exc)
            continue

        ok += 1
        LOG.info(
            "源 %s: %d 条，耗时 %.2fs%s",
            name,
            snapshot["item_count"],
            snapshot["elapsed_seconds"],
            "（dry-run，不写文件）" if args.dry_run else "",
        )
        if not args.dry_run:
            path = write_snapshot(snapshot, output_dir)
            LOG.info("已写入 %s", path)

    LOG.info("完成：成功 %d 个源，失败 %d 个源", ok, failed)
    if failed and not ok:
        LOG.error("所有源都失败（可能 RSSHub 实例未就绪）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
