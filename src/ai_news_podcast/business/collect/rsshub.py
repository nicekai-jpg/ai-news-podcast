"""C3 RSSHub 通道：把配置里的 route 拼成临时 RSSHub 实例的 URL。

RSSHub 由 daily workflow 临时起容器（``http://127.0.0.1:1200``），本模块只负责：

- ``build_feed_url``：base_url + route → 可抓取 URL（含中文/参数 route 的百分号编码）；
- ``probe_rsshub``：容器就绪探测（``/healthz`` 或根路径可访问即视为就绪）；
- ``fetch_rsshub``：复用通用 feed 解析（RSSHub 输出以 RSS 2.0 为主）。

base_url 不写死：由 ``FetchContext.rsshub_base`` 注入，本地调试 / CI 容器均可复用。
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from ai_news_podcast.business.contracts import Item

from . import rss
from .base import CollectError, ConfigError, FetchContext, SourceConfig, try_fetch_text

logger = logging.getLogger(__name__)

DEFAULT_RSSHUB_BASE = "http://127.0.0.1:1200"
_HEALTH_PATHS = ("/healthz", "/")
# route 里的 / 与 query 分隔符保留原样，其余非 ASCII 字符百分号编码。
_ROUTE_SAFE_CHARS = "/:@!$&'()*+,;=~-._"


def build_feed_url(rsshub_base: str, route: str) -> str:
    """``http://host:1200`` + ``/qbitai/category/资讯`` → 可直接抓取的 URL。"""
    base = (rsshub_base or "").strip()
    if not base:
        raise ConfigError("RSSHub base_url 不能为空")
    raw_route = (route or "").strip()
    if not raw_route:
        raise ConfigError("RSSHub route 不能为空")
    if not raw_route.startswith("/"):
        raw_route = "/" + raw_route
    path, separator, query = raw_route.partition("?")
    encoded = quote(path, safe=_ROUTE_SAFE_CHARS)
    return f"{base.rstrip('/')}{encoded}{separator}{query}"


def probe_rsshub(ctx: FetchContext, *, rsshub_base: str | None = None) -> bool:
    """探测 RSSHub 容器是否就绪；未配置或不可达返回 ``False``（不抛异常）。"""
    base = (rsshub_base or ctx.rsshub_base or "").strip()
    if not base:
        logger.info("未配置 RSSHub base_url，C3 通道跳过")
        return False
    for path in _HEALTH_PATHS:
        if try_fetch_text(ctx, f"{base.rstrip('/')}{path}", attempts=1) is not None:
            logger.info("RSSHub 就绪: %s", base)
            return True
    logger.warning("RSSHub 容器未就绪，C3 通道跳过: %s", base)
    return False


def fetch_rsshub(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    """抓取单个 RSSHub 路由（容器未就绪时抛异常，由 runner 标记 skipped / failover）。"""
    if not ctx.rsshub_base:
        raise CollectError(f"RSSHub 未配置，跳过源 {source.name}")
    if not source.rsshub_route:
        raise ConfigError(f"源 {source.name} 缺少 rsshub_route")
    url = build_feed_url(ctx.rsshub_base, source.rsshub_route)
    return rss.fetch_feed(source, ctx, urls=[url])
