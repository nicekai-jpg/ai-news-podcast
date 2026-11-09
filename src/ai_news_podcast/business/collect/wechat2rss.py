"""C5 wechat2rss 通道：0 成本公共公众号 feed。

wechat2rss（``https://wechat2rss.xlab.app``）的 feed 每条都带 ``content:encoded``
全文，因此复用 ``rss`` 的解析逻辑即可；公众号文章页对抓取不友好（反爬），所以
**不**回源抓文章页，正文只依赖 feed 自带的 ``content:encoded``。

17 个 AI 相关公众号的 feed id 见 ``config/sources_v3.yaml``。
"""

from __future__ import annotations

from ai_news_podcast.business.contracts import Item

from . import rss
from .base import ConfigError, FetchContext, SourceConfig

DEFAULT_WECHAT2RSS_BASE = "https://wechat2rss.xlab.app/feed"


def build_feed_url(feed_id: str, *, base_url: str = DEFAULT_WECHAT2RSS_BASE) -> str:
    """公众号 feed id → wechat2rss feed URL；传入完整 URL 时原样返回。"""
    value = (feed_id or "").strip()
    if not value:
        raise ConfigError("wechat2rss feed_id 不能为空")
    if value.startswith(("http://", "https://")):
        return value
    return f"{base_url.rstrip('/')}/{value}.xml"


def fetch_wechat2rss(source: SourceConfig, ctx: FetchContext) -> list[Item]:
    """抓取单个公众号 feed（正文来自 ``content:encoded``，不回源）。"""
    urls = source.page_urls()
    if not urls:
        raise ConfigError(f"源 {source.name} (c5_wechat2rss) 缺少 url")
    return rss.fetch_feed(source, ctx, urls=urls, allow_page_fetch=False)
