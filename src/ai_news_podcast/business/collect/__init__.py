"""Stage 1 采集入海：C1–C5 通道抓取 → 统一 ``contracts.Item`` → raw artifact。

模块划分：

- ``base``：源配置、抓取上下文、HTTP 封装（UA/超时/重试/限速）、通用 helper；
- ``rss``：通用 RSS 2.0 / Atom 1.0 解析（含 ``content:encoded`` 全文）；
- ``rsshub``：C3 路由 → 临时 RSSHub 实例 URL、容器就绪探测；
- ``wechat2rss``：C5 公众号 feed（复用 rss 解析）；
- ``web``：C4 网页抓取（7 个 SSR 源，配置驱动，失败隔离）；
- ``runner``：编排、单源失败隔离、artifact 落盘。
"""

from ai_news_podcast.business.collect.base import (
    CollectError,
    ConfigError,
    FetchContext,
    FetchError,
    ParseError,
    SourceConfig,
    load_sources,
)
from ai_news_podcast.business.collect.runner import (
    CollectionResult,
    SourceStat,
    collect_all,
    collect_and_write,
)

__all__ = [
    "CollectError",
    "CollectionResult",
    "ConfigError",
    "FetchContext",
    "FetchError",
    "ParseError",
    "SourceConfig",
    "SourceStat",
    "collect_all",
    "collect_and_write",
    "load_sources",
]
