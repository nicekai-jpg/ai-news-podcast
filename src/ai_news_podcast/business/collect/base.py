"""Stage 1 采集入海 —— 通道公共基础设施。

提供：
- ``SourceConfig`` / ``load_sources``：读取并校验 ``config/sources_v3.yaml``；
- ``FetchContext``：一次采集运行的上下文（日期、时间窗、HTTP 客户端、RSSHub 地址）；
- ``HttpClient``：httpx 封装，统一 UA / 超时 / 指数退避重试 / 同域限速；
- 通用 helper：HTML 清洗、正文抽取（readability）、语言/时间解析、item 构建，
  以及单源内的时效过滤 + URL 去重 + 条数上限。

只依赖 stdlib + 项目已有三方依赖；不 import ``business.pipeline`` 的实现。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import yaml
from bs4 import BeautifulSoup
from readability import Document

from ai_news_podcast.business.contracts import (
    CHANNELS,
    FULL_TEXT_STATUS,
    Item,
    canonical_link,
    make_item_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 2
DEFAULT_RETRY_WAIT = 1.0
DEFAULT_THROTTLE_INTERVAL = 1.0

SUMMARY_MAX_CHARS = 500
FULL_TEXT_MIN_CHARS = 800
FULL_TEXT_MAX_CHARS = 12000
READABLE_MIN_CHARS = 80

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_FULL_DATE_RE = re.compile(r"(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")
_SHORT_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?")
_DATE_IN_URL_RE = re.compile(r"/(20\d{2})[-/]?(\d{2})[-/]?(\d{2})(?=[/_.-]|$)")


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class CollectError(Exception):
    """采集阶段统一异常基类；单源失败由 runner 隔离，不阻断其它源。"""


class ConfigError(CollectError):
    """源配置不合法（缺字段、channel 未知、name 重复等）。"""


class FetchError(CollectError):
    """HTTP 抓取失败（含重试耗尽）。"""

    def __init__(self, url: str, attempts: int, reason: str = "") -> None:
        detail = f"抓取失败（{attempts} 次）: {url}"
        if reason:
            detail = f"{detail} ({reason})"
        super().__init__(detail)
        self.url = url
        self.attempts = attempts


class ParseError(CollectError):
    """feed / HTML 解析失败。"""


# ---------------------------------------------------------------------------
# 源配置
# ---------------------------------------------------------------------------


@dataclass
class FailoverConfig:
    """主通道失败时的备用抓取方式（仅一次，不再级联）。"""

    channel: str = ""
    url: str = ""
    route: str = ""
    parser: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailoverConfig:
        return cls(
            channel=str(data.get("channel") or "").strip(),
            url=str(data.get("url") or "").strip(),
            route=str(data.get("route") or data.get("rsshub_route") or "").strip(),
            parser=str(data.get("parser") or "").strip(),
        )


@dataclass
class SourceConfig:
    """单个采集源的配置（``config/sources_v3.yaml`` 的一行）。"""

    name: str
    label: str
    channel: str
    category: str = "news"
    limit: int = 20
    enabled: bool = True
    url: str = ""
    urls: list[str] = field(default_factory=list)
    rsshub_route: str = ""
    parser: str = ""
    selectors: dict[str, str] = field(default_factory=dict)
    changelog_url: str = ""
    failover: FailoverConfig | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        defaults: dict[str, Any] | None = None,
    ) -> SourceConfig:
        merged: dict[str, Any] = dict(defaults or {})
        merged.update(data or {})
        name = str(merged.get("name") or "").strip()
        if not name:
            raise ConfigError(f"源配置缺少 name: {data!r}")
        raw_failover = merged.get("failover")
        urls = merged.get("urls") or []
        selectors = merged.get("selectors") or {}
        return cls(
            name=name,
            label=str(merged.get("label") or name).strip(),
            channel=str(merged.get("channel") or "").strip(),
            category=str(merged.get("category") or "news").strip(),
            limit=_coerce_int(merged.get("limit"), default=20),
            enabled=_coerce_bool(merged.get("enabled"), default=True),
            url=str(merged.get("url") or "").strip(),
            urls=[str(item).strip() for item in urls if str(item).strip()],
            rsshub_route=str(merged.get("rsshub_route") or merged.get("route") or "").strip(),
            parser=str(merged.get("parser") or "").strip(),
            selectors=(
                {str(key): str(value) for key, value in selectors.items()}
                if isinstance(selectors, dict)
                else {}
            ),
            changelog_url=str(merged.get("changelog_url") or "").strip(),
            failover=(
                FailoverConfig.from_dict(raw_failover) if isinstance(raw_failover, dict) else None
            ),
        )

    def page_urls(self) -> list[str]:
        """返回该源需要抓取的 URL 列表（``url`` 优先，其后是 ``urls``）。"""
        targets = [self.url] if self.url else []
        targets.extend(item for item in self.urls if item not in targets)
        return targets


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def load_sources(config_path: str | Path) -> list[SourceConfig]:
    """读取 ``sources_v3.yaml``，返回校验后的源列表（顺序与文件一致）。"""
    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"读取源配置失败: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"源配置格式错误（顶层应为 mapping）: {path}")
    defaults = payload.get("defaults") or {}
    rows = payload.get("sources") or []
    if not isinstance(rows, list):
        raise ConfigError(f"源配置 sources 应为列表: {path}")
    sources = [SourceConfig.from_dict(row, defaults=defaults) for row in rows if row]
    validate_sources(sources)
    return sources


def validate_sources(sources: Iterable[SourceConfig]) -> None:
    """校验源唯一性、channel 合法性、抓取地址完整性。"""
    names: set[str] = set()
    for source in sources:
        if source.name in names:
            raise ConfigError(f"源 name 重复: {source.name}")
        names.add(source.name)
        if source.channel not in CHANNELS:
            raise ConfigError(f"源 {source.name} 的 channel 非法: {source.channel!r}")
        if source.channel == "c3_rsshub":
            if not source.rsshub_route:
                raise ConfigError(f"源 {source.name} (c3_rsshub) 缺少 rsshub_route")
        elif not source.page_urls():
            raise ConfigError(f"源 {source.name} ({source.channel}) 缺少 url/urls")
        if source.failover is not None and source.failover.channel not in CHANNELS:
            raise ConfigError(
                f"源 {source.name} 的 failover.channel 非法: {source.failover.channel!r}"
            )


# ---------------------------------------------------------------------------
# 通用 helper
# ---------------------------------------------------------------------------


def truncate(text: str, limit: int) -> str:
    """按字符数截断（保留 ``...`` 后缀）。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def strip_html(value: str) -> str:
    """HTML → 纯文本（折叠空白）。"""
    text = (value or "").strip()
    if not text:
        return ""
    if "<" in text:
        try:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        except (TypeError, ValueError):
            text = _TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", unescape(text)).strip()


def detect_lang(text: str) -> str:
    """粗略语言检测：中文字符占比 > 10% 判为 zh，否则 en。"""
    ratio = len(_ZH_RE.findall(text)) / max(len(text), 1)
    return "zh" if ratio > 0.1 else "en"


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_iso_datetime(text: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_utc(parsed)


_KNOWN_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y年%m月%d日",
)


def _parse_known_format(text: str) -> datetime | None:
    for fmt in _KNOWN_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _parse_rfc_datetime(text: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return _ensure_utc(parsed)


def parse_datetime(value: str) -> datetime | None:
    """尽力解析发布时间字符串（ISO 8601 / RFC 2822 / 常见日期格式）。"""
    text = (value or "").strip()
    if not text:
        return None
    return _parse_iso_datetime(text) or _parse_known_format(text) or _parse_rfc_datetime(text)


def _make_date(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day, 12, 0, tzinfo=UTC)
    except ValueError:
        return None


def find_datetime(text: str, *, now: datetime | None = None) -> datetime | None:
    """从一段文本里找日期（列表页常见 ``2026-11-08`` / ``2026年11月8日``）。

    只给「月-日」时可传 ``now`` 推断年份；日期仅精确到天，统一按 UTC 12:00 近似
    （北京时间 20:00），避免时间窗把当天发布误判成旧闻。
    """
    raw = text or ""
    match = _FULL_DATE_RE.search(raw)
    if match:
        return _make_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _SHORT_DATE_RE.search(raw)
    if match is None or now is None:
        return None
    year = now.year
    month, day = int(match.group(1)), int(match.group(2))
    candidate = _make_date(year, month, day)
    if candidate is not None and candidate - now > timedelta(days=30):
        candidate = _make_date(year - 1, month, day)
    return candidate


def find_date_in_url(url: str) -> datetime | None:
    """从 URL 路径里找日期（如 ``/2026-11/09/c_xxx.htm``、``/20251109/``）。"""
    match = _DATE_IN_URL_RE.search(urlsplit(url or "").path)
    if match is None:
        return None
    return _make_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def extract_article_text(
    html: str,
    url: str = "",
    *,
    min_chars: int = READABLE_MIN_CHARS,
    max_chars: int = FULL_TEXT_MAX_CHARS,
) -> str:
    """用 readability 抽正文；失败时退回整页文本，过短视为抽取失败。"""
    summary_html = ""
    try:
        summary_html = Document(html).summary() or ""
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("readability 抽取失败(%s): %s", url, exc)
    text = strip_html(summary_html) if summary_html else ""
    if not text:
        text = strip_html(html)
    if len(text) < min_chars:
        return ""
    return truncate(text, max_chars)


def build_item(
    *,
    channel: str,
    source: str,
    source_label: str,
    category: str,
    title: str,
    link: str = "",
    published_at: str = "",
    summary: str = "",
    full_text: str = "",
    full_text_status: str = "none",
    lang: str = "",
    extra: dict[str, Any] | None = None,
    fetched_at: str = "",
) -> Item:
    """构造统一 ``contracts.Item``（item_id / normalized_link 统一生成）。"""
    clean_title = _WHITESPACE_RE.sub(" ", (title or "")).strip()
    clean_link = (link or "").strip()
    clean_summary = truncate(_WHITESPACE_RE.sub(" ", (summary or "")).strip(), SUMMARY_MAX_CHARS)
    clean_body = truncate((full_text or "").strip(), FULL_TEXT_MAX_CHARS)
    status = full_text_status if full_text_status in FULL_TEXT_STATUS else "none"
    return Item(
        item_id=make_item_id(clean_link, clean_title, source),
        channel=channel,
        source=source,
        source_label=source_label,
        category=category,
        title=clean_title,
        link=clean_link,
        normalized_link=canonical_link(clean_link),
        published_at=published_at or "",
        fetched_at=fetched_at or utc_now_iso(),
        summary=clean_summary,
        full_text=clean_body,
        full_text_status=status,
        lang=lang or detect_lang(f"{clean_title} {clean_summary}"),
        extra=dict(extra or {}),
    )


def within_window(published_at: str, *, now: datetime, window_hours: int) -> bool:
    """发布时间是否落在采集窗口内；无发布时间的条目保留（由 Stage 2 决定去留）。"""
    parsed = parse_datetime(published_at) if published_at else None
    if parsed is None:
        return True
    return parsed >= now - timedelta(hours=window_hours)


def _item_sort_key(item: Item) -> tuple[bool, datetime]:
    parsed = parse_datetime(item.published_at) if item.published_at else None
    return (parsed is not None, parsed or datetime.min.replace(tzinfo=UTC))


def finalize_items(
    items: Iterable[Item],
    source: SourceConfig,
    ctx: FetchContext,
) -> list[Item]:
    """单源收尾：时效过滤 → URL 去重 → 时间倒序 → 条数上限。"""
    kept: list[Item] = []
    seen: set[str] = set()
    for item in items:
        if not item.title or not item.link:
            continue
        if not within_window(item.published_at, now=ctx.now, window_hours=ctx.window_hours):
            continue
        key = item.normalized_link or item.item_id
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    kept.sort(key=_item_sort_key, reverse=True)
    if source.limit > 0:
        return kept[: source.limit]
    return kept


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


class HttpClient:
    """httpx 封装：统一 UA、超时、指数退避重试、同域限速。

    ``client`` 可注入（测试 / 复用连接池）；注入的客户端由调用方负责关闭。
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        retry_wait: float = DEFAULT_RETRY_WAIT,
        throttle_interval: float = DEFAULT_THROTTLE_INTERVAL,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=True)
        self._user_agent = user_agent
        self._timeout = timeout
        self._retries = max(0, retries)
        self._retry_wait = max(0.0, retry_wait)
        self._throttle_interval = max(0.0, throttle_interval)
        self._lock = threading.Lock()
        self._last_request: dict[str, float] = {}

    def _wait(self, domain: str) -> None:
        if self._throttle_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            delay = max(0.0, self._throttle_interval - (now - last))
            self._last_request[domain] = now + delay
        if delay > 0:
            time.sleep(delay)

    def get(self, url: str, *, attempts: int | None = None) -> Any:
        """GET 并返回响应；失败按指数退避重试，最终抛 ``FetchError``。"""
        total_attempts = max(1, attempts if attempts is not None else self._retries + 1)
        domain = (urlsplit(url).hostname or "").lower()
        last_error: Exception | None = None
        for attempt in range(total_attempts):
            self._wait(domain)
            try:
                response = self._client.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout,
                    follow_redirects=True,
                )
                response.raise_for_status()
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt + 1 < total_attempts:
                    time.sleep(self._retry_wait * (2**attempt))
            else:
                return response
        raise FetchError(url, total_attempts, str(last_error)) from last_error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


# ---------------------------------------------------------------------------
# 抓取上下文
# ---------------------------------------------------------------------------


@dataclass
class FetchContext:
    """一次采集运行的上下文。"""

    date_str: str
    window_hours: int = 72
    http_client: Any | None = None
    rsshub_base: str | None = None
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    retry_wait: float = DEFAULT_RETRY_WAIT
    throttle_interval: float = DEFAULT_THROTTLE_INTERVAL
    fetch_fulltext: bool = True
    now: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    http: HttpClient = field(init=False, repr=False)
    fetched_at: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.now.tzinfo is None:
            self.now = self.now.replace(tzinfo=UTC)
        self.fetched_at = self.now.isoformat()
        self.http = HttpClient(
            client=self.http_client,
            user_agent=self.user_agent,
            timeout=self.timeout,
            retries=self.retries,
            retry_wait=self.retry_wait,
            throttle_interval=self.throttle_interval,
        )

    def window_start(self) -> datetime:
        return self.now - timedelta(hours=self.window_hours)

    def close(self) -> None:
        self.http.close()


# ---------------------------------------------------------------------------
# 抓取 / 容错 helper（正文回源、RSSHub 探测等共用）
# ---------------------------------------------------------------------------


def fetch_bytes(ctx: FetchContext, url: str, *, attempts: int | None = None) -> bytes:
    """抓取 URL 的原始字节；失败抛 ``FetchError``（由调用方决定是否隔离）。"""
    response = ctx.http.get(url, attempts=attempts)
    content = getattr(response, "content", None)
    if content is None:
        return str(getattr(response, "text", "") or "").encode("utf-8")
    return content


def try_fetch_text(ctx: FetchContext, url: str, *, attempts: int | None = None) -> str | None:
    """抓取页面文本，失败返回 ``None``（正文回源 / 就绪探测等可忽略失败）。"""
    try:
        response = ctx.http.get(url, attempts=attempts)
    except CollectError as exc:
        logger.debug("可忽略的抓取失败 %s: %s", url, exc)
        return None
    text = getattr(response, "text", None)
    if text is None:
        content = getattr(response, "content", b"")
        text = content.decode("utf-8", errors="replace") if content else ""
    return str(text)
