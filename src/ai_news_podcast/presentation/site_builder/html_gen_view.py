"""站点首页生成：骨架 + 内联 site-data + 内联 style/app/player。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.data.text_utils_dao import episode_title

_STATIC_DIR = Path(__file__).parent / "static"
_LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(
    (_STATIC_DIR / "logo.svg").read_bytes()
).decode("ascii")
# 设计稿 v23 的排版依赖 Noto Serif/Sans SC（见 DESIGN.md §3.3 / §10.6）；
# CSS 里已备好系统字体回退，离线时自动降级，不影响可用性。
_FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Noto+Serif+SC:wght@500;700;900&"
    "family=Noto+Sans+SC:wght@400;500;600;700&display=swap"
)
_DEFAULT_VOICE_NAMES = {
    "host_a": {"professional": "亲切女声", "lively": "青春女声"},
    "host_b": {"professional": "专业男声", "lively": "活力男声"},
}


def _read_static(filename: str) -> str:
    return (_STATIC_DIR / filename).read_text(encoding="utf-8")


def _norm_host_key(key: str) -> str | None:
    k = str(key).strip().lower()
    if k in ("host_a", "a"):
        return "host_a"
    if k in ("host_b", "b"):
        return "host_b"
    return None


def voice_catalog(cfg: Any) -> dict[str, list[dict[str, Any]]]:
    """全部可选音色 + 是否已真实合成，供站点渲染「置灰但保留」的音色按钮。"""
    active = _voice_labels(cfg)
    catalog: dict[str, list[dict[str, Any]]] = {}
    for host, defaults in _DEFAULT_VOICE_NAMES.items():
        available = active.get(host, {})
        entries = [
            {
                "id": variant,
                "label": available.get(variant, label),
                "available": variant in available,
            }
            for variant, label in defaults.items()
        ]
        entries.extend(
            {"id": variant, "label": label, "available": True}
            for variant, label in available.items()
            if variant not in defaults
        )
        catalog[host] = entries
    return catalog


def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    """注入播放器的音色标签；按 synth_variants（每 host）收敛，按钮只指向真实存在的音频。"""
    if cfg is None:
        return {key: dict(value) for key, value in _DEFAULT_VOICE_NAMES.items()}

    if isinstance(cfg, dict):
        custom = cfg.get("tts", {}).get("voice_names")
        raw_variants = cfg.get("tts", {}).get("cosyvoice", {}).get("synth_variants") or []
    else:
        custom = None
        raw_variants = cfg.tts.cosyvoice.synth_variants

    labels = {key: dict(value) for key, value in _DEFAULT_VOICE_NAMES.items()}
    if isinstance(custom, dict):
        labels.update(custom)
    if isinstance(raw_variants, dict):
        allowed: dict[str, list[str]] = {}
        for key, values in raw_variants.items():
            host = _norm_host_key(key)
            if host is not None and isinstance(values, (list, tuple)):
                allowed[host] = [str(value).strip() for value in values if str(value).strip()]
    elif raw_variants:
        allowed = {
            "host_a": [str(value).strip() for value in raw_variants],
            "host_b": [str(value).strip() for value in raw_variants],
        }
    else:
        allowed = {}

    for host, allowed_variants in allowed.items():
        if allowed_variants and host in labels:
            labels[host] = {
                variant: name
                for variant, name in labels[host].items()
                if variant in allowed_variants
            }
    return labels


def _safe_json(payload: Any) -> str:
    """内联 JSON：转义 </ 防止提前闭合 script；ensure_ascii=False 保留中文。"""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _build_site_data(
    podcast_title: str, episodes: list[dict[str, Any]], base_url: str
) -> dict[str, Any]:
    """首页内联的站点元数据：日期列表（新→旧）+ 每期标题/音频地址。"""
    dates_list: list[str] = []
    episodes_map: dict[str, dict[str, str]] = {}
    for episode in episodes:
        ep_id = str(episode.get("id") or episode.get("guid") or "").rsplit("/", 1)[-1]
        ep_id = ep_id.replace(".mp3", "")
        if not ep_id or ep_id in episodes_map:
            continue
        dates_list.append(ep_id)
        episodes_map[ep_id] = {
            "title": str(episode.get("title") or episode_title(ep_id)),
            "mp3": str(episode.get("enclosure_url") or f"{base_url}/episodes/{ep_id}.mp3"),
        }
    dates_list.sort(reverse=True)
    return {"dates": dates_list, "episodes": episodes_map}


def build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
    cfg: dict[str, Any] | AppConfig | None = None,
) -> None:
    """用设计稿 v23 的页面骨架生成首页：内联 CSS / app.js / site-data。

    正文、日报、来源、雷达等数据由前端按日期拉取 digest JSON 后填充
    （设计稿约定的 {{...}} 注入点由 app.js 的 hydrate 完成）。
    """
    template = _read_static("page_template.html").replace('src="logo.svg"', f'src="{_LOGO_URI}"', 1)
    site_data = _build_site_data(podcast_title, episodes, base_url)
    html = (
        "<!doctype html>\n"
        '<html lang="zh-CN" data-theme="light">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>今日AI看点</title>\n"
        f'<link rel="icon" href="{_LOGO_URI}">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link href="{_FONT_CSS_URL}" rel="stylesheet">\n'
        f"<style>\n{_read_static('style.css')}\n</style>\n"
        "<script>\n"
        "(function() {\n"
        "  var saved = localStorage.getItem('theme');\n"
        "  var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;\n"
        "  document.documentElement.setAttribute('data-theme', saved || (prefersDark ? 'dark' : 'light'));\n"
        "})();\n"
        "</script>\n"
        "</head>\n"
        "<body>\n"
        f"{template}\n"
        '<script id="site-data" type="application/json">'
        f"{_safe_json(site_data)}</script>\n"
        f"<script>\n{_read_static('app.js')}\n</script>\n"
        "</body>\n"
        "</html>\n"
    )
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text(html, encoding="utf-8")
