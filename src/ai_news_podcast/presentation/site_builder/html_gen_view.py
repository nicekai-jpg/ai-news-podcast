"""站点首页生成：骨架 + 内联 site-data + 内联 style/app/player。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_news_podcast.data.config.models_dao import AppConfig

_STATIC_DIR = Path(__file__).parent / "static"
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


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <header class="site-header">
    <div class="header-inner">
      <div class="brand-group">
        <img src="./logo.png" alt="{title}" class="brand-logo-img">
        <div class="brand-text">
          <div class="brand-title">{title}<span class="brand-badge" id="issue-badge">—</span></div>
          <div class="brand-subtitle">每日 AI 情报刊</div>
        </div>
      </div>
      <nav class="header-nav">
        <a href="#section-hero" class="nav-anchor-link active">🎧 播客速览</a>
        <a href="#section-daily" class="nav-anchor-link">📰 科技日报</a>
        <a href="#section-sources" class="nav-anchor-link" id="nav-sources">🌐 来源速递</a>
        <a href="#section-radar" class="nav-anchor-link">📡 项目雷达</a>
      </nav>
      <div class="header-actions">
        <a href="./feed.xml" target="_blank" class="btn-header-pill btn-header-rss">📡 RSS 订阅</a>
      </div>
    </div>
  </header>

  <main class="app-layout">
    <section class="shared-date-axis" aria-label="日期选择">
      <div class="date-nav-controls">
        <button class="date-step-btn" id="date-prev" aria-label="前一日">← 前一日</button>
        <button class="date-step-btn" id="date-next" aria-label="后一日">后一日 →</button>
      </div>
      <div class="date-pills-carousel" id="date-pills"></div>
      <div class="date-archive-link" id="date-latest" hidden>回到最新一期</div>
    </section>

    <section id="section-hero" class="content-section" aria-label="播客速览">
      <div class="hero-chronicle-grid">
        <div class="podcast-hero-card" id="podcast-card">
          <div class="podcast-cover-meta">
            <div class="podcast-disc-cover"><div class="disc-wave-bars">
              <div class="disc-wave-bar"></div><div class="disc-wave-bar"></div>
              <div class="disc-wave-bar"></div><div class="disc-wave-bar"></div>
              <div class="disc-wave-bar"></div>
            </div></div>
            <div class="podcast-title-area">
              <div class="podcast-tag-row">
                <span class="podcast-tag">双主持对谈</span>
                <span class="podcast-duration" id="podcast-duration">—</span>
              </div>
              <h2 class="podcast-main-heading" id="podcast-title">AI 新闻快报</h2>
              <div class="podcast-subhead" id="podcast-subtitle"></div>
            </div>
          </div>
          <div class="host-control-box">
            <div class="host-box-header">
              <span>双主持音色设定</span>
              <span class="speaker-live-pill" id="live-speaker-badge"><span class="badge-dot">●</span> 待播放</span>
            </div>
            <div class="host-rows">
              <div class="host-voice-row">
                <div class="host-identity"><div class="host-avatar host-a">A</div><span class="host-name">苏晴</span></div>
                <div class="voice-segmented" id="host-a-voices"></div>
              </div>
              <div class="host-voice-row">
                <div class="host-identity"><div class="host-avatar host-b">B</div><span class="host-name">周航</span></div>
                <div class="voice-segmented" id="host-b-voices"></div>
              </div>
            </div>
          </div>
          <div class="podcast-cta-row">
            <button class="btn-play-primary" id="hero-play-btn" aria-pressed="false">
              <span id="hero-play-icon">▶</span><span id="hero-play-text">立即收听本期</span>
            </button>
            <button class="btn-transcript-cta" id="hero-transcript-btn">📜 逐字稿</button>
          </div>
        </div>

        <div class="editorial-lead-card">
          <div class="editorial-header-stamp">
            <div class="stamp-label">✨ 今日总览导语</div>
            <div class="stamp-date" id="lead-date">—</div>
          </div>
          <div class="lead-quote-text" id="lead-text"></div>
          <div class="metrics-summary-strip" id="metrics-strip"></div>
          <div class="chapters-row" id="chapters-row"></div>
        </div>
      </div>
    </section>

    <section id="section-daily" class="content-section" aria-label="科技日报">
      <div class="daily-reading-layout">
        <div class="section-header-row">
          <div class="section-tag-title"><span>📰</span> 《科技日报》深度解读
            <span class="section-counter-badge" id="daily-counter"></span></div>
          <div class="section-desc-hint">黄金阅读行宽 760px · 结构化长篇研判</div>
        </div>
        <article class="daily-report-wrapper">
          <div class="daily-meta-bar">
            <div class="daily-meta-left">
              <span class="daily-pill">深度研判</span>
              <span class="daily-read-time" id="daily-meta"></span>
            </div>
            <div class="daily-toc-pills" id="daily-toc"></div>
          </div>
          <div id="daily-report-body"></div>
        </article>
      </div>
    </section>

    <section id="section-sources" class="content-section" aria-label="来源速递">
      <div class="section-header-row">
        <div class="section-tag-title"><span>🌐</span> 今日参考信源速览
          <span class="section-counter-badge" id="source-count-badge">—</span></div>
        <div class="section-desc-hint">按媒体分类聚合 · 支持实时检索与全文直达</div>
      </div>
      <div class="sources-controls-row">
        <div class="source-filter-pills" id="source-filters"></div>
        <div class="source-search-box">
          <span class="source-search-icon">🔍</span>
          <input type="text" class="source-search-input" id="source-search"
                 placeholder="检索标题或来源…" aria-label="检索标题或来源">
        </div>
      </div>
      <div class="sources-grid-layout" id="sources-grid"></div>
      <div class="source-expand-bar" id="sources-expand-wrap" hidden>
        <button class="btn-expand-sources" id="btn-expand-sources">展开全部信源 ▾</button>
      </div>
    </section>

    <section id="section-radar" class="content-section" aria-label="项目雷达">
      <div class="section-header-row">
        <div class="section-tag-title"><span>📡</span> GitHub 项目雷达
          <span class="section-counter-badge" id="radar-badge">—</span></div>
        <div class="section-desc-hint" id="radar-rules"></div>
      </div>
      <div class="radar-card-container" id="radar-root"></div>
    </section>
  </main>

  <footer class="docked-audio-player" id="docked-player">
    <div class="player-inner">
      <div class="player-left">
        <img src="./logo.png" alt="cover" class="player-thumb-img">
        <div class="player-track-info">
          <div class="player-track-title" id="player-title">AI 新闻快报</div>
          <div class="player-live-sentence" id="player-sentence" aria-live="polite">点击播放开始收听…</div>
        </div>
      </div>
      <div class="player-center">
        <div class="player-button-row">
          <button class="btn-player-skip" id="skip-back" aria-label="后退 15 秒">⏪ 15s</button>
          <button class="btn-player-play" id="dock-play-btn" aria-label="播放或暂停" aria-pressed="false"><span id="dock-play-icon">▶</span></button>
          <button class="btn-player-skip" id="skip-forward" aria-label="前进 15 秒">15s ⏩</button>
        </div>
        <div class="player-scrubber-row">
          <span class="player-time" id="time-current">00:00</span>
          <div class="scrubber-track" id="scrubber-track" role="slider" tabindex="0" aria-label="播放进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="scrubber-progress" id="scrubber-fill"><div class="scrubber-thumb"></div></div></div>
          <span class="player-time" id="time-duration">00:00</span>
        </div>
      </div>
      <div class="player-right">
        <button class="btn-speed" id="speed-btn" aria-label="播放倍速">1.0x</button>
        <div class="volume-box">
          <button class="btn-vol" id="mute-btn" aria-label="静音或恢复音量">🔊</button>
          <input type="range" class="vol-slider" id="vol-slider" min="0" max="1" step="0.05" value="0.8">
        </div>
        <button class="btn-transcript-drawer-toggle" id="btn-drawer-toggle" aria-label="打开逐字稿" aria-expanded="false">📜 逐字稿</button>
      </div>
    </div>
  </footer>

  <aside class="transcript-drawer" id="transcript-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" aria-hidden="true">
    <div class="drawer-header">
      <div class="drawer-title" id="drawer-title"><span>🎙️</span> 对话逐字稿 <span class="brand-badge" id="drawer-count"></span></div>
      <button class="drawer-close-btn" id="drawer-close" aria-label="关闭逐字稿">✕</button>
    </div>
    <div class="drawer-filters">
      <div class="drawer-filter-group">
        <button class="btn-drawer-filter active" data-speaker="all" aria-pressed="true">全部</button>
        <button class="btn-drawer-filter" data-speaker="A" aria-pressed="false">只看苏晴</button>
        <button class="btn-drawer-filter" data-speaker="B" aria-pressed="false">只看周航</button>
      </div>
      <span class="drawer-hint">点击任意句子快速跳转播放</span>
    </div>
    <div class="drawer-body" id="drawer-chunk-list"></div>
  </aside>

  <div class="toast-pill" id="toast-pill" role="status" aria-live="polite"></div>
  <audio id="main-audio" preload="metadata"></audio>

  <script type="application/json" id="site-data">{site_data}</script>
  <script>{app_js}</script>
  <script>{player_js}</script>
</body>
</html>
"""


def build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
    cfg: dict[str, Any] | AppConfig | None = None,
) -> None:
    dates_list: list[str] = []
    episodes_map: dict[str, dict[str, str]] = {}
    for episode in episodes:
        ep_id = str(episode.get("id") or episode.get("guid") or "").rsplit("/", 1)[-1]
        ep_id = ep_id.replace(".mp3", "")
        if not ep_id or ep_id in episodes_map:
            continue
        dates_list.append(ep_id)
        episodes_map[ep_id] = {
            "title": str(episode.get("title") or f"AI 新闻快报 | {ep_id}"),
            "mp3": str(episode.get("enclosure_url") or f"{base_url}/episodes/{ep_id}.mp3"),
        }
    dates_list.sort(reverse=True)

    site_data = {
        "dates": dates_list,
        "episodes": episodes_map,
        "voices": _voice_labels(cfg),
    }
    html = _PAGE_TEMPLATE
    for placeholder, value in (
        ("{title}", str(podcast_title)),
        ("{css}", _read_static("style.css")),
        ("{app_js}", _read_static("app.js")),
        ("{player_js}", _read_static("player.js")),
        ("{site_data}", _safe_json(site_data)),
    ):
        html = html.replace(placeholder, value)
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text(html, encoding="utf-8")
