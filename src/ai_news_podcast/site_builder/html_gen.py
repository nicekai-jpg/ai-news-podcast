from __future__ import annotations

import contextlib
import datetime
import json
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_news_podcast.config.models import AppConfig

_STATIC_DIR = Path(__file__).parent / "static"


def _read_static(filename: str) -> str:
    return (_STATIC_DIR / filename).read_text(encoding="utf-8")


def format_friendly_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        dt = parsedate_to_datetime(date_str)
        with contextlib.suppress(ValueError, TypeError):
            dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        try:
            dt = datetime.datetime.fromisoformat(date_str)
            with contextlib.suppress(ValueError, TypeError):
                dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return date_str


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
    cfg: dict[str, Any] | AppConfig | None = None,
) -> None:
    dates_list = []
    episodes_map: dict[str, dict[str, Any]] = {}

    for ep in episodes:
        ep_id = ep.get("id", "") or ep.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        if not ep_id:
            continue
        dates_list.append(ep_id)
        episodes_map[ep_id] = ep

    dates_json = json.dumps(dates_list)
    episodes_map_json = json.dumps(
        {
            ep_id: {
                "title": ep.get("title", f"AI 新闻快报 | {ep_id}"),
                "mp3": ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3"),
                "desc": ep.get("description", ""),
            }
            for ep_id, ep in episodes_map.items()
        }
    )

    if cfg is None:
        cfg = {}
    if isinstance(cfg, dict):
        voices_config = cfg.get("tts", {}).get(
            "voice_names",
            {
                "host_a": {"professional": "亲切女声", "lively": "青春女声"},
                "host_b": {"professional": "专业男声", "lively": "活力男声"},
            },
        )
    else:
        # cfg is AppConfig
        voices_config = {
            "host_a": {"professional": "亲切女声", "lively": "青春女声"},
            "host_b": {"professional": "专业男声", "lively": "活力男声"},
        }
    voices_config_json = json.dumps(voices_config, ensure_ascii=False)

    shanghai_tz = ZoneInfo("Asia/Shanghai")
    build_time = datetime.datetime.now(tz=shanghai_tz).strftime("%Y-%m-%d %H:%M:%S")

    css_content = _read_static("style.css")
    js_content = _read_static("player.js")

    html_template = (
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
"""
        + css_content
        + """
  </style>
</head>
<body>
  <div class="ambient-glow-1"></div>
  <div class="container">
    <header>
      <a href="./" class="logo-block">
        <img src="./logo.png" alt="Logo" class="logo-img">
        <div><div class="brand-name">{podcast_title}</div><div class="brand-tagline">AI 前沿动态</div></div>
      </a>

      <!-- 顶级双 Tab 页切换选项卡 (Two Top Tab Pages Navigation) -->
      <nav class="top-tab-nav">
        <button id="btn-mode-podcast" class="top-tab-btn active" onclick="switchMode('podcast')">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v1a7 7 0 0 1-14 0v-1"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          🎙️ AI 播客电台
        </button>
        <button id="btn-mode-report" class="top-tab-btn" onclick="switchMode('report')">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          📰 科技日报
        </button>
      </nav>

      <div class="header-right">
        <a href="./feed.xml" target="_blank" class="nav-btn nav-btn-rss">RSS 订阅</a>
      </div>
    </header>

    <div class="main-layout">
      <!-- ==================== Tab 页 1：🎙️ AI 播客电台页面 ==================== -->
      <div class="tab-page-workspace" id="panel-podcast">
        <!-- 播客专属节目历程日历 -->
        <div class="date-selector-wrap">
          <div class="studio-calendar-header">
            <span>📅 播客期数日历 (Podcast Program Calendar)</span>
            <span class="calendar-sub-badge">点击切换播客对谈单集</span>
          </div>
          <div class="date-pills" id="date-pills"></div>
        </div>

        <!-- 播客主体：左右 2 栏黄金比例网格 (2 Column Grid Matching Concept Mockup) -->
        <div class="podcast-grid-layout">
          <!-- 左栏：唱片主播放器卡片 + 声轨音色卡片 -->
          <div class="podcast-left-column">
            <!-- 1. 唱片主播放器卡片 -->
            <div class="studio-player-card">
              <div class="vinyl-display-box">
                <div class="vinyl-disc-lg" id="vinyl-disc">
                  <img src="./logo.png" alt="Album Art" class="vinyl-art">
                  <div class="vinyl-center"></div>
                </div>
                <div class="visualizer-waves" id="visualizer-waves">
                  <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                  <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
                </div>
              </div>

              <div class="player-meta-box">
                <span class="live-broadcast-badge"><span class="live-dot"></span> LIVE BROADCAST</span>
                <h2 class="ep-main-title" id="side-podcast-title">AI 新闻快报</h2>
                <span class="ep-date-sub" id="podcast-date-tag">—</span>
              </div>

              <!-- 播放时间进度条 -->
              <div class="player-progress-area">
                <div class="time-row">
                  <span id="current-time" class="time-text">0:00</span>
                  <div class="console-progress-track" id="console-progress-track" onclick="seekAudio(event)">
                    <div class="console-progress-fill" id="console-progress-fill"></div>
                    <div class="console-progress-handle"></div>
                  </div>
                  <span id="total-time" class="time-text">0:00</span>
                </div>
              </div>

              <!-- 播控按键栏 -->
              <div class="player-ctrl-row">
                <button class="ctrl-btn" onclick="skipAudio(-15)" title="-15秒">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M2.5 2v6h6M2.66 15.57a10 10 0 1 0-.57-8.38l.41 1.31"/></svg>
                </button>
                <button class="console-play-btn" id="console-btn-play" onclick="toggleAudio()">▶</button>
                <button class="ctrl-btn" onclick="skipAudio(15)" title="+15秒">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1 .57-8.38l-.41 1.31"/></svg>
                </button>
                <span id="console-speed-btn" class="speed-pill-btn" onclick="cycleSpeed()">1.0x</span>
                <div class="volume-control">
                  <svg id="volume-icon" onclick="toggleMute()" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
                  <input type="range" id="volume-slider" min="0" max="1" step="0.05" value="0.8" oninput="changeVolume(this.value)">
                </div>
              </div>
            </div>

            <!-- 2. 双声轨音色定制独立卡片 (Voice Pills Card) -->
            <div class="voice-pills-card">
              <div class="voice-card-header">
                <span>🎙️ 双声轨音色定制 (Voice Pills)</span>
              </div>
              <div class="host-voice-rows">
                <div class="console-host-item host-a-box" id="host-card-a">
                  <div class="host-item-left">
                    <span class="host-emoji">👩‍💼</span>
                    <span class="host-name-sm">苏晴 <span class="host-badge-a">Female Host</span></span>
                  </div>
                  <div class="host-voice-segmented" id="host-a-voice-pills"></div>
                </div>
                <div class="console-host-item host-b-box" id="host-card-b">
                  <div class="host-item-left">
                    <span class="host-emoji">👨‍💼</span>
                    <span class="host-name-sm">周航 <span class="host-badge-b">Male Host</span></span>
                  </div>
                  <div class="host-voice-segmented" id="host-b-voice-pills"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- 右栏：对谈提词器卡片 + 新闻源 -->
          <div class="podcast-right-column">
            <div class="podcast-pane-full">
              <div class="pane-header">
                <span>🎙️ 播客对谈剧本 (Script Teleprompter)</span>
                <span class="teleprompter-badge">🔴 智能音文同步 · 点击段落跳转</span>
              </div>

              <!-- 本期核心研判卡片 -->
              <div class="episode-insight-banner" id="episode-insight-banner">
                <div class="insight-badge">💡 本期核心研判</div>
                <div class="insight-text" id="insight-text-content">每天 5 分钟，聚合 AI 领域最新发布、技术进展与行业观察。</div>
              </div>

              <div class="transcript-container-wrapper" style="position: relative; flex: 1; overflow: hidden; display: flex; flex-direction: column;">
                <div class="transcript-container" id="cast-panel-body" onscroll="handleTranscriptScroll()"></div>
                <!-- 人机共存滚动打断悬浮按钮 -->
                <button class="back-to-sync-btn" id="back-to-sync-btn" onclick="resumeSyncScroll()">
                  <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="18 15 12 9 6 15"/></svg>
                  返回播音位置
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 底部 100% 全宽：本期引证新闻源 Section -->
        <div class="sources-workspace" id="sources-card">
          <div class="sources-section-header">
            <div class="section-title">
              <span class="title-icon">🔗</span>
              <span class="title-text">本期引证新闻源</span>
              <span class="title-en">Verified Sources</span>
            </div>
            <span class="sources-count-badge" id="sources-count-tag">权威数据源交叉验证</span>
          </div>
          <div class="sources-list-body" id="sources-list-body"></div>
        </div>
      </div>

      <!-- ==================== Tab 页 2：📰 科技日报页面 ==================== -->
      <div class="tab-page-workspace" id="panel-report" style="display: none;">
        <!-- 日报专属出版历程日历 -->
        <div class="date-selector-wrap">
          <div class="studio-calendar-header">
            <span>📅 日报出版日历 (Daily Publishing Calendar)</span>
            <span class="calendar-sub-badge">点击切换往期深度科技日报</span>
          </div>
          <div class="date-pills" id="report-date-pills"></div>
        </div>

        <!-- 科技日报杂志排版 (零任何播放器控件) -->
        <div class="magazine-container">
          <div class="magazine-header">
            <div class="magazine-issue" id="report-date-tag">—</div>
            <h1 class="magazine-title" id="magazine-main-title">✨ 科技新闻日报</h1>
            <div class="magazine-divider"></div>
          </div>

          <div class="magazine-body">
            <div class="magazine-main-content" id="report-panel-body"></div>

            <div class="editor-section-wrapper" id="editor-verdict-wrapper" style="display: none;">
              <div class="editor-card">
                <div class="editor-header">
                  <div class="editor-avatar">💡</div>
                  <div>
                    <h4 class="editor-name">AI 小编独家锐评</h4>
                    <span class="editor-title">Editor's Verdict</span>
                  </div>
                </div>
                <div class="editor-quote" id="editor-quote-body"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <audio id="main-audio" style="display:none"></audio>
    <audio id="bgm-audio" loop style="display:none"></audio>
    <div class="toast" id="toast">✅ 已复制</div>
    <div class="footer"><p>{podcast_title} · 构建于 {build_time}</p></div>
  </div>

  <script>
"""
        + js_content
        + """
  </script>
</body>
</html>"""
    )

    html = (
        html_template.replace("{podcast_title}", str(podcast_title))
        .replace("{base_url}", str(base_url))
        .replace("{dates_json}", str(dates_json))
        .replace("{episodes_map_json}", str(episodes_map_json))
        .replace("{voices_config_json}", str(voices_config_json))
        .replace("{build_time}", str(build_time))
    )

    _write_text(site_dir / "index.html", html)
