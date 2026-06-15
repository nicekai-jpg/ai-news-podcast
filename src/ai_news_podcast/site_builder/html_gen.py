import datetime
import json
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_STATIC_DIR = Path(__file__).parent / "static"


def _read_static(filename: str) -> str:
    return (_STATIC_DIR / filename).read_text(encoding="utf-8")


def format_friendly_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        dt = parsedate_to_datetime(date_str)
        try:
            dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
        except (ValueError, TypeError):
            pass
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        try:
            dt = datetime.datetime.fromisoformat(date_str)
            try:
                dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
            except (ValueError, TypeError):
                pass
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
      <div class="header-right">
        <a href="./feed.xml" target="_blank" class="nav-btn nav-btn-rss">RSS 订阅</a>
      </div>
    </header>

    <div class="hero-section">
      <div class="hero-inner">
        <img src="./logo.png" alt="Cover" class="hero-cover">
        <div class="hero-meta">
          <h1 class="hero-title" id="hero-title">AI 新闻快报</h1>
          <p class="hero-desc" id="hero-desc">AI 播客电台 · 每日 AI 资讯的声音解说与剧本展示</p>
        </div>
      </div>
    </div>

    <div class="brand-switcher-wrap">
      <div class="brand-switcher">
        <button id="btn-mode-podcast" class="mode-btn active" onclick="switchMode('podcast')">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v1a7 7 0 0 1-14 0v-1"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          AI 播客电台
        </button>
        <button id="btn-mode-report" class="mode-btn" onclick="switchMode('report')">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          科技日报
        </button>
      </div>
    </div>

    <div class="date-selector-wrap"><div class="date-pills" id="date-pills"></div></div>

    <div class="main-layout">
      <!-- AI 播客模式布局 -->
      <div class="podcast-workspace" id="panel-podcast">
        <!-- 左侧播客播控台 -->
        <div class="podcast-sidebar">
          <div class="station-card">
            <div class="vinyl-wrapper">
              <div class="vinyl-disc" id="vinyl-disc">
                <img src="./logo.png" alt="Album Art" class="vinyl-art">
                <div class="vinyl-center"></div>
              </div>
              <div class="visualizer-waves" id="visualizer-waves">
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
              </div>
            </div>

            <div class="podcast-info">
              <h3 class="side-title" id="side-podcast-title">AI 新闻快报</h3>
              <p class="side-meta-date" id="podcast-date-tag">—</p>
            </div>

            <div class="console-player">
              <div class="playback-mode-switcher">
                <button id="playback-btn-full" class="playback-mode-btn active" onclick="setPlaybackMode('full')">📻 整轨广播</button>
                <button id="playback-btn-sentence" class="playback-mode-btn" onclick="setPlaybackMode('sentence')" style="display: none;">📖 智能句读</button>
              </div>
              <div class="console-time-row">
                <span id="current-time">0:00</span>
                <div class="console-progress-track" id="console-progress-track" onclick="seekAudio(event)">
                  <div class="console-progress-fill" id="console-progress-fill"></div>
                  <div class="console-progress-handle"></div>
                </div>
                <span id="total-time">0:00</span>
              </div>
              <div class="console-controls">
                <button class="ctrl-btn" onclick="skipAudio(-15)" title="-15秒">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M2.5 2v6h6M2.66 15.57a10 10 0 1 0-.57-8.38l.41 1.31"/></svg>
                </button>
                <button class="console-play-btn" id="console-btn-play" onclick="toggleAudio()">▶</button>
                <button class="ctrl-btn" onclick="skipAudio(15)" title="+15秒">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1 .57-8.38l-.41 1.31"/></svg>
                </button>
              </div>
              <div class="console-extra">
                <span id="console-speed-btn" onclick="cycleSpeed()">1.0x</span>
                <div class="volume-control">
                  <svg id="volume-icon" onclick="toggleMute()" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
                  <input type="range" id="volume-slider" min="0" max="1" step="0.05" value="0.8" oninput="changeVolume(this.value)">
                </div>
              </div>
            </div>
          </div>

          <!-- 参考源卡片 -->
          <div class="sources-card" id="sources-card">
            <div class="card-header">
              <span>🔗 本期引证新闻源</span>
            </div>
            <div class="sources-list-body" id="sources-list-body"></div>
          </div>
        </div>

        <!-- 右侧剧本 -->
        <div class="podcast-script-pane">
          <div class="pane-header">
            <span>🎙️ 播客转写同步剧本</span>
            <span class="teleprompter-badge">智能音文联动</span>
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

      <!-- 科技日报模式布局 -->
      <div class="report-workspace" id="panel-report" style="display: none;">
        <div class="magazine-container">
          <div class="magazine-header">
            <div class="magazine-issue" id="report-date-tag">—</div>
            <h1 class="magazine-title" id="magazine-main-title">🌍 科技新闻日报</h1>
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
        .replace("{build_time}", str(build_time))
    )

    _write_text(site_dir / "index.html", html)
