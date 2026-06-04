import datetime
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def format_friendly_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        dt = parsedate_to_datetime(date_str)
        try:
            dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
        except Exception:
            pass
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        try:
            dt = datetime.datetime.fromisoformat(date_str)
            try:
                dt = dt.astimezone(ZoneInfo("Asia/Shanghai"))
            except Exception:
                pass
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
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
    episodes_map_json = json.dumps({
        ep_id: {
            "title": ep.get("title", f"AI 新闻快报 | {ep_id}"),
            "mp3": ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3"),
        }
        for ep_id, ep in episodes_map.items()
    })

    shanghai_tz = ZoneInfo("Asia/Shanghai")
    build_time = datetime.datetime.now(tz=shanghai_tz).strftime("%Y-%m-%d %H:%M:%S")

    html_template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <link rel="alternate" type="application/rss+xml" title="{podcast_title}" href="./feed.xml">
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {{
      --bg: #07060d;
      --bg-gradient: radial-gradient(circle at 50% 0%, #150f30 0%, #07060d 70%);
      --surface: rgba(19, 17, 32, 0.55);
      --surface-hover: rgba(29, 26, 48, 0.75);
      --accent: #8b5cf6;
      --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
      --accent-glow: rgba(139, 92, 246, 0.35);
      --accent-light: #c084fc;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --text-dark: #6b7280;
      --border: rgba(255, 255, 255, 0.06);
      --border-focus: rgba(167, 139, 250, 0.35);
      --radius-sm: 10px;
      --radius-md: 16px;
      --radius-lg: 24px;
      --success: #10b981;
      --shadow-lg: 0 16px 40px -10px rgba(0, 0, 0, 0.6);
      --shadow-glow: 0 0 25px rgba(139, 92, 246, 0.15);
      --host-a: #06b6d4;
      --host-b: #f472b6;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
      background: var(--bg);
      background-image: var(--bg-gradient);
      background-attachment: fixed;
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }}

    .ambient-glow-1 {{
      position: fixed; top: -20%; right: 5%; width: 60vw; height: 60vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none; z-index: -1;
      animation: drift-1 30s ease-in-out infinite alternate;
    }}
    .ambient-glow-2 {{
      position: fixed; bottom: -15%; left: 5%; width: 50vw; height: 50vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(6, 182, 212, 0.08) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none; z-index: -1;
      animation: drift-2 25s ease-in-out infinite alternate;
    }}
    @keyframes drift-1 {{ 0% {{ transform: translate(0, 0) scale(1); }} 100% {{ transform: translate(-8%, 8%) scale(1.15); }} }}
    @keyframes drift-2 {{ 0% {{ transform: translate(0, 0) scale(1); }} 100% {{ transform: translate(8%, -8%) scale(1.1); }} }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 24px 180px;
    }}

    /* Header */
    header {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 20px 0; margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100;
      background: rgba(7, 6, 13, 0.4);
      backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    }}
    .logo-block {{
      display: flex; align-items: center; gap: 14px;
      text-decoration: none; color: #fff; transition: opacity 0.2s;
    }}
    .logo-block:hover {{ opacity: 0.9; }}
    .logo-img {{
      width: 44px; height: 44px; border-radius: 12px;
      border: 1px solid var(--border); box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .brand-name {{
      font-size: 1.4rem; font-weight: 800; letter-spacing: -0.03em;
      background: linear-gradient(to right, #fff, #e5e7eb);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .nav-links {{ display: flex; gap: 16px; }}
    .nav-links a {{
      display: inline-flex; align-items: center; gap: 8px;
      color: var(--text-muted); text-decoration: none;
      font-size: 0.9rem; font-weight: 600;
      padding: 8px 16px; border-radius: 99px;
      background: rgba(255,255,255,0.02); border: 1px solid var(--border);
      transition: all 0.2s ease;
    }}
    .nav-links a:hover {{
      color: #fff; background: rgba(255,255,255,0.06);
      border-color: var(--border-focus); transform: translateY(-1px);
    }}
    .nav-links svg {{ width: 16px; height: 16px; }}

    /* Main Layout: Date List + Content */
    .layout-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      margin-top: 24px;
    }}
    @media (min-width: 960px) {{
      .layout-grid {{
        grid-template-columns: 200px 1fr;
      }}
    }}

    /* Date List Sidebar */
    .date-list {{
      background: var(--surface);
      backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 20px 16px;
      box-shadow: var(--shadow-lg);
      max-height: calc(100vh - 140px);
      position: sticky; top: 100px;
      overflow-y: auto;
    }}
    .date-list h2 {{
      font-size: 1rem; font-weight: 700; color: #fff;
      margin-bottom: 16px; padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .date-list ul {{
      list-style: none; display: flex; flex-direction: column; gap: 4px;
    }}
    .date-item {{
      display: flex; align-items: center; gap: 8px;
      padding: 10px 12px; border-radius: var(--radius-sm);
      cursor: pointer; transition: all 0.2s ease;
      color: var(--text-muted); font-size: 0.85rem; font-weight: 600;
      border: 1px solid transparent;
    }}
    .date-item:hover {{
      background: rgba(255,255,255,0.04); color: #fff;
      border-color: var(--border);
    }}
    .date-item.active {{
      background: rgba(139, 92, 246, 0.12); color: #fff;
      border-color: var(--border-focus);
    }}
    .date-item .play-icon {{
      font-size: 0.7rem; opacity: 0; transition: opacity 0.2s;
    }}
    .date-item:hover .play-icon, .date-item.active .play-icon {{
      opacity: 1;
    }}

    /* Content Area */
    .content-area {{
      display: flex; flex-direction: column; gap: 24px;
    }}
    .content-split {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }}
    @media (min-width: 960px) {{
      .content-split {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    /* Panel Cards */
    .panel-card {{
      background: var(--surface);
      backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 28px;
      box-shadow: var(--shadow-lg);
      min-height: 400px;
    }}
    .panel-header {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 20px; padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .panel-header h2 {{
      font-size: 1.15rem; font-weight: 700; color: #fff;
      display: flex; align-items: center; gap: 8px;
    }}
    .panel-date-label {{
      font-size: 0.85rem; color: var(--text-muted); font-weight: 600;
    }}

    /* Report Markdown */
    .report-markdown {{
      font-size: 0.95rem; line-height: 1.75; color: #d1d5db;
    }}
    .report-markdown h1 {{
      font-size: 1.5rem; font-weight: 900; color: #fff;
      margin-bottom: 20px; text-align: center;
      background: var(--accent-gradient);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .report-markdown h2 {{
      font-size: 1.15rem; font-weight: 700; color: #fff;
      margin: 28px 0 14px; padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 8px;
    }}
    .report-markdown h2::before {{
      content: ''; display: inline-block; width: 4px; height: 16px;
      background: var(--accent-gradient); border-radius: 99px;
    }}
    .report-markdown p {{ margin-bottom: 14px; }}
    .report-markdown ul {{ padding-left: 20px; margin-bottom: 18px; list-style: none; }}
    .report-markdown li {{ margin-bottom: 8px; position: relative; padding-left: 18px; }}
    .report-markdown li::before {{
      content: '\\2726'; position: absolute; left: 0;
      color: var(--accent-light); font-size: 0.8rem;
    }}
    .report-markdown strong {{ color: #fff; font-weight: 700; }}
    .report-markdown blockquote {{
      border-left: 4px solid var(--accent);
      padding: 16px 20px; margin: 20px 0;
      color: var(--text-muted); font-style: italic;
      background: rgba(139, 92, 246, 0.04);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }}

    /* Transcript Dialogue */
    .transcript-content {{
      font-size: 0.9rem; line-height: 1.7;
    }}
    .dialogue-line {{
      display: flex; gap: 12px; margin-bottom: 16px;
      padding: 12px 16px; border-radius: var(--radius-sm);
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      transition: background 0.2s;
    }}
    .dialogue-line:hover {{
      background: rgba(255,255,255,0.04);
    }}
    .dialogue-line.host-a {{
      border-left: 3px solid var(--host-a);
    }}
    .dialogue-line.host-b {{
      border-left: 3px solid var(--host-b);
    }}
    .speaker-label {{
      font-weight: 700; font-size: 0.8rem; white-space: nowrap;
      padding: 2px 8px; border-radius: 99px; flex-shrink: 0;
      margin-top: 2px;
    }}
    .speaker-label.host-a {{
      color: var(--host-a); background: rgba(6, 182, 212, 0.1);
    }}
    .speaker-label.host-b {{
      color: var(--host-b); background: rgba(244, 114, 182, 0.1);
    }}
    .dialogue-text {{
      color: #d1d5db; flex: 1;
    }}

    .loading-placeholder {{
      text-align: center; color: var(--text-muted); padding: 60px 0;
      font-size: 0.9rem;
    }}

    /* Player Bar */
    .player-bar {{
      position: fixed; bottom: -150px; left: 50%; transform: translateX(-50%);
      width: calc(100% - 48px); max-width: 960px;
      background: rgba(12, 11, 20, 0.85);
      backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
      border: 1px solid var(--border-focus);
      border-radius: var(--radius-lg); padding: 16px 24px;
      z-index: 1000;
      box-shadow: 0 20px 50px rgba(0,0,0,0.6), 0 0 30px rgba(139,92,246,0.15);
      transition: all 0.4s cubic-bezier(0.16,1,0.3,1);
    }}
    .player-bar.active {{ bottom: 24px; }}
    .player-container {{
      display: grid; grid-template-columns: 1fr; align-items: center; gap: 16px;
    }}
    @media (min-width: 768px) {{
      .player-container {{ grid-template-columns: 240px 1fr 240px; }}
    }}
    .player-track-info {{ display: flex; align-items: center; gap: 12px; overflow: hidden; }}
    .player-logo-mini {{
      width: 40px; height: 40px; border-radius: 8px;
      border: 1px solid var(--border); flex-shrink: 0;
      animation: spin 8s linear infinite; animation-play-state: paused;
    }}
    .player-logo-mini.animating {{ animation-play-state: running; }}
    @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
    .player-metadata {{ overflow: hidden; }}
    .player-track-title {{
      font-size: 0.85rem; font-weight: 700; color: #fff;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .player-track-subtitle {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }}
    .player-controls-main {{ display: flex; flex-direction: column; align-items: center; gap: 8px; width: 100%; }}
    .player-buttons {{ display: flex; align-items: center; gap: 18px; }}
    .player-btn {{
      background: transparent; border: none; color: var(--text-muted);
      cursor: pointer; font-size: 1.1rem; transition: color 0.2s;
    }}
    .player-btn:hover {{ color: #fff; }}
    .player-btn-play {{
      width: 36px; height: 36px; border-radius: 50%;
      background: #fff; color: #07060d; font-size: 1rem;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.25); transition: all 0.2s;
      padding-left: 2px; border: none; cursor: pointer;
    }}
    .player-btn-play:hover {{ transform: scale(1.06); background: var(--accent-light); color: #fff; }}
    .player-btn-play.playing {{ padding-left: 0; }}
    .player-timeline {{
      display: flex; align-items: center; gap: 12px; width: 100%;
      font-size: 0.75rem; color: var(--text-muted);
    }}
    .progress-input {{
      flex: 1; height: 4px; border-radius: 99px; background: var(--border);
      outline: none; cursor: pointer; appearance: none; -webkit-appearance: none;
    }}
    .progress-input::-webkit-slider-thumb {{
      appearance: none; -webkit-appearance: none;
      width: 12px; height: 12px; border-radius: 50%;
      background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    .player-utils {{ display: flex; align-items: center; justify-content: flex-end; gap: 16px; }}
    .speed-control {{
      background: rgba(255,255,255,0.04); border: 1px solid var(--border);
      border-radius: var(--radius-sm); overflow: hidden;
    }}
    .speed-btn {{
      background: transparent; border: none; color: var(--text-muted);
      font-size: 0.75rem; font-weight: 700; padding: 6px 10px;
      cursor: pointer; transition: all 0.2s;
    }}
    .speed-btn:hover {{ color: #fff; background: rgba(255,255,255,0.04); }}
    .volume-control {{ display: flex; align-items: center; gap: 8px; }}
    .volume-input {{
      width: 70px; height: 4px; border-radius: 99px; background: var(--border);
      outline: none; cursor: pointer; appearance: none; -webkit-appearance: none;
    }}
    .volume-input::-webkit-slider-thumb {{
      appearance: none; -webkit-appearance: none;
      width: 10px; height: 10px; border-radius: 50%; background: #fff;
    }}
    .visualizer {{ display: flex; align-items: flex-end; gap: 3px; height: 18px; width: 24px; }}
    .v-bar {{
      width: 3px; background: var(--success); height: 3px;
      border-radius: 99px; transition: height 0.1s ease;
      box-shadow: 0 0 6px var(--success);
    }}
    .visualizer.animating .v-bar:nth-child(1) {{ animation: bounce 0.8s ease-in-out infinite alternate; }}
    .visualizer.animating .v-bar:nth-child(2) {{ animation: bounce 0.5s ease-in-out infinite alternate; }}
    .visualizer.animating .v-bar:nth-child(3) {{ animation: bounce 0.9s ease-in-out infinite alternate; }}
    .visualizer.animating .v-bar:nth-child(4) {{ animation: bounce 0.6s ease-in-out infinite alternate; }}
    @keyframes bounce {{ 0% {{ height: 3px; }} 100% {{ height: 16px; }} }}

    @media (max-width: 768px) {{
      .player-bar {{
        left: 0; transform: none; width: 100%; max-width: 100%;
        border-radius: 0; border-left: none; border-right: none; border-bottom: none;
        padding: 14px 16px;
      }}
      .player-bar.active {{ bottom: 0; }}
      .player-utils {{ display: none; }}
      .player-container {{ grid-template-columns: 1fr 1fr; }}
      .player-controls-main {{ align-items: flex-end; }}
    }}

    /* Toast */
    .toast {{
      position: fixed; top: 24px; left: 50%; transform: translate(-50%, -100px);
      background: rgba(16,185,129,0.95); color: #fff;
      padding: 12px 24px; border-radius: 99px;
      font-weight: 700; font-size: 0.85rem; z-index: 2000;
      box-shadow: 0 10px 30px rgba(16,185,129,0.3);
      transition: transform 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
      pointer-events: none;
      backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255,255,255,0.1);
    }}
    .toast.show {{ transform: translate(-50%, 0); }}

    .footer {{
      margin-top: 100px; padding: 40px 0; text-align: center;
      color: var(--text-dark); font-size: 0.8rem;
      border-top: 1px solid var(--border);
    }}
    .footer a {{ color: var(--text-muted); text-decoration: none; font-weight: 600; }}
    .footer a:hover {{ text-decoration: underline; color: #fff; }}
  </style>
</head>
<body>
  <div class="ambient-glow-1"></div>
  <div class="ambient-glow-2"></div>

  <div class="container">
    <header>
      <a href="./" class="logo-block">
        <img src="./logo.png" alt="Logo" class="logo-img">
        <span class="brand-name">{podcast_title}</span>
      </a>
      <div class="nav-links">
        <a href="./feed.xml" target="_blank">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11a9 9 0 0 1 9 9"></path><path d="M4 4a16 16 0 0 1 16 16"></path><circle cx="5" cy="19" r="1"></circle></svg>
          RSS
        </a>
        <a href="https://github.com/nicekai-jpg/ai-news-podcast" target="_blank">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
          GitHub
        </a>
      </div>
    </header>

    <div class="layout-grid">
      <aside class="date-list">
        <h2>📅 日期</h2>
        <ul id="date-list-ul"></ul>
      </aside>

      <main class="content-area">
        <div class="content-split">
          <div class="panel-card">
            <div class="panel-header">
              <h2>📰 科技日报</h2>
              <span class="panel-date-label" id="report-date-label"></span>
            </div>
            <div class="report-markdown" id="report-content-box">
              <div class="loading-placeholder">选择日期查看日报</div>
            </div>
          </div>
          <div class="panel-card">
            <div class="panel-header">
              <h2>🎙️ 播客剧本</h2>
              <span class="panel-date-label" id="transcript-date-label"></span>
            </div>
            <div class="transcript-content" id="transcript-content-box">
              <div class="loading-placeholder">选择日期查看剧本</div>
            </div>
          </div>
        </div>
      </main>
    </div>

    <audio controls id="main-audio" style="display: none;"></audio>

    <div class="toast" id="toast">RSS 订阅链接已复制到剪贴板</div>

    <div class="footer">
      <p>{podcast_title} · {build_time}</p>
      <p>RSS 订阅: <a href="./feed.xml">{base_url}/feed.xml</a></p>
    </div>
  </div>

  <div class="player-bar" id="player-bar">
    <div class="player-container">
      <div class="player-track-info">
        <img src="./logo.png" alt="Mini Logo" class="player-logo-mini" id="player-mini-logo">
        <div class="player-metadata">
          <div class="player-track-title" id="bar-title">未在播放</div>
          <div class="player-track-subtitle">{podcast_title}</div>
        </div>
      </div>
      <div class="player-controls-main">
        <div class="player-buttons">
          <button class="player-btn" onclick="skip(-15)">⏪</button>
          <button class="player-btn player-btn-play" id="bar-play-btn" onclick="toggleBarPlay()">▶</button>
          <button class="player-btn" onclick="skip(15)">⏩</button>
        </div>
        <div class="player-timeline">
          <span id="time-elapsed">0:00</span>
          <input type="range" class="progress-input" id="progress-bar" min="0" max="100" value="0">
          <span id="time-duration">0:00</span>
        </div>
      </div>
      <div class="player-utils">
        <div class="speed-control">
          <button class="speed-btn" onclick="cycleSpeed()" id="speed-label">1.0x</button>
        </div>
        <div class="volume-control">
          <span style="font-size:0.95rem;cursor:pointer;" onclick="toggleMute()" id="volume-icon">🔊</span>
          <input type="range" class="volume-input" id="volume-bar" min="0" max="100" value="80">
        </div>
        <div class="visualizer" id="visualizer">
          <div class="v-bar"></div><div class="v-bar"></div><div class="v-bar"></div><div class="v-bar"></div>
        </div>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const dates = {dates_json};
    const episodesMap = {episodes_map_json};

    const audio = document.getElementById('main-audio');
    let currentDate = null;
    let currentPlayingId = null;
    let isPlaying = false;
    let lastVolume = 0.8;
    const speeds = [1.0, 1.25, 1.5, 1.75, 2.0];
    let currentSpeedIdx = 0;
    if (audio) audio.volume = lastVolume;

    // ── Date List ──
    function initDateList() {{
      const ul = document.getElementById('date-list-ul');
      if (!ul) return;
      ul.innerHTML = '';
      dates.forEach(dateStr => {{
        const li = document.createElement('li');
        li.className = 'date-item';
        li.setAttribute('data-date', dateStr);
        li.onclick = () => loadDate(dateStr);

        let display = dateStr;
        try {{
          const p = dateStr.split('-');
          if (p.length === 3) display = p[1] + '月' + p[2] + '日';
        }} catch(e) {{}}

        li.innerHTML = '<span class="play-icon">▶</span> ' + display;
        ul.appendChild(li);
      }});
      if (dates.length > 0) loadDate(dates[0]);
    }}

    function setActiveDate(dateStr) {{
      currentDate = dateStr;
      document.querySelectorAll('.date-item').forEach(li => {{
        li.classList.toggle('active', li.getAttribute('data-date') === dateStr);
      }});
    }}

    // ── Load Date Content ──
    async function loadDate(dateStr) {{
      setActiveDate(dateStr);

      let displayFull = dateStr;
      try {{
        const p = dateStr.split('-');
        if (p.length === 3) displayFull = p[0] + '年' + p[1] + '月' + p[2] + '日';
      }} catch(e) {{}}

      const reportLabel = document.getElementById('report-date-label');
      const transcriptLabel = document.getElementById('transcript-date-label');
      if (reportLabel) reportLabel.textContent = displayFull;
      if (transcriptLabel) transcriptLabel.textContent = displayFull;

      // Load report
      const reportBox = document.getElementById('report-content-box');
      if (reportBox) {{
        reportBox.innerHTML = '<div class="loading-placeholder">正在加载日报…</div>';
        try {{
          const resp = await fetch('./reports/daily_report_' + dateStr + '.md');
          if (!resp.ok) throw new Error('Not found');
          const md = await resp.text();
          reportBox.innerHTML = marked.parse(md);
        }} catch(e) {{
          reportBox.innerHTML = '<div class="loading-placeholder">该日期暂无日报</div>';
        }}
      }}

      // Load transcript
      const transcriptBox = document.getElementById('transcript-content-box');
      if (transcriptBox) {{
        transcriptBox.innerHTML = '<div class="loading-placeholder">正在加载剧本…</div>';
        try {{
          const resp = await fetch('./episodes/' + dateStr + '.txt');
          if (!resp.ok) throw new Error('Not found');
          const text = await resp.text();
          transcriptBox.innerHTML = parseTranscript(text);
        }} catch(e) {{
          transcriptBox.innerHTML = '<div class="loading-placeholder">该日期暂无剧本</div>';
        }}
      }}

      // Auto-play
      const ep = episodesMap[dateStr];
      if (ep && ep.mp3) {{
        togglePlay(dateStr, ep.mp3, ep.title || ('AI 新闻快报 | ' + dateStr));
      }}
    }}

    // ── Parse SSML Transcript ──
    function parseTranscript(text) {{
      if (!text || !text.trim()) return '<div class="loading-placeholder">剧本内容为空</div>';

      // Try SSML parsing
      if (text.includes('<voice') || text.includes('<speak')) {{
        const lines = [];
        const voiceRegex = /<voice\\s+name="([^"]+)">([\\s\\S]*?)<\\/voice>/g;
        let match;
        while ((match = voiceRegex.exec(text)) !== null) {{
          const voiceName = match[1];
          const content = match[2].trim();
          if (!content) continue;
          const isXiaoxiao = voiceName.includes('Xiaoxiao');
          const hostClass = isXiaoxiao ? 'host-b' : 'host-a';
          const speakerName = isXiaoxiao ? '晓晓' : '博文';
          lines.push(
            '<div class="dialogue-line ' + hostClass + '">' +
            '<span class="speaker-label ' + hostClass + '">' + speakerName + '</span>' +
            '<span class="dialogue-text">' + escapeHtml(content) + '</span>' +
            '</div>'
          );
        }}
        if (lines.length > 0) return lines.join('');
      }}

      // Fallback: [Host A] / [Host B] format
      const hostRegex = /\\[Host\\s*([AB])\\]\\s*([^\\[]*)/g;
      let fallbackMatch;
      const fallbackLines = [];
      while ((fallbackMatch = hostRegex.exec(text)) !== null) {{
        const host = fallbackMatch[1];
        const content = fallbackMatch[2].trim();
        if (!content) continue;
        const hostClass = host === 'B' ? 'host-b' : 'host-a';
        const speakerName = host === 'B' ? '晓晓' : '博文';
        fallbackLines.push(
          '<div class="dialogue-line ' + hostClass + '">' +
          '<span class="speaker-label ' + hostClass + '">' + speakerName + '</span>' +
          '<span class="dialogue-text">' + escapeHtml(content) + '</span>' +
          '</div>'
        );
      }}
      if (fallbackLines.length > 0) return fallbackLines.join('');

      // Plain text fallback
      return '<div class="dialogue-line host-a"><span class="dialogue-text">' + escapeHtml(text) + '</span></div>';
    }}

    function escapeHtml(str) {{
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }}

    // ── Audio Player ──
    function togglePlay(episodeId, mp3Url, titleText) {{
      const playerBar = document.getElementById('player-bar');
      const barTitle = document.getElementById('bar-title');
      if (currentPlayingId === episodeId) {{
        if (audio.paused) audio.play(); else audio.pause();
      }} else {{
        currentPlayingId = episodeId;
        audio.src = mp3Url;
        audio.play();
        if (barTitle) barTitle.textContent = titleText;
        if (playerBar) playerBar.classList.add('active');
        currentSpeedIdx = 0;
        audio.playbackRate = 1.0;
        const speedLabel = document.getElementById('speed-label');
        if (speedLabel) speedLabel.textContent = '1.0x';
      }}
    }}

    function toggleBarPlay() {{
      if (!currentPlayingId) return;
      if (audio.paused) audio.play(); else audio.pause();
    }}

    function updatePlayState(playing) {{
      isPlaying = playing;
      const barPlayBtn = document.getElementById('bar-play-btn');
      const visualizer = document.getElementById('visualizer');
      const miniLogo = document.getElementById('player-mini-logo');
      if (barPlayBtn) {{
        barPlayBtn.textContent = playing ? '⏸' : '▶';
        if (playing) barPlayBtn.classList.add('playing'); else barPlayBtn.classList.remove('playing');
      }}
      if (playing) {{
        if (visualizer) visualizer.classList.add('animating');
        if (miniLogo) miniLogo.classList.add('animating');
      }} else {{
        if (visualizer) visualizer.classList.remove('animating');
        if (miniLogo) miniLogo.classList.remove('animating');
      }}
      document.querySelectorAll('.date-item').forEach(li => {{
        const icon = li.querySelector('.play-icon');
        if (icon) {{
          if (li.getAttribute('data-date') === currentPlayingId) {{
            icon.textContent = playing ? '⏸' : '▶';
          }} else {{
            icon.textContent = '▶';
          }}
        }}
      }});
    }}

    if (audio) {{
      audio.addEventListener('play', () => updatePlayState(true));
      audio.addEventListener('pause', () => updatePlayState(false));
      audio.addEventListener('timeupdate', () => {{
        const progressBar = document.getElementById('progress-bar');
        const timeElapsed = document.getElementById('time-elapsed');
        const timeDuration = document.getElementById('time-duration');
        if (!isNaN(audio.duration)) {{
          const pct = (audio.currentTime / audio.duration) * 100;
          progressBar.value = pct;
          progressBar.style.background = 'linear-gradient(to right, var(--accent) 0%, var(--accent) ' + pct + '%, var(--border) ' + pct + '%, var(--border) 100%)';
          timeElapsed.textContent = formatTime(audio.currentTime);
          timeDuration.textContent = formatTime(audio.duration);
        }}
      }});
    }}

    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {{
      progressBar.addEventListener('input', e => {{
        if (audio && audio.duration) audio.currentTime = (e.target.value / 100) * audio.duration;
      }});
    }}

    function skip(secs) {{
      if (audio) audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + secs));
    }}

    const volumeBar = document.getElementById('volume-bar');
    if (volumeBar) {{
      volumeBar.style.background = 'linear-gradient(to right, var(--accent) 0%, var(--accent) ' + volumeBar.value + '%, var(--border) ' + volumeBar.value + '%, var(--border) 100%)';
      volumeBar.addEventListener('input', e => {{
        if (audio) {{
          audio.volume = e.target.value / 100;
          lastVolume = audio.volume;
          volumeBar.style.background = 'linear-gradient(to right, var(--accent) 0%, var(--accent) ' + e.target.value + '%, var(--border) ' + e.target.value + '%, var(--border) 100%)';
          updateVolumeIcon(audio.volume);
        }}
      }});
    }}

    function toggleMute() {{
      const volIcon = document.getElementById('volume-icon');
      if (audio) {{
        if (audio.volume > 0) {{
          lastVolume = audio.volume; audio.volume = 0;
          volumeBar.value = 0;
          volumeBar.style.background = 'linear-gradient(to right, var(--accent) 0%, var(--accent) 0%, var(--border) 0%, var(--border) 100%)';
          if (volIcon) volIcon.textContent = '🔇';
        }} else {{
          audio.volume = lastVolume; volumeBar.value = lastVolume * 100;
          volumeBar.style.background = 'linear-gradient(to right, var(--accent) 0%, var(--accent) ' + (lastVolume*100) + '%, var(--border) ' + (lastVolume*100) + '%, var(--border) 100%)';
          updateVolumeIcon(lastVolume);
        }}
      }}
    }}

    function updateVolumeIcon(vol) {{
      const volIcon = document.getElementById('volume-icon');
      if (!volIcon) return;
      if (vol === 0) volIcon.textContent = '🔇';
      else if (vol < 0.4) volIcon.textContent = '🔈';
      else volIcon.textContent = '🔊';
    }}

    function cycleSpeed() {{
      if (!audio) return;
      currentSpeedIdx = (currentSpeedIdx + 1) % speeds.length;
      audio.playbackRate = speeds[currentSpeedIdx];
      document.getElementById('speed-label').textContent = speeds[currentSpeedIdx].toFixed(2).replace('.00','') + 'x';
    }}

    function formatTime(secs) {{
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return m + ':' + (s < 10 ? '0' : '') + s;
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      initDateList();
    }});
  </script>
</body>
</html>"""

    html = (
        html_template.replace("{podcast_title}", str(podcast_title))
        .replace("{base_url}", str(base_url))
        .replace("{dates_json}", str(dates_json))
        .replace("{episodes_map_json}", str(episodes_map_json))
        .replace("{build_time}", str(build_time))
    )

    _write_text(site_dir / "index.html", html)
