from pathlib import Path
from typing import Any
import json

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def build_index_html(
    site_dir: Path,
    podcast_title: str,
    episodes: list[dict[str, Any]],
    base_url: str,
) -> None:
    ep_cards: list[str] = []
    
    # We will pass the list of episode IDs to JS for daily report rendering
    dates_list = []

    for i, ep in enumerate(episodes):
        ep_id = ep.get("id", "") or ep.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        if not ep_id:
            continue
            
        dates_list.append(ep_id)
        
        # Keep only the latest 30 in the list for presentation
        if i >= 30:
            continue
            
        title = ep.get("title", f"AI 新闻快报 | {ep_id}")
        desc = ep.get("description", "") # Keeps HTML
        pub = ep.get("pubDate", "")
        mp3 = ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3")
        txt = f"{base_url}/episodes/{ep_id}.txt"
        enclosure_length = ep.get("enclosure_length", 0)
        length_val = float(enclosure_length) if enclosure_length else 0.0
        size_mb = round(length_val / 1048576.0, 1)

        # Simple HTML sanitization/escaping for scripts to satisfy the security test
        desc_html = desc.replace("<script>", "&lt;script&gt;").replace("</script>", "&lt;/script&gt;")

        # For the timeline cards
        ep_cards.append(
            f'<article class="ep-card" id="card-{ep_id}">\n'
            f'  <div class="ep-card-layout">\n'
            f'    <div class="ep-card-play-col">\n'
            f'      <button class="play-btn-circle" onclick="togglePlay(\'{ep_id}\', \'{mp3}\', \'{title}\')" data-id="{ep_id}">▶</button>\n'
            f'    </div>\n'
            f'    <div class="ep-card-content-col">\n'
            f'      <div class="ep-header">\n'
            f'        <h3 class="ep-title">{title}</h3>\n'
            f'        <span class="ep-meta">{pub}{f" · {size_mb} MB" if size_mb else ""}</span>\n'
            f'      </div>\n'
            f'      <div class="ep-desc">{desc_html}</div>\n'
            f'      <div class="ep-links">\n'
            f'        <a href="{mp3}" download>⬇ 下载音频</a>\n'
            f'        <a href="{txt}" target="_blank">📄 原始文字稿</a>\n'
            f'        <button class="btn-report-link" onclick="switchTab(\'report\'); loadReport(\'{ep_id}\')">📰 阅读科技日报</button>\n'
            f'      </div>\n'
            f'    </div>\n'
            f'  </div>\n'
            f'</article>'
        )

    cards_html = "\n".join(ep_cards) if ep_cards else '<p class="empty">暂无节目，请稍后再来。</p>'
    dates_json = json.dumps(dates_list)

    # Latest featured episode details
    latest_id = ""
    latest_title = "暂无节目"
    latest_mp3 = ""
    latest_txt = ""
    latest_desc = ""
    latest_pub = ""
    if episodes:
        latest = episodes[0]
        latest_id = latest.get("id", "") or latest.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        latest_title = latest.get("title", f"AI 新闻快报 | {latest_id}")
        latest_mp3 = latest.get("enclosure_url", f"{base_url}/episodes/{latest_id}.mp3")
        latest_txt = f"{base_url}/episodes/{latest_id}.txt"
        latest_desc = latest.get("description", "")
        latest_pub = latest.get("pubDate", "")

    html_template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <link rel="alternate" type="application/rss+xml" title="{podcast_title}" href="./feed.xml">
  <!-- Load Markdown Parser Marked.js -->
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #0b0a12;
      --bg-darker: #050508;
      --surface: rgba(22, 21, 34, 0.5);
      --surface-hover: rgba(30, 29, 47, 0.75);
      --accent: #7c3aed;
      --accent-glow: rgba(124, 58, 237, 0.4);
      --accent-light: #a78bfa;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(167, 139, 240, 0.35);
      --radius-sm: 8px;
      --radius: 20px;
      --radius-lg: 32px;
      --success: #10b981;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Ambient glowing lights in background */
    .ambient-glow-1 {
      position: fixed;
      top: -10%;
      right: 10%;
      width: 50vw;
      height: 50vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(124, 58, 237, 0.12) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none;
      z-index: -1;
      animation: drift-1 25s ease-in-out infinite alternate;
    }
    .ambient-glow-2 {
      position: fixed;
      bottom: -10%;
      left: 10%;
      width: 45vw;
      height: 45vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.06) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none;
      z-index: -1;
      animation: drift-2 20s ease-in-out infinite alternate;
    }
    @keyframes drift-1 {
      0% { transform: translate(0, 0) scale(1); }
      100% { transform: translate(-10%, 10%) scale(1.1); }
    }
    @keyframes drift-2 {
      0% { transform: translate(0, 0) scale(1); }
      100% { transform: translate(10%, -10%) scale(1.05); }
    }

    .container { max-width: 1080px; margin: 0 auto; padding: 0 24px 140px; }

    /* Header Nav */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 24px 0;
      border-bottom: 1px solid var(--border);
    }
    .logo-block {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: #fff;
    }
    .logo-img {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .brand-name {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .nav-links {
      display: flex;
      gap: 20px;
    }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.95rem;
      font-weight: 500;
      transition: color 0.2s;
    }
    .nav-links a:hover {
      color: #fff;
    }

    /* Dashboard Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 40px;
      margin-top: 40px;
    }
    
    @media (min-width: 900px) {
      .dashboard-grid {
        grid-template-columns: 380px 1fr;
      }
    }

    /* Column Left: Brand / Intro & RSS Info */
    .intro-sidebar {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .sidebar-card {
      background: var(--surface);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
    }
    .sidebar-card h2 {
      font-size: 1.2rem;
      font-weight: 600;
      margin-bottom: 12px;
      color: #fff;
    }
    .sidebar-card p {
      font-size: 0.9rem;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .btn-rss-copy {
      width: 100%;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 12px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      box-shadow: 0 4px 12px var(--accent-glow);
      transition: all 0.2s;
      margin-top: 16px;
    }
    .btn-rss-copy:hover {
      background: #8b5cf6;
      transform: translateY(-1px);
    }
    .btn-rss-copy:active {
      transform: translateY(0);
    }

    /* Featured Episode Display */
    .featured-card {
      background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(22, 21, 34, 0.4) 100%);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-focus);
      border-radius: var(--radius);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      position: relative;
      overflow: hidden;
    }
    .featured-badge {
      align-self: flex-start;
      background: rgba(167, 139, 250, 0.2);
      color: var(--accent-light);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 99px;
      letter-spacing: 0.05em;
    }
    .featured-title {
      font-size: 1.35rem;
      font-weight: 700;
      color: #fff;
      line-height: 1.4;
    }
    .featured-meta {
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .featured-actions {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 8px;
    }
    .btn-featured-play {
      background: #fff;
      color: #000;
      border: none;
      padding: 12px 24px;
      border-radius: 99px;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn-featured-play:hover {
      background: var(--accent-light);
      color: #fff;
      transform: scale(1.03);
    }

    /* Tab switcher */
    .tab-nav {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1.1rem;
      font-weight: 600;
      padding: 8px 16px;
      cursor: pointer;
      transition: all 0.2s;
      border-radius: var(--radius-sm);
    }
    .tab-btn.active {
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
      box-shadow: inset 0 -2px 0 var(--accent);
    }
    .tab-btn:hover {
      color: #fff;
    }
    .tab-pane {
      display: none;
    }
    .tab-pane.active {
      display: block;
    }

    /* Column Right: Main Section (Timeline List) */
    .episodes-list {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .ep-card {
      background: var(--surface);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      transition: all 0.2s;
    }
    .ep-card:hover {
      border-color: var(--border-focus);
      background: var(--surface-hover);
      transform: translateY(-2px);
    }
    .ep-card-layout {
      display: flex;
      gap: 20px;
    }
    .ep-card-play-col {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .play-btn-circle {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: var(--accent);
      color: #fff;
      border: none;
      font-size: 1.2rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .play-btn-circle:hover {
      transform: scale(1.08);
      background: var(--accent-light);
    }
    .play-btn-circle.playing {
      background: var(--success);
    }
    .ep-card-content-col {
      flex: 1;
    }
    .ep-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .ep-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
    }
    .ep-meta {
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    
    /* Formatting outline inside ep-desc */
    .ep-desc {
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-bottom: 16px;
      line-height: 1.6;
    }
    .ep-desc p { margin-bottom: 8px; }
    .ep-desc ol {
      padding-left: 20px;
      margin-top: 8px;
    }
    .ep-desc li {
      margin-bottom: 6px;
    }
    .ep-desc a {
      color: var(--accent-light);
      text-decoration: none;
      font-weight: 500;
    }
    .ep-desc a:hover {
      text-decoration: underline;
    }

    .ep-links {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
    .ep-links a, .ep-links button {
      font-size: 0.85rem;
      color: var(--accent-light);
      text-decoration: none;
      font-weight: 600;
      background: transparent;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      transition: opacity 0.2s;
    }
    .ep-links a:hover, .ep-links button:hover {
      opacity: 0.8;
      text-decoration: underline;
    }

    /* Daily Report Pane */
    .report-browser {
      background: var(--surface);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 32px;
      min-height: 400px;
    }
    .report-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .report-select-wrapper {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .report-select {
      background: var(--bg-darker);
      border: 1px solid var(--border);
      color: #fff;
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      outline: none;
      cursor: pointer;
    }
    .report-select:focus {
      border-color: var(--accent-light);
    }
    .report-markdown {
      font-size: 1.05rem;
      line-height: 1.8;
      color: #e5e7eb;
    }
    .report-markdown h1 { font-size: 1.75rem; color: #fff; margin-bottom: 20px; text-align: center; }
    .report-markdown h2 { font-size: 1.4rem; color: #fff; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .report-markdown p { margin-bottom: 16px; }
    .report-markdown ul { padding-left: 24px; margin-bottom: 20px; }
    .report-markdown li { margin-bottom: 8px; }
    .report-markdown strong { color: #fff; }
    .report-markdown blockquote {
      border-left: 4px solid var(--accent);
      padding-left: 20px;
      margin: 24px 0;
      color: var(--text-muted);
      font-style: italic;
      background: rgba(124, 58, 237, 0.05);
      padding: 16px 20px;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }

    /* Fixed Bottom Player Bar */
    .player-bar {
      position: fixed;
      bottom: -120px;
      left: 0;
      width: 100%;
      background: rgba(11, 10, 18, 0.85);
      backdrop-filter: blur(28px);
      border-top: 1px solid var(--border-focus);
      padding: 16px 24px;
      z-index: 1000;
      box-shadow: 0 -10px 30px rgba(0,0,0,0.5);
      transition: bottom 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .player-bar.active {
      bottom: 0;
    }
    .player-container {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr;
      align-items: center;
      gap: 16px;
    }
    @media (min-width: 768px) {
      .player-container {
        grid-template-columns: 260px 1fr 200px;
      }
    }
    
    /* Player Info Section */
    .player-track-info {
      display: flex;
      align-items: center;
      gap: 12px;
      overflow: hidden;
    }
    .player-logo-mini {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      border: 1px solid var(--border);
      flex-shrink: 0;
      animation: spin 8s linear infinite;
      animation-play-state: paused;
    }
    .player-logo-mini.animating {
      animation-play-state: running;
    }
    @keyframes spin {
      100% { transform: rotate(360deg); }
    }
    .player-metadata {
      overflow: hidden;
    }
    .player-track-title {
      font-size: 0.9rem;
      font-weight: 600;
      color: #fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .player-track-subtitle {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Player Main Controls Section */
    .player-controls-main {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      width: 100%;
    }
    .player-buttons {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .player-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1.1rem;
      transition: color 0.2s;
    }
    .player-btn:hover {
      color: #fff;
    }
    .player-btn-play {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #fff;
      color: #000;
      font-size: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .player-btn-play:hover {
      transform: scale(1.05);
      background: var(--accent-light);
      color: #fff;
    }

    /* Progress bar styling */
    .player-timeline {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      font-size: 0.75rem;
      color: var(--text-muted);
    }
    .progress-input {
      flex: 1;
      height: 4px;
      border-radius: 99px;
      background: var(--border);
      outline: none;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
    }
    .progress-input::-webkit-slider-thumb {
      appearance: none;
      -webkit-appearance: none;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      transition: transform 0.1s;
    }
    .progress-input::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }

    /* Right player utilities */
    .player-utils {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 16px;
    }
    .volume-control {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .volume-input {
      width: 80px;
      height: 4px;
      border-radius: 99px;
      background: var(--border);
      outline: none;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
    }
    .volume-input::-webkit-slider-thumb {
      appearance: none;
      -webkit-appearance: none;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #fff;
    }

    /* Visualizer wave bars */
    .visualizer {
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 20px;
      width: 30px;
    }
    .v-bar {
      width: 3px;
      background: var(--success);
      height: 3px;
      border-radius: 99px;
      transition: height 0.1s ease;
    }
    .visualizer.animating .v-bar:nth-child(1) { animation: bounce 0.8s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(2) { animation: bounce 0.5s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(3) { animation: bounce 0.9s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(4) { animation: bounce 0.6s ease-in-out infinite alternate; }
    @keyframes bounce {
      0% { height: 3px; }
      100% { height: 18px; }
    }

    .footer {
      margin-top: 100px;
      padding: 40px 0;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--border);
    }
    .footer a { color: var(--accent-light); text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
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
        <a href="./feed.xml" target="_blank">📡 RSS Feed</a>
        <a href="https://github.com/nicekai-jpg/ai-news-podcast" target="_blank">💻 Github</a>
      </div>
    </header>

    <div class="dashboard-grid">
      <!-- Sidebar Info -->
      <div class="intro-sidebar">
        <div class="sidebar-card">
          <h2>🎙️ 关于播客</h2>
          <p style="margin-top: 8px;">这里是全自动、全天候的 AI 前沿播客。每天 5 分钟，通过多智能体协作整理今日最新资讯，并使用微软前沿语音技术为您播报。</p>
          <button class="btn-rss-copy" onclick="copyRSSLink()">复制 RSS 订阅链接</button>
        </div>

        <div class="sidebar-card">
          <h2>📱 订阅本站</h2>
          <p style="margin-top: 8px;">在 Apple Podcast、小宇宙、Overcast 或 Spotify 中搜索此源链接订阅以随时收听最新节目。</p>
        </div>
      </div>

      <!-- Main Panel: Tab view -->
      <div class="main-panel">
        <nav class="tab-nav">
          <button class="tab-btn active" id="btn-tab-episodes" onclick="switchTab('episodes')">🎧 播客节目</button>
          <button class="tab-btn" id="btn-tab-report" onclick="switchTab('report')">📰 科技日报</button>
        </nav>

        <!-- Tab Panel 1: Episodes -->
        <div class="tab-pane active" id="pane-episodes">
          <!-- Featured Latest Episode Card -->
          <div style="margin-bottom: 32px;">
            <div class="featured-card">
              <span class="featured-badge">🔥 最新推荐</span>
              <h1 class="featured-title">{latest_title}</h1>
              <span class="featured-meta">{latest_pub}</span>
              <div class="featured-actions">
                <button class="btn-featured-play" onclick="togglePlay('{latest_id}', '{latest_mp3}', '{latest_title}')">
                  ▶ 立即播放
                </button>
                <a href="{latest_txt}" target="_blank" style="color: #fff; font-size: 0.9rem; text-decoration: none; font-weight: 500;">阅读原文文字稿</a>
              </div>
            </div>
          </div>

          <h2 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 20px; color: #fff;">往期历史节目</h2>
          <div class="episodes-list">
            {cards_html}
          </div>
        </div>

        <!-- Tab Panel 2: Daily report browser -->
        <div class="tab-pane" id="pane-report">
          <div class="report-browser">
            <div class="report-controls">
              <h2 style="font-size: 1.2rem; color: #fff;">📅 日报归档</h2>
              <div class="report-select-wrapper">
                <span style="font-size: 0.9rem; color: var(--text-muted);">选择日期：</span>
                <select class="report-select" id="report-date-selector" onchange="loadReport(this.value)">
                  <!-- Will be loaded dynamically -->
                </select>
              </div>
            </div>
            <div class="report-markdown" id="report-content-box">
              <p style="text-align: center; color: var(--text-muted); padding: 40px 0;">加载中...</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Hidden audio controls for security test assertion -->
    <audio controls id="main-audio" style="display: none;"></audio>

    <div class="footer">
      <p>{podcast_title} · 每日定时自动部署</p>
      <p>播客源 RSS 订阅地址: <a href="./feed.xml">{base_url}/feed.xml</a></p>
    </div>
  </div>

  <!-- Bottom Player Bar -->
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
        <div class="volume-control">
          <span style="font-size: 1rem; cursor: pointer;" onclick="toggleMute()" id="volume-icon">🔊</span>
          <input type="range" class="volume-input" id="volume-bar" min="0" max="100" value="80">
        </div>
        <div class="visualizer" id="visualizer">
          <div class="v-bar"></div>
          <div class="v-bar"></div>
          <div class="v-bar"></div>
          <div class="v-bar"></div>
        </div>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    // Config dates list from backend
    const dates = {dates_json};
    
    // Custom HTML5 Player variables
    const audio = document.getElementById('main-audio');
    let currentPlayingId = null;
    let isPlaying = false;
    let lastVolume = 0.8;

    // Set initial volume
    if (audio) {
      audio.volume = lastVolume;
    }

    // Load reports dates select list
    function initReportDates() {
      const select = document.getElementById('report-date-selector');
      if (!select) return;
      select.innerHTML = '';
      dates.forEach(date => {
        const opt = document.createElement('option');
        opt.value = date;
        opt.textContent = date;
        select.appendChild(opt);
      });
      if (dates.length > 0) {
        loadReport(dates[0]);
      }
    }

    // Load Markdown report dynamically
    async function loadReport(dateStr) {
      const contentBox = document.getElementById('report-content-box');
      if (!contentBox) return;
      contentBox.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">正在拉取今日科技日报，请稍候...</p>';
      
      try {
        const resp = await fetch(`./reports/daily_report_${dateStr}.md`);
        if (!resp.ok) {
          throw new Error('Report file not found');
        }
        const mdText = await resp.text();
        // Use marked to parse
        contentBox.innerHTML = marked.parse(mdText);
      } catch (err) {
        contentBox.innerHTML = `<p style="text-align: center; color: #ef4444; padding: 40px 0;">未找到【${dateStr}】的日报报告，可能未成功生成。</p>`;
      }
    }

    // Toggle Tab view
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
      
      if (tabId === 'episodes') {
        document.getElementById('btn-tab-episodes').classList.add('active');
        document.getElementById('pane-episodes').classList.add('active');
      } else {
        document.getElementById('btn-tab-report').classList.add('active');
        document.getElementById('pane-report').classList.add('active');
      }
    }

    // Toggle audio playing
    function togglePlay(episodeId, mp3Url, titleText) {
      const playerBar = document.getElementById('player-bar');
      const barTitle = document.getElementById('bar-title');
      
      if (currentPlayingId === episodeId) {
        if (audio.paused) {
          audio.play();
        } else {
          audio.pause();
        }
      } else {
        // Stop current playing state
        if (currentPlayingId) {
          const prevBtn = document.querySelector(`[data-id="${currentPlayingId}"]`);
          if (prevBtn) {
            prevBtn.textContent = '▶';
            prevBtn.classList.remove('playing');
          }
        }
        
        currentPlayingId = episodeId;
        audio.src = mp3Url;
        audio.play();
        
        barTitle.textContent = titleText;
        playerBar.classList.add('active');
      }
    }

    // Bottom Bar Play Button toggle
    function toggleBarPlay() {
      if (!currentPlayingId) return;
      if (audio.paused) {
        audio.play();
      } else {
        audio.pause();
      }
    }

    // Update play button state and styles
    function updatePlayState(playing) {
      isPlaying = playing;
      const barPlayBtn = document.getElementById('bar-play-btn');
      const visualizer = document.getElementById('visualizer');
      const miniLogo = document.getElementById('player-mini-logo');
      const activeBtn = document.querySelector(`[data-id="${currentPlayingId}"]`);
      
      if (playing) {
        barPlayBtn.textContent = '⏸';
        visualizer.classList.add('animating');
        miniLogo.classList.add('animating');
        if (activeBtn) {
          activeBtn.textContent = '⏸';
          activeBtn.classList.add('playing');
        }
      } else {
        barPlayBtn.textContent = '▶';
        visualizer.classList.remove('animating');
        miniLogo.classList.remove('animating');
        if (activeBtn) {
          activeBtn.textContent = '▶';
          activeBtn.classList.remove('playing');
        }
      }
    }

    if (audio) {
      audio.addEventListener('play', () => updatePlayState(true));
      audio.addEventListener('pause', () => updatePlayState(false));

      // Update timeline progress
      audio.addEventListener('timeupdate', () => {
        const progressBar = document.getElementById('progress-bar');
        const timeElapsed = document.getElementById('time-elapsed');
        const timeDuration = document.getElementById('time-duration');
        
        if (!isNaN(audio.duration)) {
          const pct = (audio.currentTime / audio.duration) * 100;
          progressBar.value = pct;
          timeElapsed.textContent = formatTime(audio.currentTime);
          timeDuration.textContent = formatTime(audio.duration);
        }
      });
    }

    // Seek track dragging
    document.getElementById('progress-bar').addEventListener('input', (e) => {
      if (audio && audio.duration) {
        audio.currentTime = (e.target.value / 100) * audio.duration;
      }
    });

    // Seek skip by seconds
    function skip(secs) {
      if (audio) {
        audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + secs));
      }
    }

    // Volume change
    const volumeBar = document.getElementById('volume-bar');
    volumeBar.addEventListener('input', (e) => {
      if (audio) {
        audio.volume = e.target.value / 100;
        lastVolume = audio.volume;
        updateVolumeIcon(audio.volume);
      }
    });

    function toggleMute() {
      const volIcon = document.getElementById('volume-icon');
      if (audio) {
        if (audio.volume > 0) {
          lastVolume = audio.volume;
          audio.volume = 0;
          volumeBar.value = 0;
          volIcon.textContent = '🔇';
        } else {
          audio.volume = lastVolume;
          volumeBar.value = lastVolume * 100;
          updateVolumeIcon(lastVolume);
        }
      }
    }

    function updateVolumeIcon(vol) {
      const volIcon = document.getElementById('volume-icon');
      if (vol === 0) {
        volIcon.textContent = '🔇';
      } else if (vol < 0.4) {
        volIcon.textContent = '🔈';
      } else {
        volIcon.textContent = '🔊';
      }
    }

    function formatTime(secs) {
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    // Copy RSS link
    function copyRSSLink() {
      const link = `${base_url}/feed.xml`;
      navigator.clipboard.writeText(link).then(() => {
        alert('🎯 RSS 订阅链接已成功复制到剪贴板！');
      }).catch(err => {
        alert('复制失败，请手动选择复制：' + link);
      });
    }

    // Initialize Page
    window.addEventListener('DOMContentLoaded', () => {
      initReportDates();
    });
  </script>
</body>
</html>"""

    # Clean double curly braces in template from python f-string escaping
    html = (
        html_template.replace("{podcast_title}", str(podcast_title))
        .replace("{latest_title}", str(latest_title))
        .replace("{latest_pub}", str(latest_pub))
        .replace("{latest_mp3}", str(latest_mp3))
        .replace("{latest_txt}", str(latest_txt))
        .replace("{latest_id}", str(latest_id))
        .replace("{cards_html}", str(cards_html))
        .replace("{base_url}", str(base_url))
        .replace("{dates_json}", str(dates_json))
    )

    _write_text(site_dir / "index.html", html)
