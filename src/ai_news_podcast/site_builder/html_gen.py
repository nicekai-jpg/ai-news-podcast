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
    ep_cards: list[str] = []

    # We will pass the list of episode IDs to JS for daily report rendering
    dates_list = []

    for i, ep in enumerate(episodes):
        ep_id = ep.get("id", "") or ep.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        if not ep_id:
            continue

        dates_list.append(ep_id)

        # Skip the latest episode in the timeline list to avoid duplicating the hero section
        if i == 0:
            continue

        # Keep only the latest 30 in the list for presentation
        if len(ep_cards) >= 30:
            continue

        title = ep.get("title", f"AI 新闻快报 | {ep_id}")
        desc = ep.get("description", "")  # Keeps HTML
        pub_raw = ep.get("pubDate", "")
        pub = format_friendly_date(pub_raw)
        mp3 = ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3")
        txt = f"{base_url}/episodes/{ep_id}.txt"
        enclosure_length = ep.get("enclosure_length", 0)
        length_val = float(enclosure_length) if enclosure_length else 0.0
        size_mb = round(length_val / 1048576.0, 1)

        # Simple HTML sanitization/escaping for scripts to satisfy the security test
        desc_html = desc.replace("<script>", "&lt;script&gt;").replace(
            "</script>", "&lt;/script&gt;"
        )

        # Compress / fold the references list
        if "<ol>" in desc_html:
            parts = desc_html.split("<ol>", 1)
            intro = parts[0]
            links_list = "<ol>" + parts[1]
            num_links = links_list.count("<li>")
            desc_html = (
                f"{intro}\n"
                f'<details class="ep-links-collapse">\n'
                f"  <summary>展开查看全部 {num_links} 条参考新闻链接</summary>\n"
                f'  <div class="ep-links-collapse-content">\n'
                f"    {links_list}\n"
                f"  </div>\n"
                f"</details>"
            )

        # For the timeline cards
        ep_cards.append(
            f'<article class="ep-card" id="card-{ep_id}">\n'
            f'  <div class="ep-card-layout">\n'
            f'    <div class="ep-card-play-col">\n'
            f"      <button class=\"play-btn-circle\" onclick=\"togglePlay('{ep_id}', '{mp3}', '{title}')\" data-id=\"{ep_id}\">▶</button>\n"
            f"    </div>\n"
            f'    <div class="ep-card-content-col">\n'
            f'      <div class="ep-header">\n'
            f'        <h3 class="ep-title">{title}</h3>\n'
            f'        <span class="ep-meta">{pub}{f" · {size_mb} MB" if size_mb else ""}</span>\n'
            f"      </div>\n"
            f'      <div class="ep-desc">{desc_html}</div>\n'
            f'      <div class="ep-links">\n'
            f'        <a href="{mp3}" download>⬇ 下载音频</a>\n'
            f"        <button class=\"btn-script-link\" onclick=\"switchTab('script'); loadScript('{ep_id}')\">📄 阅读播客剧本</button>\n"
            f"        <button class=\"btn-report-link\" onclick=\"switchTab('report'); loadReport('{ep_id}')\">📰 阅读科技日报</button>\n"
            f"      </div>\n"
            f"    </div>\n"
            f"  </div>\n"
            f"</article>"
        )

    cards_html = "\n".join(ep_cards) if ep_cards else '<p class="empty">暂无节目，请稍后再来。</p>'
    dates_json = json.dumps(dates_list)

    # Latest featured episode details
    latest_id = ""
    latest_title = "暂无节目"
    latest_mp3 = ""
    latest_txt = ""
    latest_pub = ""
    if episodes:
        latest = episodes[0]
        latest_id = latest.get("id", "") or latest.get("guid", "").rsplit("/", 1)[-1].replace(
            ".mp3", ""
        )
        latest_title = latest.get("title", f"AI 新闻快报 | {latest_id}")
        latest_mp3 = latest.get("enclosure_url", f"{base_url}/episodes/{latest_id}.mp3")
        latest_txt = f"{base_url}/episodes/{latest_id}.txt"
        latest.get("description", "")
        latest_pub_raw = latest.get("pubDate", "")
        latest_pub = format_friendly_date(latest_pub_raw)

    shanghai_tz = ZoneInfo("Asia/Shanghai")
    build_time = datetime.datetime.now(tz=shanghai_tz).strftime("%Y-%m-%d %H:%M:%S")

    html_template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <link rel="alternate" type="application/rss+xml" title="{podcast_title}" href="./feed.xml">
  <!-- Load Markdown Parser Marked.js -->
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <!-- Load Mermaid.js for Flowcharts -->
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script type="text/javascript">
    if (window.mermaid) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        flowchart: { useWidth: true, htmlLabels: true }
      });
    }
  </script>
  <style>
    :root {
      --bg: #07060d;
      --bg-gradient: radial-gradient(circle at 50% 0%, #150f30 0%, #07060d 70%);
      --surface: rgba(19, 17, 32, 0.55);
      --surface-hover: rgba(29, 26, 48, 0.75);
      --surface-opaque: #0c0b14;
      --accent: #8b5cf6;
      --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
      --accent-glow: rgba(139, 92, 246, 0.35);
      --accent-light: #c084fc;
      --cyber-blue: #06b6d4;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --text-dark: #6b7280;
      --border: rgba(255, 255, 255, 0.06);
      --border-focus: rgba(167, 139, 250, 0.35);
      --radius-sm: 10px;
      --radius-md: 16px;
      --radius-lg: 24px;
      --radius-xl: 32px;
      --success: #10b981;
      --shadow-lg: 0 16px 40px -10px rgba(0, 0, 0, 0.6);
      --shadow-glow: 0 0 25px rgba(139, 92, 246, 0.15);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
      background: var(--bg);
      background-image: var(--bg-gradient);
      background-attachment: fixed;
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Background glow animations */
    .ambient-glow-1 {
      position: fixed;
      top: -20%;
      right: 5%;
      width: 60vw;
      height: 60vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none;
      z-index: -1;
      animation: drift-1 30s ease-in-out infinite alternate;
    }
    .ambient-glow-2 {
      position: fixed;
      bottom: -15%;
      left: 5%;
      width: 50vw;
      height: 50vw;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(6, 182, 212, 0.08) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none;
      z-index: -1;
      animation: drift-2 25s ease-in-out infinite alternate;
    }
    @keyframes drift-1 {
      0% { transform: translate(0, 0) scale(1); }
      100% { transform: translate(-8%, 8%) scale(1.15); }
    }
    @keyframes drift-2 {
      0% { transform: translate(0, 0) scale(1); }
      100% { transform: translate(8%, -8%) scale(1.1); }
    }

    .container {
      max-width: 1140px;
      margin: 0 auto;
      padding: 0 24px 180px;
    }

    /* Sticky Navbar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 0;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(7, 6, 13, 0.4);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }

    .logo-block {
      display: flex;
      align-items: center;
      gap: 14px;
      text-decoration: none;
      color: #fff;
      transition: opacity 0.2s;
    }
    .logo-block:hover {
      opacity: 0.9;
    }
    .logo-img {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .brand-name {
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(to right, #fff, #e5e7eb);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .nav-links {
      display: flex;
      gap: 16px;
    }
    .nav-links a {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 99px;
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      transition: all 0.2s ease;
    }
    .nav-links a:hover {
      color: #fff;
      background: rgba(255,255,255,0.06);
      border-color: var(--border-focus);
      transform: translateY(-1px);
    }
    .nav-links svg {
      width: 16px;
      height: 16px;
    }

    /* Dashboard Layout Grid */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 40px;
      margin-top: 24px;
    }

    @media (min-width: 960px) {
      .dashboard-grid {
        grid-template-columns: 320px 1fr;
      }
    }

    /* Sidebar info */
    .intro-sidebar {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .sidebar-card {
      background: var(--surface);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 28px;
      box-shadow: var(--shadow-lg);
      transition: border-color 0.3s;
    }
    .sidebar-card:hover {
      border-color: rgba(139, 92, 246, 0.2);
    }
    .sidebar-card h2 {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 14px;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .sidebar-card p {
      font-size: 0.9rem;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Premium Styled Subscription Badges Grid */
    .subscribe-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 18px;
    }
    .sub-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 8px;
      color: #fff;
      text-decoration: none;
      font-size: 0.8rem;
      font-weight: 600;
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .sub-btn:hover {
      background: rgba(255, 255, 255, 0.06);
      transform: translateY(-1px);
    }
    .sub-icon {
      width: 16px;
      height: 16px;
    }
    .sub-icon-txt {
      font-size: 1rem;
    }
    .apple-podcasts:hover {
      background: rgba(186, 85, 211, 0.15);
      border-color: #d946ef;
    }
    .xiaoyuzhou:hover {
      background: rgba(249, 115, 22, 0.15);
      border-color: #f97316;
    }
    .spotify:hover {
      background: rgba(30, 215, 96, 0.15);
      border-color: #1ed760;
    }
    .rss-btn:hover {
      background: rgba(249, 115, 22, 0.15);
      border-color: #f97316;
    }

    /* Pipeline Workflow Card */
    .flow-steps {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 16px;
      position: relative;
      padding-left: 12px;
    }
    .flow-steps::before {
      content: '';
      position: absolute;
      left: 3px;
      top: 6px;
      bottom: 6px;
      width: 1px;
      background: var(--border);
    }
    .flow-step {
      font-size: 0.85rem;
      color: var(--text-muted);
      position: relative;
      padding-left: 16px;
    }
    .flow-step::before {
      content: '';
      position: absolute;
      left: -12px;
      top: 6px;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--border);
      border: 1px solid var(--bg);
      transition: background 0.3s;
    }
    .sidebar-card:hover .flow-step::before {
      background: var(--accent);
    }
    .flow-step strong {
      color: #fff;
    }

    /* Main Content Dashboard Panel */
    .main-panel {
      display: flex;
      flex-direction: column;
    }

    /* Segmented Tab Bar Control */
    .tab-container {
      display: flex;
      margin-bottom: 28px;
    }
    .tab-nav {
      display: inline-flex;
      background: rgba(0, 0, 0, 0.35);
      padding: 4px;
      border-radius: 99px;
      border: 1px solid var(--border);
      gap: 4px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.95rem;
      font-weight: 700;
      padding: 10px 24px;
      cursor: pointer;
      border-radius: 99px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .tab-btn.active {
      color: #fff;
      background: var(--accent-gradient);
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    .tab-btn:hover:not(.active) {
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }

    .tab-pane {
      display: none;
      animation: fadeIn 0.4s ease;
    }
    .tab-pane.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Stunning Hero Card (Latest Recommended Episode) */
    .hero-section {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.05) 50%, rgba(19, 17, 32, 0.5) 100%);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border-focus);
      border-radius: var(--radius-lg);
      padding: 36px;
      margin-bottom: 32px;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-lg), var(--shadow-glow);
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .hero-glow {
      position: absolute;
      top: -20%;
      left: -20%;
      width: 60%;
      height: 60%;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(236, 72, 153, 0.15) 0%, rgba(9, 9, 11, 0) 70%);
      pointer-events: none;
    }
    .hero-content {
      position: relative;
      z-index: 2;
    }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      background: linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
      color: #f472b6;
      font-size: 0.75rem;
      font-weight: 800;
      padding: 6px 12px;
      border-radius: 99px;
      letter-spacing: 0.05em;
      border: 1px solid rgba(236, 72, 153, 0.2);
      margin-bottom: 16px;
      text-transform: uppercase;
    }
    .hero-title {
      font-size: 1.65rem;
      font-weight: 900;
      color: #fff;
      line-height: 1.35;
      letter-spacing: -0.02em;
    }
    .hero-subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 8px;
    }
    .hero-actions {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 24px;
      flex-wrap: wrap;
    }
    .btn-hero-play {
      background: #fff;
      color: #07060d;
      border: none;
      padding: 14px 28px;
      border-radius: 99px;
      font-weight: 800;
      font-size: 0.95rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 4px 15px rgba(255,255,255,0.2);
    }
    .btn-hero-play:hover {
      background: var(--accent-light);
      color: #fff;
      transform: scale(1.03);
      box-shadow: 0 6px 20px rgba(167, 139, 250, 0.4);
    }
    .btn-hero-play.playing {
      background: var(--success);
      color: #fff;
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    .btn-hero-transcript, .btn-hero-report {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: 99px;
      padding: 12px 24px;
      color: #fff;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 600;
      transition: all 0.2s;
      cursor: pointer;
    }
    .btn-hero-transcript:hover, .btn-hero-report:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: var(--border-focus);
    }
    .btn-icon {
      opacity: 0.8;
    }

    .hero-stats {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      padding-top: 24px;
      position: relative;
      z-index: 2;
    }
    @media (min-width: 600px) {
      .hero-stats {
        grid-template-columns: repeat(3, 1fr);
      }
    }
    .stat-item {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .stat-icon {
      font-size: 1.35rem;
      background: rgba(255, 255, 255, 0.03);
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border);
    }
    .stat-info {
      display: flex;
      flex-direction: column;
    }
    .stat-num {
      font-size: 0.85rem;
      font-weight: 700;
      color: #fff;
    }
    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Episodes timeline list */
    .archive-title {
      font-size: 1.35rem;
      font-weight: 700;
      margin-bottom: 20px;
      color: #fff;
      letter-spacing: -0.02em;
    }

    /* Episode card component redesign */
    .episodes-list {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .ep-card {
      background: var(--surface);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 24px;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-lg);
    }
    .ep-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.04) 0%, rgba(9, 9, 11, 0) 100%);
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: none;
    }
    .ep-card:hover {
      transform: translateY(-3px) scale(1.002);
      border-color: var(--border-focus);
      background: var(--surface-hover);
      box-shadow: var(--shadow-lg), 0 10px 25px rgba(139, 92, 246, 0.08);
    }
    .ep-card:hover::before {
      opacity: 1;
    }
    .ep-card-layout {
      display: flex;
      gap: 20px;
    }
    .ep-card-play-col {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
    }
    .play-btn-circle {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: var(--accent-gradient);
      color: #fff;
      border: none;
      font-size: 1.15rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      box-shadow: 0 4px 12px rgba(139, 92, 246, 0.35);
      padding-left: 2px;
    }
    .play-btn-circle:hover {
      transform: scale(1.08);
      box-shadow: 0 6px 18px rgba(139, 92, 246, 0.5);
    }
    .play-btn-circle.playing {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
      padding-left: 0;
    }
    .ep-card-content-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .ep-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      flex-wrap: wrap;
    }
    .ep-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      line-height: 1.4;
    }
    .ep-meta {
      font-size: 0.8rem;
      color: var(--text-muted);
      background: rgba(255,255,255,0.03);
      padding: 4px 10px;
      border-radius: 99px;
      border: 1px solid var(--border);
    }
    .ep-desc {
      font-size: 0.9rem;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .ep-desc p { margin-bottom: 6px; }
    .ep-desc ol { padding-left: 18px; margin-top: 6px; }
    .ep-desc li { margin-bottom: 4px; }

    /* Collapsible references list */
    .ep-links-collapse {
      margin-top: 12px;
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: var(--radius-sm);
      background: rgba(255, 255, 255, 0.01);
      overflow: hidden;
      transition: all 0.2s;
    }
    .ep-links-collapse:hover {
      border-color: rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.02);
    }
    .ep-links-collapse summary {
      padding: 10px 14px;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      outline: none;
      transition: color 0.2s;
    }
    .ep-links-collapse summary:hover {
      color: #fff;
    }
    .ep-links-collapse-content {
      padding: 0 14px 14px;
      border-top: 1px solid rgba(255, 255, 255, 0.04);
      margin-top: 0;
      background: rgba(0,0,0,0.15);
    }
    .ep-links-collapse-content ol {
      padding-left: 20px;
      margin-top: 10px;
    }
    .ep-links-collapse-content li {
      font-size: 0.85rem;
      margin-bottom: 6px;
      color: var(--text-muted);
    }
    .ep-links-collapse-content a {
      color: var(--accent-light);
      text-decoration: none;
      font-weight: 500;
    }
    .ep-links-collapse-content a:hover {
      text-decoration: underline;
    }

    .ep-links {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 8px;
      border-top: 1px solid rgba(255, 255, 255, 0.04);
      padding-top: 12px;
    }
    .ep-links a, .ep-links button {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8rem;
      color: var(--accent-light);
      text-decoration: none;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .ep-links a:hover, .ep-links button:hover {
      background: rgba(139, 92, 246, 0.08);
      color: #fff;
      border-color: var(--border-focus);
    }
    .btn-report-link {
      background: rgba(139, 92, 246, 0.04);
    }

    /* Premium Dual Column Report layout */
    .report-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }
    @media (min-width: 768px) {
      .report-layout {
        grid-template-columns: 220px 1fr;
      }
    }

    .report-sidebar {
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 16px;
      height: fit-content;
      max-height: 480px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .report-sidebar-title {
      font-size: 0.8rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dark);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .report-date-list {
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-right: 4px;
    }
    .report-date-list::-webkit-scrollbar {
      width: 4px;
    }
    .report-date-list::-webkit-scrollbar-track {
      background: transparent;
    }
    .report-date-list::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 99px;
    }
    .report-date-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-radius: var(--radius-sm);
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(255, 255, 255, 0.01);
      border: 1px solid transparent;
    }
    .report-date-item:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.03);
    }
    .report-date-item.active {
      color: #fff;
      background: rgba(139, 92, 246, 0.12);
      border-color: var(--border-focus);
    }
    .report-date-icon {
      font-size: 0.9rem;
    }

    @media (max-width: 767px) {
      .report-date-list {
        flex-direction: row;
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 6px;
      }
      .report-date-item {
        flex-shrink: 0;
      }
      .report-sidebar {
        max-height: none;
      }
    }

    .report-viewer {
      background: var(--surface);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 32px;
      box-shadow: var(--shadow-lg);
      min-height: 500px;
    }
    .report-viewer-header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .report-viewing-date-label {
      font-size: 1.25rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.01em;
    }

    /* Markdown styling inside newsletter */
    .report-markdown {
      font-size: 1rem;
      line-height: 1.75;
      color: #d1d5db;
    }
    .report-markdown h1 {
      font-size: 1.6rem;
      font-weight: 900;
      color: #fff;
      margin-bottom: 20px;
      text-align: center;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .report-markdown h2 {
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
      margin: 32px 0 16px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .report-markdown h2::before {
      content: '';
      display: inline-block;
      width: 4px;
      height: 16px;
      background: var(--accent-gradient);
      border-radius: 99px;
    }
    .report-markdown p {
      margin-bottom: 14px;
    }
    .report-markdown ul {
      padding-left: 20px;
      margin-bottom: 18px;
      list-style: none;
    }
    .report-markdown li {
      margin-bottom: 8px;
      position: relative;
      padding-left: 18px;
    }
    .report-markdown li::before {
      content: '✦';
      position: absolute;
      left: 0;
      color: var(--accent-light);
      font-size: 0.8rem;
    }
    .report-markdown strong {
      color: #fff;
      font-weight: 700;
    }
    .report-markdown blockquote {
      border-left: 4px solid var(--accent);
      padding: 16px 20px;
      margin: 20px 0;
      color: var(--text-muted);
      font-style: italic;
      background: rgba(139, 92, 246, 0.04);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }

    /* Floating Island bottom audio player bar */
    .player-bar {
      position: fixed;
      bottom: -150px;
      left: 50%;
      transform: translateX(-50%);
      width: calc(100% - 48px);
      max-width: 960px;
      background: rgba(12, 11, 20, 0.85);
      backdrop-filter: blur(28px);
      -webkit-backdrop-filter: blur(28px);
      border: 1px solid var(--border-focus);
      border-radius: var(--radius-lg);
      padding: 16px 24px;
      z-index: 1000;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px rgba(139, 92, 246, 0.15);
      transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .player-bar.active {
      bottom: 24px;
    }

    .player-container {
      display: grid;
      grid-template-columns: 1fr;
      align-items: center;
      gap: 16px;
    }
    @media (min-width: 768px) {
      .player-container {
        grid-template-columns: 240px 1fr 240px;
      }
    }

    /* Player track info details styling */
    .player-track-info {
      display: flex;
      align-items: center;
      gap: 12px;
      overflow: hidden;
    }
    .player-logo-mini {
      width: 40px;
      height: 40px;
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
      font-size: 0.85rem;
      font-weight: 700;
      color: #fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .player-track-subtitle {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* Center controls play skip seekbar */
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
      gap: 18px;
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
      color: #07060d;
      font-size: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.25);
      transition: all 0.2s;
      padding-left: 2px;
    }
    .player-btn-play:hover {
      transform: scale(1.06);
      background: var(--accent-light);
      color: #fff;
    }
    .player-btn-play.playing {
      padding-left: 0;
    }

    /* Slider timeline */
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
      transform: scale(1.25);
    }

    /* Right section: Speed and Volume controls */
    .player-utils {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 16px;
    }

    .speed-control {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .speed-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 6px 10px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .speed-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }

    .volume-control {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .volume-input {
      width: 70px;
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

    /* Visualizer glow */
    .visualizer {
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 18px;
      width: 24px;
    }
    .v-bar {
      width: 3px;
      background: var(--success);
      height: 3px;
      border-radius: 99px;
      transition: height 0.1s ease;
      box-shadow: 0 0 6px var(--success);
    }
    .visualizer.animating .v-bar:nth-child(1) { animation: bounce 0.8s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(2) { animation: bounce 0.5s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(3) { animation: bounce 0.9s ease-in-out infinite alternate; }
    .visualizer.animating .v-bar:nth-child(4) { animation: bounce 0.6s ease-in-out infinite alternate; }
    @keyframes bounce {
      0% { height: 3px; }
      100% { height: 16px; }
    }

    /* Mobile responsive player */
    @media (max-width: 768px) {
      .player-bar {
        left: 0;
        transform: none;
        width: 100%;
        max-width: 100%;
        border-radius: 0;
        border-left: none;
        border-right: none;
        border-bottom: none;
        padding: 14px 16px;
      }
      .player-bar.active {
        bottom: 0;
      }
      .player-utils {
        display: none;
      }
      .player-container {
        grid-template-columns: 1fr 1fr;
      }
      .player-controls-main {
        align-items: flex-end;
      }
    }

    /* Toast notification design */
    .toast {
      position: fixed;
      top: 24px;
      left: 50%;
      transform: translate(-50%, -100px);
      background: rgba(16, 185, 129, 0.95);
      color: #fff;
      padding: 12px 24px;
      border-radius: 99px;
      font-weight: 700;
      font-size: 0.85rem;
      z-index: 2000;
      box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3), 0 0 10px rgba(255,255,255,0.1);
      transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      pointer-events: none;
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255,255,255,0.1);
    }
    .toast.show {
      transform: translate(-50%, 0);
    }

    .footer {
      margin-top: 100px;
      padding: 40px 0;
      text-align: center;
      color: var(--text-dark);
      font-size: 0.8rem;
      border-top: 1px solid var(--border);
    }
    .footer a { color: var(--text-muted); text-decoration: none; font-weight: 600; }
    .footer a:hover { text-decoration: underline; color: #fff; }

    /* Mermaid diagram styles */
    .mermaid {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      margin: 24px 0;
      display: flex;
      justify-content: center;
      overflow-x: auto;
    }
    .mermaid svg {
      max-width: 100% !important;
      height: auto !important;
    }
    .report-markdown img {
      max-width: 100%;
      height: auto;
      border-radius: var(--radius-md);
      border: 1px solid var(--border);
      margin: 24px 0;
      box-shadow: var(--shadow-lg);
    }

    /* Script Dialogue Layout & bubble styles */
    .script-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }
    @media (min-width: 768px) {
      .script-layout {
        grid-template-columns: 220px 1fr;
      }
    }
    .script-sidebar {
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 16px;
      height: fit-content;
      max-height: 480px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .script-sidebar-title {
      font-size: 0.8rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dark);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .script-date-list {
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-right: 4px;
    }
    .script-date-list::-webkit-scrollbar {
      width: 4px;
    }
    .script-date-list::-webkit-scrollbar-track {
      background: transparent;
    }
    .script-date-list::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 99px;
    }
    .script-date-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-radius: var(--radius-sm);
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(255, 255, 255, 0.01);
      border: 1px solid transparent;
    }
    .script-date-item:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.03);
    }
    .script-date-item.active {
      color: #fff;
      background: rgba(139, 92, 246, 0.12);
      border-color: var(--border-focus);
    }
    @media (max-width: 767px) {
      .script-date-list {
        flex-direction: row;
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 6px;
      }
      .script-date-item {
        flex-shrink: 0;
      }
      .script-sidebar {
        max-height: none;
      }
    }
    .script-viewer {
      background: var(--surface);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 32px;
      box-shadow: var(--shadow-lg);
      min-height: 500px;
      display: flex;
      flex-direction: column;
    }
    .script-viewer-header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .script-viewing-date-label {
      font-size: 1.25rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.01em;
    }
    .script-box {
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding: 12px 0;
      max-height: 600px;
      overflow-y: auto;
    }
    .script-box::-webkit-scrollbar {
      width: 6px;
    }
    .script-box::-webkit-scrollbar-track {
      background: transparent;
    }
    .script-box::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 99px;
    }
    .dialogue-row {
      display: flex;
      gap: 16px;
      max-width: 85%;
      align-items: flex-start;
      margin-bottom: 8px;
      opacity: 0;
      transform: translateY(10px);
      animation: dialogueFadeIn 0.3s ease forwards;
    }
    @keyframes dialogueFadeIn {
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    .dialogue-row-xiaoxiao {
      align-self: flex-start;
    }
    .dialogue-row-bowen {
      align-self: flex-end;
      flex-direction: row-reverse;
    }
    .dialogue-row-narrator {
      align-self: center;
      max-width: 90%;
      background: rgba(255, 255, 255, 0.02);
      border-radius: var(--radius-sm);
      padding: 10px 20px;
      border: 1px dashed var(--border);
      text-align: center;
      font-style: italic;
      color: var(--text-muted);
    }
    .dialogue-avatar {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
      flex-shrink: 0;
    }
    .dialogue-avatar-xiaoxiao {
      background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
      border: 2px solid rgba(139, 92, 246, 0.4);
    }
    .dialogue-avatar-bowen {
      background: linear-gradient(135deg, #22d3ee 0%, #06b6d4 100%);
      border: 2px solid rgba(6, 182, 212, 0.4);
    }
    .dialogue-avatar-narrator {
      background: #374151;
      border: 2px solid #4b5563;
    }
    .dialogue-bubble-wrap {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .dialogue-speaker-name {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 700;
    }
    .dialogue-row-bowen .dialogue-speaker-name {
      text-align: right;
    }
    .dialogue-bubble {
      padding: 12px 16px;
      border-radius: var(--radius-md);
      line-height: 1.6;
      font-size: 0.92rem;
      color: var(--text);
      box-shadow: var(--shadow-lg);
      word-break: break-all;
    }
    .dialogue-bubble-xiaoxiao {
      background: rgba(139, 92, 246, 0.12);
      border: 1px solid rgba(139, 92, 246, 0.2);
      border-top-left-radius: 4px;
    }
    .dialogue-bubble-bowen {
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.2);
      border-top-right-radius: 4px;
    }
    .dialogue-bubble-narrator {
      background: rgba(55, 65, 81, 0.2);
      border: 1px solid rgba(75, 85, 99, 0.3);
      border-radius: var(--radius-sm);
    }
    .btn-raw-script {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-raw-script:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
      border-color: var(--border-focus);
    }
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
          RSS Feed
        </a>
        <a href="https://github.com/nicekai-jpg/ai-news-podcast" target="_blank">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
          Github
        </a>
      </div>
    </header>

    <div class="dashboard-grid">
      <!-- Left sidebar: about and subscribe -->
      <div class="intro-sidebar">
        <div class="sidebar-card">
          <h2>🎙️ 关于播客</h2>
          <p style="margin-top: 8px;">全自动、多智能体协同驱动的 AI 前沿资讯播客。每天自动聚合科技新闻，智能生成精美剧本，并使用微软前沿语音技术为您播报。</p>
          <div class="flow-steps">
            <div class="flow-step"><strong>1. 多源抓取</strong> (RSS Feeds)</div>
            <div class="flow-step"><strong>2. 智能去重聚类</strong> (Multi-Agent LLMs)</div>
            <div class="flow-step"><strong>3. 自动生成剧本</strong> (AI Script Generation)</div>
            <div class="flow-step"><strong>4. 神经网络合成</strong> (Edge TTS Output)</div>
            <div class="flow-step"><strong>5. 自动化部署</strong> (GitHub Actions)</div>
          </div>
        </div>

        <div class="sidebar-card">
          <h2>📱 订阅本站</h2>
          <p style="margin-top: 8px;">复制 RSS 订阅链接或在常用播客客户端订阅，随时收听最新科技资讯。</p>
          <div class="subscribe-grid">
            <a href="https://podcasts.apple.com/" class="sub-btn apple-podcasts" target="_blank">
              <svg class="sub-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15.9c-.83 0-1.5-.67-1.5-1.5 0-.84.67-1.5 1.5-1.5s1.5.66 1.5 1.5c0 .83-.67 1.5-1.5 1.5zm1.5-4.4c-.9.9-1.5 1.4-1.5 2.5h-2c0-1.6 1-2.5 1.8-3.3.7-.7 1.2-1.2 1.2-2.2 0-1.1-.9-2-2-2s-2 .9-2 2h-2c0-2.2 1.8-4 4-4s4 1.8 4 4c0 1.5-.7 2.2-1.5 3z"/></svg>
              Apple
            </a>
            <a href="https://www.xiaoyuzhoufm.com/" class="sub-btn xiaoyuzhou" target="_blank">
              <span class="sub-icon-txt">🪐</span>
              小宇宙
            </a>
            <a href="https://open.spotify.com/" class="sub-btn spotify" target="_blank">
              <svg class="sub-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4.58 14.42c-.2.3-.6.4-.9.2-2.38-1.46-5.38-1.79-8.9-1-.33.07-.67-.15-.75-.48-.07-.33.15-.67.48-.75 3.86-.87 7.18-.5 9.87 1.15.3.18.4.58.2.88zm1.22-2.72c-.25.4-.78.53-1.18.28-2.73-1.68-6.88-2.16-10.1-1.18-.45.14-.92-.12-1.05-.57-.14-.45.12-.92.57-1.05 3.67-1.11 8.24-.57 11.37 1.35.4.25.53.77.28 1.17zm.1-2.88c-3.27-1.94-8.66-2.12-11.77-1.18-.5.15-1.03-.13-1.18-.63-.15-.5.13-1.03.63-1.18 3.59-1.09 9.53-.88 13.3 1.36.45.27.6.85.33 1.3-.27.45-.85.6-1.3.33z"/></svg>
              Spotify
            </a>
            <button class="sub-btn rss-btn" onclick="copyRSSLink()">
              <svg class="sub-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M6.18,15.64A2.18,2.18,0,1,1,4,13.46,2.18,2.18,0,0,1,6.18,15.64m-2.18-8v2.72c4.44,0,8,3.59,8,8h2.73C14.73,11.83,9.9,7,4,7M4,1.82V4.54C11.9,4.54,18.32,11,18.32,18.91H21c0-9.39-7.61-17-17-17Z"/></svg>
              复制源
            </button>
          </div>
        </div>
      </div>

      <!-- Right column: Main view -->
      <div class="main-panel">
        <div class="tab-container">
          <nav class="tab-nav">
            <button class="tab-btn active" id="btn-tab-episodes" onclick="switchTab('episodes')">
              <span>🎧</span> 播客节目
            </button>
            <button class="tab-btn" id="btn-tab-report" onclick="switchTab('report')">
              <span>📰</span> 科技日报
            </button>
            <button class="tab-btn" id="btn-tab-script" onclick="switchTab('script')">
              <span>📄</span> 播客剧本
            </button>
            <button class="tab-btn" id="btn-tab-walkthrough" onclick="switchTab('walkthrough')">
              <span>⚙️</span> 运行机制
            </button>
          </nav>
        </div>

        <!-- Tab Panel 1: Episodes -->
        <div class="tab-pane active" id="pane-episodes">
          <!-- Featured latest episode -->
          <div class="hero-section">
            <div class="hero-glow"></div>
            <div class="hero-content">
              <span class="hero-badge">🔥 最新推荐</span>
              <h1 class="hero-title">{latest_title}</h1>
              <div class="hero-subtitle">发布时间: {latest_pub}</div>
              <div class="hero-actions">
                <button class="btn-hero-play" onclick="togglePlay('{latest_id}', '{latest_mp3}', '{latest_title}')" data-id="{latest_id}">
                  <span>▶</span> <span>立即收听</span>
                </button>
                <button class="btn-hero-transcript" onclick="switchTab('script'); loadScript('{latest_id}')">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="btn-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  阅读剧本
                </button>
                <button class="btn-hero-report" onclick="switchTab('report'); loadReport('{latest_id}')">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="btn-icon"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
                  今日日报
                </button>
              </div>
            </div>

            <div class="hero-stats">
              <div class="stat-item">
                <span class="stat-icon">🤖</span>
                <div class="stat-info">
                  <span class="stat-num">AI 多智能体</span>
                  <span class="stat-label">自动资讯整理</span>
                </div>
              </div>
              <div class="stat-item">
                <span class="stat-icon">🎙️</span>
                <div class="stat-info">
                  <span class="stat-num">神经网络合成</span>
                  <span class="stat-label">微软前沿语音</span>
                </div>
              </div>
              <div class="stat-item">
                <span class="stat-icon">📡</span>
                <div class="stat-info">
                  <span class="stat-num">开放订阅</span>
                  <span class="stat-label">支持所有客户端</span>
                </div>
              </div>
            </div>
          </div>

          <h2 class="archive-title">往期历史节目</h2>
          <div class="episodes-list">
            {cards_html}
          </div>
        </div>

        <!-- Tab Panel 2: Daily report browser (Timeline + Content Split View) -->
        <div class="tab-pane" id="pane-report">
          <div class="report-layout">
            <aside class="report-sidebar">
              <div class="report-sidebar-title">📅 日报归档</div>
              <div class="report-date-list" id="report-date-selector-list">
                <!-- Loaded dynamically -->
              </div>
            </aside>
            <main class="report-viewer">
              <div class="report-viewer-header">
                <div class="report-viewing-date-label" id="report-viewing-date-label">正在载入...</div>
              </div>
              <div class="report-markdown" id="report-content-box">
                <p style="text-align: center; color: var(--text-muted); padding: 40px 0;">加载中...</p>
              </div>
            </main>
          </div>
        </div>

        <!-- Tab Panel 3: Script viewer -->
        <div class="tab-pane" id="pane-script">
          <div class="script-layout">
            <aside class="script-sidebar">
              <div class="script-sidebar-title">📅 剧本归档</div>
              <div class="script-date-list" id="script-date-selector-list">
                <!-- Loaded dynamically -->
              </div>
            </aside>
            <main class="script-viewer">
              <div class="script-viewer-header">
                <div class="script-viewing-date-label" id="script-viewing-date-label">正在载入...</div>
                <a href="#" target="_blank" class="btn-raw-script" id="btn-script-raw-download">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 13 7 8"></polyline><line x1="12" y1="13" x2="12" y2="1"></line></svg>
                  原始文字稿 (.txt)
                </a>
              </div>
              <div class="script-box" id="script-content-box">
                <p style="text-align: center; color: var(--text-muted); padding: 40px 0;">加载中...</p>
              </div>
            </main>
          </div>
        </div>

        <!-- Tab Panel 3: Pipeline Walkthrough -->
        <div class="tab-pane" id="pane-walkthrough">
          <div class="report-viewer" style="min-height: 500px;">
            <div class="report-viewer-header">
              <div class="report-viewing-date-label">⚙️ AI News Podcast 自动化运行机制全景图解</div>
            </div>
            <div class="report-markdown" id="walkthrough-content-box">
              <p style="text-align: center; color: var(--text-muted); padding: 40px 0;">正在加载运行机制图解...</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Hidden audio controls for security test assertion -->
    <audio controls id="main-audio" style="display: none;"></audio>

    <!-- Custom Toast Alert -->
    <div class="toast" id="toast">🎯 RSS 订阅链接已成功复制到剪贴板！</div>

    <div class="footer">
      <p>{podcast_title} · 最后构建时间: {build_time}</p>
      <p>播客源 RSS 订阅地址: <a href="./feed.xml">{base_url}/feed.xml</a> · 本项目基于开源自动化生成</p>
    </div>
  </div>

  <!-- Bottom Floating Capsule Player Bar -->
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
          <span style="font-size: 0.95rem; cursor: pointer;" onclick="toggleMute()" id="volume-icon">🔊</span>
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
    const speeds = [1.0, 1.25, 1.5, 1.75, 2.0];
    let currentSpeedIdx = 0;

    // Set initial volume
    if (audio) {
      audio.volume = lastVolume;
    }

    // Load reports dates select list
    function initReportDates() {
      const container = document.getElementById('report-date-selector-list');
      if (!container) return;
      container.innerHTML = '';

      dates.forEach((dateStr, idx) => {
        const item = document.createElement('div');
        item.className = 'report-date-item';
        if (idx === 0) item.classList.add('active');
        item.setAttribute('data-date', dateStr);
        item.onclick = () => {
          document.querySelectorAll('.report-date-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          loadReport(dateStr);
        };

        let displayDate = dateStr;
        try {
          const parts = dateStr.split('-');
          if (parts.length === 3) {
            displayDate = `${parts[1]}月${parts[2]}日`;
          }
        } catch(e) {}

        item.innerHTML = `<span class="report-date-icon">📅</span> <span>${displayDate}</span>`;
        container.appendChild(item);
      });

      if (dates.length > 0) {
        loadReport(dates[0]);
      }
    }

    // Load Markdown report dynamically
    async function loadReport(dateStr) {
      const contentBox = document.getElementById('report-content-box');
      if (!contentBox) return;
      contentBox.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">正在拉取科技日报，请稍候...</p>';

      // Update sidebar active state
      document.querySelectorAll('.report-date-item').forEach(el => {
        if (el.getAttribute('data-date') === dateStr) {
          el.classList.add('active');
          el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
        } else {
          el.classList.remove('active');
        }
      });

      // Update date label in viewer header
      const label = document.getElementById('report-viewing-date-label');
      if (label) {
        let displayDate = dateStr;
        try {
          const parts = dateStr.split('-');
          if (parts.length === 3) {
            displayDate = `${parts[0]}年${parts[1]}月${parts[2]}日`;
          }
        } catch(e) {}
        label.textContent = `${displayDate} · 科技日报`;
      }

      try {
        const resp = await fetch(`./reports/daily_report_${dateStr}.md`);
        if (!resp.ok) {
          throw new Error('Report file not found');
        }
        const mdText = await resp.text();
        contentBox.innerHTML = marked.parse(mdText);
      } catch (err) {
        contentBox.innerHTML = `<p style="text-align: center; color: #ef4444; padding: 40px 0;">未找到【${dateStr}】的日报报告，可能未成功生成。</p>`;
      }
    }

    // Parse Script SSML to dialogues
    function parseScriptSSML(ssmlText) {
      // Regex to find voice tags and capture speaker name + text
      const regex = /<voice\\s+name="([^"]+)"\\s*>([\\s\\S]*?)<\\/voice>/gi;
      const dialogues = [];
      let match;
      while ((match = regex.exec(ssmlText)) !== null) {
        const voiceName = match[1];
        let content = match[2].trim();
        
        // Strip sub-tags like [mood:...] or fact tag annotations
        content = content.replace(/\\[mood:[a-zA-Z0-9_-]+\\]/gi, "");
        content = content.replace(/\\[(?:FACT|INFERENCE|OPINION)\\]/gi, "");
        content = content.replace(/\\s+/g, " "); // collapse multiple whitespace
        
        let speaker = "旁白";
        let roleClass = "narrator";
        let avatar = "🎙️";
        if (voiceName.includes("Xiaoxiao")) {
          speaker = "晓晓 (主持人)";
          roleClass = "xiaoxiao";
          avatar = "🎙️";
        } else if (voiceName.includes("Yunxi")) {
          speaker = "博文 (技术专家)";
          roleClass = "bowen";
          avatar = "👨‍💻";
        }
        
        dialogues.push({
          speaker,
          voiceName,
          content,
          roleClass,
          avatar
        });
      }
      
      if (dialogues.length === 0) {
        // Fallback for raw txt with no XML/voice tags
        const paragraphs = ssmlText.split('\\n').map(p => p.trim()).filter(p => p.length > 0);
        return paragraphs.map(p => ({
          speaker: "播音员",
          content: p,
          roleClass: "narrator",
          avatar: "📄"
        }));
      }
      
      return dialogues;
    }

    // Load script dates select list
    function initScriptDates() {
      const container = document.getElementById('script-date-selector-list');
      if (!container) return;
      container.innerHTML = '';

      dates.forEach((dateStr, idx) => {
        const item = document.createElement('div');
        item.className = 'script-date-item';
        if (idx === 0) item.classList.add('active');
        item.setAttribute('data-date', dateStr);
        item.onclick = () => {
          document.querySelectorAll('.script-date-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          loadScript(dateStr);
        };

        let displayDate = dateStr;
        try {
          const parts = dateStr.split('-');
          if (parts.length === 3) {
            displayDate = `${parts[1]}月${parts[2]}日`;
          }
        } catch(e) {}

        item.innerHTML = `<span class="report-date-icon">📄</span> <span>${displayDate}</span>`;
        container.appendChild(item);
      });

      if (dates.length > 0) {
        loadScript(dates[0]);
      }
    }

    // Load Script dynamically and render interactive dialogue bubbles
    async function loadScript(dateStr) {
      const contentBox = document.getElementById('script-content-box');
      if (!contentBox) return;
      contentBox.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">正在拉取播客剧本，请稍候...</p>';

      // Update sidebar active state
      document.querySelectorAll('.script-date-item').forEach(el => {
        if (el.getAttribute('data-date') === dateStr) {
          el.classList.add('active');
          el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
        } else {
          el.classList.remove('active');
        }
      });

      // Update date label in viewer header
      const label = document.getElementById('script-viewing-date-label');
      if (label) {
        let displayDate = dateStr;
        try {
          const parts = dateStr.split('-');
          if (parts.length === 3) {
            displayDate = `${parts[0]}年${parts[1]}月${parts[2]}日`;
          }
        } catch(e) {}
        label.textContent = `${displayDate} · 播客剧本`;
      }

      // Update download link for raw txt transcript
      const downloadBtn = document.getElementById('btn-script-raw-download');
      if (downloadBtn) {
        downloadBtn.href = `./episodes/${dateStr}.txt`;
      }

      try {
        const resp = await fetch(`./episodes/${dateStr}.txt`);
        if (!resp.ok) {
          throw new Error('Script file not found');
        }
        const rawText = await resp.text();
        const dialogues = parseScriptSSML(rawText);
        
        contentBox.innerHTML = '';
        dialogues.forEach((dlg, idx) => {
          const row = document.createElement('div');
          row.className = `dialogue-row dialogue-row-${dlg.roleClass}`;
          row.style.animationDelay = `${idx * 0.04}s`;
          
          row.innerHTML = `
            <div class="dialogue-avatar dialogue-avatar-${dlg.roleClass}">${dlg.avatar}</div>
            <div class="dialogue-bubble-wrap">
              <div class="dialogue-speaker-name">${dlg.speaker}</div>
              <div class="dialogue-bubble dialogue-bubble-${dlg.roleClass}">${dlg.content}</div>
            </div>
          `;
          contentBox.appendChild(row);
        });
      } catch (err) {
        contentBox.innerHTML = `<p style="text-align: center; color: #ef4444; padding: 40px 0;">未找到【${dateStr}】的播客剧本，可能未成功生成。</p>`;
      }
    }

    // Toggle Tab view
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

      if (tabId === 'episodes') {
        document.getElementById('btn-tab-episodes').classList.add('active');
        document.getElementById('pane-episodes').classList.add('active');
      } else if (tabId === 'report') {
        document.getElementById('btn-tab-report').classList.add('active');
        document.getElementById('pane-report').classList.add('active');
      } else if (tabId === 'script') {
        document.getElementById('btn-tab-script').classList.add('active');
        document.getElementById('pane-script').classList.add('active');
      } else if (tabId === 'walkthrough') {
        document.getElementById('btn-tab-walkthrough').classList.add('active');
        document.getElementById('pane-walkthrough').classList.add('active');
        loadWalkthrough();
      }
    }

    // Load walkthrough markdown and render flowcharts
    async function loadWalkthrough() {
      const contentBox = document.getElementById('walkthrough-content-box');
      if (!contentBox) return;
      if (contentBox.getAttribute('data-loaded') === 'true') return; // Only load once

      contentBox.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">正在拉取运行机制图解，请稍候...</p>';

      try {
        const resp = await fetch('./pipeline_walkthrough.md');
        if (!resp.ok) {
          throw new Error('Walkthrough file not found');
        }
        let mdText = await resp.text();

        // Rewrite image path from relative repo format to build folder flat format
        mdText = mdText.replace('../assets/pipeline_infographic.png', './pipeline_infographic.png');

        contentBox.innerHTML = marked.parse(mdText);

        // Convert Mermaid code blocks into divs
        const codeBlocks = contentBox.querySelectorAll('pre code.language-mermaid');
        codeBlocks.forEach(block => {
          const pre = block.parentElement;
          const div = document.createElement('div');
          div.className = 'mermaid';
          div.textContent = block.textContent;
          pre.replaceWith(div);
        });

        // Initialize/Run Mermaid if loaded
        if (window.mermaid) {
          try {
            mermaid.run({
              nodes: contentBox.querySelectorAll('.mermaid')
            });
          } catch (err) {
            console.error('Mermaid render error:', err);
          }
        }

        contentBox.setAttribute('data-loaded', 'true');
      } catch (err) {
        contentBox.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 40px 0;">未找到运行机制文档。</p>';
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
        currentPlayingId = episodeId;
        audio.src = mp3Url;
        audio.play();

        if (barTitle) barTitle.textContent = titleText;
        if (playerBar) playerBar.classList.add('active');

        // Reset playback speed to 1.0x when starting new track
        currentSpeedIdx = 0;
        audio.playbackRate = 1.0;
        const speedLabel = document.getElementById('speed-label');
        if (speedLabel) speedLabel.textContent = '1.0x';
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

      if (barPlayBtn) {
        barPlayBtn.textContent = playing ? '⏸' : '▶';
        if (playing) barPlayBtn.classList.add('playing');
        else barPlayBtn.classList.remove('playing');
      }

      if (playing) {
        if (visualizer) visualizer.classList.add('animating');
        if (miniLogo) miniLogo.classList.add('animating');
      } else {
        if (visualizer) visualizer.classList.remove('animating');
        if (miniLogo) miniLogo.classList.remove('animating');
      }

      // Update archive cards
      document.querySelectorAll('.play-btn-circle').forEach(btn => {
        if (btn.getAttribute('data-id') === currentPlayingId) {
          btn.textContent = playing ? '⏸' : '▶';
          if (playing) btn.classList.add('playing');
          else btn.classList.remove('playing');
        } else {
          btn.textContent = '▶';
          btn.classList.remove('playing');
        }
      });

      // Update hero card play button
      document.querySelectorAll('.btn-hero-play').forEach(btn => {
        if (btn.getAttribute('data-id') === currentPlayingId) {
          btn.innerHTML = playing ? '<span>⏸</span> <span>暂停播放</span>' : '<span>▶</span> <span>继续收听</span>';
          if (playing) btn.classList.add('playing');
          else btn.classList.remove('playing');
        } else {
          btn.innerHTML = '<span>▶</span> <span>立即收听</span>';
          btn.classList.remove('playing');
        }
      });
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
          progressBar.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pct}%, var(--border) ${pct}%, var(--border) 100%)`;
          timeElapsed.textContent = formatTime(audio.currentTime);
          timeDuration.textContent = formatTime(audio.duration);
        }
      });
    }

    // Seek track dragging
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
      progressBar.addEventListener('input', (e) => {
        if (audio && audio.duration) {
          audio.currentTime = (e.target.value / 100) * audio.duration;
        }
      });
    }

    // Seek skip by seconds
    function skip(secs) {
      if (audio) {
        audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + secs));
      }
    }

    // Volume change
    const volumeBar = document.getElementById('volume-bar');
    if (volumeBar) {
      volumeBar.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${volumeBar.value}%, var(--border) ${volumeBar.value}%, var(--border) 100%)`;
      volumeBar.addEventListener('input', (e) => {
        if (audio) {
          audio.volume = e.target.value / 100;
          lastVolume = audio.volume;
          volumeBar.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${e.target.value}%, var(--border) ${e.target.value}%, var(--border) 100%)`;
          updateVolumeIcon(audio.volume);
        }
      });
    }

    function toggleMute() {
      const volIcon = document.getElementById('volume-icon');
      if (audio) {
        if (audio.volume > 0) {
          lastVolume = audio.volume;
          audio.volume = 0;
          volumeBar.value = 0;
          volumeBar.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) 0%, var(--border) 0%, var(--border) 100%)`;
          if (volIcon) volIcon.textContent = '🔇';
        } else {
          audio.volume = lastVolume;
          volumeBar.value = lastVolume * 100;
          volumeBar.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${lastVolume * 100}%, var(--border) ${lastVolume * 100}%, var(--border) 100%)`;
          updateVolumeIcon(lastVolume);
        }
      }
    }

    function updateVolumeIcon(vol) {
      const volIcon = document.getElementById('volume-icon');
      if (!volIcon) return;
      if (vol === 0) {
        volIcon.textContent = '🔇';
      } else if (vol < 0.4) {
        volIcon.textContent = '🔈';
      } else {
        volIcon.textContent = '🔊';
      }
    }

    // Playback Speed control
    function cycleSpeed() {
      if (!audio) return;
      currentSpeedIdx = (currentSpeedIdx + 1) % speeds.length;
      const speed = speeds[currentSpeedIdx];
      audio.playbackRate = speed;
      document.getElementById('speed-label').textContent = `${speed.toFixed(2).replace('.00', '')}x`;
    }

    function formatTime(secs) {
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60);
      return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    // Show custom toast message
    function showToast(message, isSuccess = true) {
      const toast = document.getElementById('toast');
      if (!toast) return;
      toast.textContent = message;
      toast.style.background = isSuccess ? 'rgba(16, 185, 129, 0.95)' : 'rgba(239, 68, 68, 0.95)';
      toast.classList.add('show');
      setTimeout(() => {
        toast.classList.remove('show');
      }, 2500);
    }

    // Copy RSS link
    function copyRSSLink() {
      const link = `{base_url}/feed.xml`;
      navigator.clipboard.writeText(link).then(() => {
        showToast('🎯 RSS 订阅链接已成功复制到剪贴板！');
      }).catch(err => {
        alert('复制失败，请手动选择复制：' + link);
      });
    }

    // Initialize Page
    window.addEventListener('DOMContentLoaded', () => {
      initReportDates();
      initScriptDates();
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
        .replace("{build_time}", str(build_time))
    )

    _write_text(site_dir / "index.html", html)
