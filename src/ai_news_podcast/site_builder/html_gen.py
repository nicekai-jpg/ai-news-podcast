from pathlib import Path
from typing import Any

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
    for i, ep in enumerate(episodes):
        if i >= 30:
            break
        ep_id = ep.get("guid", "").rsplit("/", 1)[-1].replace(".mp3", "")
        title = ep.get("title", ep_id)
        desc = ep.get("description", "")
        pub = ep.get("pubDate", "")
        mp3 = ep.get("enclosure_url", f"{base_url}/episodes/{ep_id}.mp3")
        txt = f"{base_url}/episodes/{ep_id}.txt"
        enclosure_length = ep.get("enclosure_length", 0)
        length_val = float(enclosure_length) if enclosure_length else 0.0
        size_mb = round(length_val / 1048576.0, 1)

        desc_html = (
            desc.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        ep_cards.append(
            f'<article class="ep-card">\n'
            f'  <div class="ep-header">\n'
            f'    <h3 class="ep-title">{title}</h3>\n'
            f'    <span class="ep-meta">{pub}{f" · {size_mb} MB" if size_mb else ""}</span>\n'
            f"  </div>\n"
            f'  <p class="ep-desc">{desc_html}</p>\n'
            f'  <audio controls preload="none" src="{mp3}"></audio>\n'
            f'  <div class="ep-links">\n'
            f'    <a href="{mp3}" download>⬇ 下载音频</a>\n'
            f'    <a href="{txt}" target="_blank">📄 文字稿</a>\n'
            f"  </div>\n"
            f"</article>"
        )

    cards_html = (
        "\n".join(ep_cards)
        if ep_cards
        else '<p class="empty">暂无节目，请稍后再来。</p>'
    )

    html_template = '''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{podcast_title}</title>
  <link rel="alternate" type="application/rss+xml" title="{podcast_title}" href="./feed.xml">
  <style>
    :root {
      --bg: #09090b;
      --surface: #18181b;
      --surface-hover: #27272a;
      --accent: #6d28d9;
      --accent-glow: rgba(109, 40, 217, 0.5);
      --accent-light: #a78bfa;
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
      --border: #27272a;
      --radius: 16px;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      min-height: 100vh;
      overflow-x: hidden;
    }
    .bg-gradient {
      position: fixed;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(circle at 50% 50%, rgba(109, 40, 217, 0.15) 0%, rgba(9, 9, 11, 0) 50%);
      pointer-events: none;
      z-index: -1;
      animation: pulse 15s ease-in-out infinite alternate;
    }
    @keyframes pulse {
      0% { transform: scale(1) translate(0, 0); }
      50% { transform: scale(1.1) translate(2%, 2%); }
      100% { transform: scale(1) translate(-2%, -2%); }
    }
    .container { max-width: 860px; margin: 0 auto; padding: 0 24px; }
    .hero { text-align: center; padding: 80px 0 60px; position: relative; }
    .logo-container { margin-bottom: 24px; }
    .logo {
      width: 160px; height: 160px; border-radius: 32px;
      box-shadow: 0 20px 40px var(--accent-glow);
      border: 1px solid rgba(255,255,255,0.1);
      transition: transform 0.3s, box-shadow 0.3s;
    }
    .logo:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 30px 60px var(--accent-glow); }
    .hero h1 {
      font-size: 3rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: 16px;
      background: linear-gradient(135deg, #fff, #a1a1aa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: var(--text-muted); font-size: 1.15rem; max-width: 500px; margin: 0 auto; }
    .hero-actions { margin-top: 32px; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; }
    .btn {
      display: inline-flex; align-items: center; padding: 12px 24px;
      border-radius: 999px; font-weight: 600; text-decoration: none; transition: all .2s;
    }
    .btn-primary { background: var(--accent); color: #fff; box-shadow: 0 0 20px var(--accent-glow); }
    .btn-primary:hover { background: #7c3aed; transform: translateY(-2px); }
    .btn-outline { background: var(--surface); border: 1px solid var(--border); color: var(--text); }
    .btn-outline:hover { background: var(--surface-hover); border-color: var(--text-muted); }
    
    .section-title {
      font-size: 1.5rem; font-weight: 600; margin-bottom: 32px; padding-bottom: 12px;
      border-bottom: 1px solid var(--border); text-align: center; color: var(--text);
    }
    .ep-card {
      background: rgba(24,24,27,0.6); backdrop-filter: blur(10px);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 24px; margin-bottom: 20px; transition: transform .2s, border-color .2s;
    }
    .ep-card:hover { transform: translateY(-2px); border-color: rgba(167,139,250,0.3); }
    .ep-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
    .ep-title { font-size: 1.15rem; font-weight: 600; color: #fff; flex: 1; }
    .ep-meta { font-size: 0.85rem; color: var(--text-muted); flex-shrink: 0; }
    .ep-desc { font-size: 0.95rem; color: var(--text-muted); margin: 16px 0; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    audio { width: 100%; height: 44px; border-radius: 8px; outline: none; margin-bottom: 16px; }
    audio::-webkit-media-controls-panel { background: var(--surface-hover); color: #fff; }
    .ep-links { display: flex; gap: 20px; }
    .ep-links a { font-size: 0.9rem; color: var(--accent-light); text-decoration: none; font-weight: 500; transition: opacity .2s; }
    .ep-links a:hover { opacity: .8; text-decoration: underline; }
    
    .empty { text-align: center; color: var(--text-muted); padding: 60px 20px; font-size: 1.05rem; }
    .footer { margin-top: 60px; padding: 40px 20px; text-align: center; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border); }
    .footer a { color: var(--accent-light); text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="bg-gradient"></div>
  <div class="container">
    <div class="hero">
      <div class="logo-container">
        <img src="./logo.png" alt="Logo" class="logo">
      </div>
      <h1>{podcast_title}</h1>
      <p>每天 5 分钟，为你捕捉 AI 时代的每一次脉搏。全自动、全天候的前沿科技资讯。</p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="./feed.xml">📡 RSS 订阅</a>
        <a class="btn btn-outline" href="https://github.com/nicekai-jpg/ai-news-podcast" target="_blank">⭐ GitHub 源码</a>
      </div>
    </div>

    <h2 class="section-title">往期节目</h2>
    {cards_html}

    <div class="footer">
      <p>{podcast_title} · 每日定时生成</p>
      <p>播客 RSS 订阅地址: <a href="./feed.xml">{base_url}/feed.xml</a></p>
    </div>
  </div>
</body>
</html>'''

    html = (
        html_template.replace("{podcast_title}", str(podcast_title))
        .replace("{cards_html}", str(cards_html))
        .replace("{base_url}", str(base_url))
    )

    _write_text(site_dir / "index.html", html)
