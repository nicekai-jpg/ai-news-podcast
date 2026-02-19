# 脑活素 AI 新闻播客（全免费、全自动）

这个仓库会每天抓取 AI 领域的 RSS/Atom 新闻源，生成一集 MP3，并更新播客订阅用的 RSS（`docs/feed.xml`）。

快速入口
- 主页：`https://<你的GitHub用户名>.github.io/<仓库名>/`
- 订阅地址（Podcast RSS）：`https://<你的GitHub用户名>.github.io/<仓库名>/feed.xml`

你会得到什么
- 播客订阅地址：`https://<你的GitHub用户名>.github.io/<仓库名>/feed.xml`
- 每集音频：`docs/episodes/YYYY-MM-DD.mp3`
- 每集简介页：`docs/episodes/YYYY-MM-DD.html`

怎么“调用”（本地跑一遍）

推荐用 uv（更快、更稳定）
```bash
cd ai-news-podcast
uv sync
uv run python scripts/run_daily.py --base-url http://localhost --no-audio
```

如果你本机没有 uv，也可以用 pip
```bash
cd ai-news-podcast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_daily.py --base-url http://localhost --no-audio
```

本地生成音频（可选）
- 如果你的网络环境能连上 `edge-tts`，可以去掉 `--no-audio`。

参数
- `--no-audio`：只生成稿子与简介页（用于你先看效果）
- `--date 2026-02-18`：指定生成哪一天（默认按北京时间当天）
- `--base-url https://xxx.github.io/yyy`：用于生成 feed.xml 里的链接（Actions 会自动推断，一般不用手填）

启用 GitHub Pages
1. 把本项目推到 GitHub 公共仓库。
2. 仓库 Settings -> Pages。
3. Build and deployment -> Deploy from a branch。
4. Branch 选 `main`，Folder 选 `/docs`。

每天自动更新
- 已配置 GitHub Actions：`.github/workflows/daily.yml`
- 默认每天 07:30（北京时间）生成一集（workflow 使用 UTC 定时）。

怎么“调用”（让 GitHub 立刻生成一集）
1. 打开仓库 -> Actions -> Daily Podcast
2. 点击 Run workflow

如何保证只播 AI 专业新闻
- 新闻源：编辑 `sources.yaml`，只启用垂直 AI 来源。
- 关键词过滤：编辑 `config.yaml` 里的 `selection.include_keywords`。
