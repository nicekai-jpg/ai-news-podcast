# AI News Podcast (Chinese, Free)

Chinese README: `README.zh-CN.md`

This repo generates a daily AI news podcast episode (MP3) and a podcast RSS feed (`feed.xml`) using only free resources.

What it does
- Input: a curated list of AI-related RSS/Atom feeds (`sources.yaml`)
- Output: `docs/feed.xml` + `docs/episodes/YYYY-MM-DD.mp3` + `docs/episodes/YYYY-MM-DD.html`
- Automation: GitHub Actions runs daily and commits updates
- Hosting: GitHub Pages serves `feed.xml` and the audio files

Local run
```bash
cd ai-news-podcast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_daily.py --base-url http://localhost --no-audio
```

If you are able to access Edge Read Aloud from your network, you can omit `--no-audio` to generate the MP3.

GitHub Pages setup
1. Create a GitHub repo and push this folder.
2. Repo Settings -> Pages
3. Build and deployment -> Deploy from a branch
4. Branch: `main`, Folder: `/docs`

Subscribe in a podcast app
- RSS URL: `https://<your-github-username>.github.io/<repo>/feed.xml`

Notes
- The schedule in `.github/workflows/daily.yml` is UTC. `23:30 UTC` is `07:30` in China.
- If you want to include official English sources, set `enabled: true` for them in `sources.yaml`.
