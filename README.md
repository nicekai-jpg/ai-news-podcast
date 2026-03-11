# AI Daily Pioneer

Chinese README: `README.zh-CN.md`

This repository generates a daily AI news podcast episode (MP3), daily text reports, and an RSS feed (`feed.xml`) using AI models and text-to-speech. It has been structured as a standard Python package.

## Features
- **RSS Fetching & Scoring**: Fetches news from curated feeds (`config/sources.yaml`) and scores them based on relevance.
- **LLM Integration**: Uses LLMs (OpenAI, Gemini, or local models via Ollama) to summarize news and write podcast scripts or text reports.
- **Text-to-Speech**: Synthesizes audio using Edge TTS and mixes it with background music.
- **Automation**: GitHub Actions (`.github/workflows/daily.yml`) runs daily to commit updates.
- **Hosting**: GitHub Pages serves the podcast `feed.xml` and HTML/audio assets.
- **Specialized Reports**: Includes scripts for general daily AI reports and specialized AI-in-Education reports.

## Local Setup

### 1. Environment Variables
Copy or create a `.env` file in the project root:
```env
# Example .env configuration
GEMINI_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key" # Works with DeepSeek, SiliconFlow, or native OpenAI
LLM_API_KEY="ollama" # For local model deployment
```

### 2. Installation (using `uv`)
We recommend using [uv](https://docs.astral.sh/uv/) for fast dependency management.
```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync
```

Alternatively, use standard pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Running the Sub-Commands

**A. Generate Podcast (MP3 + RSS)**
This is the main script that pulls news, generates a script via LLM, synthesizes audio with Edge-TTS, and updates the site/RSS feed.
```bash
uv run python -m ai_news_podcast.cli.run_daily --base-url http://localhost
```
*Note: Add `--no-audio` if you cannot connect to Edge-TTS or only want the text script.*

**B. Generate Markdown Text Report (General AI)**
Generates a markdown text report directly targeting local Ollama models (default model: `qwen3.5:27b`).
```bash
uv run python -m ai_news_podcast.cli.daily_report
```

**C. Generate Markdown Text Report (AI in Education)**
Uses a specialized configuration (`config_edu.yaml`, `sources_edu.yaml`) to output a daily AI in Education report.
```bash
uv run python -m ai_news_podcast.cli.daily_report_edu
```

## Configuration Files
The project logic is controlled by files in the `config/` directory:
- `config.yaml` / `config_edu.yaml`: Core settings for LLM prompts, TTS voices, and scraping limits.
- `sources.yaml` / `sources_edu.yaml`: List of RSS/Atom feeds to fetch news from. Include or disable specific sources here.

## GitHub Pages Deployment
1. Push this repository to GitHub.
2. Go to **Settings** -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `main` branch and the `/docs` folder.
5. Your Podcast RSS URL will be: `https://<your-username>.github.io/<repo-name>/feed.xml`

Subscribe to this URL in your favorite podcast app!
