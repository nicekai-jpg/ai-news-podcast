# AI Daily Pioneer

Chinese README: `README.zh-CN.md`

This repository generates a daily AI news podcast episode (MP3), daily text reports, and an RSS feed (`feed.xml`) using AI models and text-to-speech. It has been structured as a standard Python package.

## Features
- **RSS Fetching & Scoring**: Fetches news from curated feeds (`config/sources.yaml`), de-duplicates stories (semantic similarity + overlaping keywords), and scores them based on relevance.
- **LLM Integration**: Uses **MiniMax-M3** (via MiniMax's OpenAI-compatible API) to score/cluster stories, generate summaries/outlines (Editor Agent), write dual-host dialogue scripts in SSML format (Writer Agent), and compile markdown reports.
- **Text-to-Speech (TTS)**: Synthesizes high-fidelity audio using **CosyVoice 2** (via zero-shot voice cloning with host reference audio clips) as the primary engine, with **Edge-TTS** acting as the fallback.
- **Audio Mixing & Mastering**: Utilizes **pydub** and **FFmpeg** to pad speech segments, mix vocal tracks with background music, and apply standard loudness normalization (`loudnorm`).
- **Automation**: Runs daily via GitHub Actions (`.github/workflows/daily.yml`) to commit script updates and deploy media assets.
- **Hosting**: GitHub Pages serves the podcast RSS feed (`feed.xml`), show notes HTML, and audio assets.

## Local Setup

### 1. Environment Variables
Copy or create a `.env` file in the project root:
```env
# Example .env configuration
MINIMAX_API_KEY="your-minimax-api-key"
```

### 2. Installation (using `uv`)
We recommend using [uv](https://docs.astral.sh/uv/) for fast dependency management.
```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync
### 3. Running the Sub-Commands

**A. Generate Podcast (MP3 + RSS)**
Main entrypoint that pulls RSS feeds, runs the Multi-Agent pipeline (via MiniMax-M3) to write scripts, synthesizes speech with CosyVoice 2 (or Edge-TTS fallback), mixes BGM, and publishes RSS feeds and web players.
```bash
uv run podcast-daily --base-url http://localhost
```
*Note: Default TTS is CosyVoice 2. Add `--no-audio` to skip audio synthesis and generate text-only materials.*

**B. Generate Markdown Text Report**
Generates a detailed daily AI news markdown report using the MiniMax-M3 model.
```bash
uv run podcast-report
```

## TTS Evaluation & Selection
We have benchmarked multiple Text-to-Speech (TTS) models (including Edge-TTS, ChatTTS, CosyVoice 2, F5-TTS, GPT-SoVITS, Kokoro, MOSS-TTS) for our dual-host podcast scene. 
* For a complete guide covering model evaluations, optimal 2C2G ECS server queue designs, code implementations, web player buffering state machines, and GHA workflows, see the [TTS Complete Guide & System Design](docs/tts_complete_guide.md).

## Configuration Files
The project logic is controlled by files in the `config/` directory:
- `config.yaml`: Core settings for LLM prompts, TTS voices, and scraping limits.
- `sources.yaml`: List of RSS/Atom feeds to fetch news from. Include or disable specific sources here.

## GitHub Pages Deployment
1. Push this repository to GitHub.
2. Go to **Settings** -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `gh-pages` branch and the `/ (root)` folder.
5. Your Podcast RSS URL will be: `https://<your-username>.github.io/<repo-name>/feed.xml`

Subscribe to this URL in your favorite podcast app!

## Daily Automation Setup
- **Configure Secret**: In your GitHub repository, go to **Settings** -> **Secrets and variables** -> **Actions**. Click **New repository secret**, name it `MINIMAX_API_KEY`, and paste your MiniMax API key as the value.
- The repository has a built-in GitHub Actions workflow at `.github/workflows/daily.yml`.
- It runs automatically every day to generate the Markdown daily report and MP3 podcast episode, committing the outputs back to the repository.
- You can also manually trigger the run under the **Actions** tab of your repository by clicking **Run workflow**.
