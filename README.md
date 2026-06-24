# AI Daily Pioneer 🎙️ (AI 每日先锋)

[![GitHub license](https://img.shields.io/github/license/nicekai-jpg/ai-news-podcast?style=flat-square)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![LLM Engine](https://img.shields.io/badge/LLM--Engine-MiniMax--M3-red?style=flat-square&logo=openai)](https://api.minimaxi.com/)
[![TTS Engine](https://img.shields.io/badge/TTS--Engine-CosyVoice2-green?style=flat-square&logo=googleplay)](https://github.com/FunAudioLLM/CosyVoice)
[![Automation](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blueviolet?style=flat-square&logo=githubactions)](.github/workflows/daily.yml)
[![Hosting](https://img.shields.io/badge/Hosting-GitHub%20Pages-orange?style=flat-square&logo=githubpages)](https://pages.github.com/)

[**中文说明文档** (Chinese README)](README.zh-CN.md)

**AI Daily Pioneer** is a fully automated, production-grade AI podcast and news briefing generator. Every day, it fetches tech news from curated RSS/Atom sources, filters and prioritizes them using NLP and scoring algorithms, orchestrates a **Multi-Agent Writer** to compose a natural dual-host script, synthesizes high-fidelity voice dialogue using zero-shot clone voice cloning via **CosyVoice 2**, and deploys the generated audio and web-pages automatically to **GitHub Pages**.

---

## 🗺️ System Pipeline & Architecture

The project is structured as an end-to-end data pipeline executed daily:

```
[Curated RSS/Atom Sources]
           │
           ▼  (fetcher.py)
   [httpx Async Fetching & readability-lxml Full-text Parsing]
           │
           ▼  (processor.py)
   [Deduplication: TF-IDF + Overlap Keywords + Semantic Similarity]
           │
           ▼
   [DBSCAN Clustering & 5-Dimensional Relevance Scoring]
           │
           ▼  (scriptwriter.py)
   [Editor Agent: Outlines thesis, headlines, and quick updates] ──► (MiniMax-M3 LLM)
           │
           ▼
   [Writer Agent: Composes SSML dual-host dialogue scripts] ──► (MiniMax-M3 LLM)
           │
           ▼  (tts_engine.py)
   [TTS Voice Synthesis: Zero-shot speaker cloning] ──► (CosyVoice 2 Engine)
           │
           ▼
   [Audio Mastering: Dynamic segment padding, BGM mixing, & EBU R128 loudnorm] ──► (pydub & FFmpeg)
           │
           ▼  (site_builder/)
   [Static Web Pages & Apple Podcasts Feed XML Generation]
           │
           ▼  (daily.yml / prune_pages.yml)
   [Git Commit & Push to main / Deploy to gh-pages Branch]
```

---

## 🌟 Technical Highlights & Core Features

### 1. Intelligent Filtering & Scoring
* **Semantic Deduplication**: Uses TF-IDF, RapidFuzz, overlapping keywords, and multilingual semantic embedding models (`paraphrase-multilingual-MiniLM-L12-v2`) to prevent duplicate stories across feeds.
* **DBSCAN Clustering**: Groups related news reports from different publishers to detect hot trends.
* **5-Dimensional Scoring**: Ranks news items based on Impact Scope, Novelty, Explainability, Relevance, and Source Authority, selecting the absolute best for the daily show.

### 2. Multi-Agent Dual-Host Scriptwriting
* Powered by the flagship **MiniMax-M3** reasoning model, using a multi-agent framework:
  - **Editor Agent**: Distills raw material into a JSON structure defining the episode thesis, two deep headlines, and three quick updates.
  - **Writer Agent**: Translates the outline into a lively, colloquial script featuring two hosts: **晓晓 (PM-perspective, conversational, metaphor-heavy)** and **博文 (Tech-極客 perspective, analytical, structural)**.
* Restricts reasoning output (`<think>`) lengths via custom system guidelines to fit complete XML/SSML scripts within the 4096-token output limit without clipping.

### 3. High-Fidelity Zero-Shot Clone TTS
* Synthesizes audio segments via **CosyVoice 2** (`CosyVoice2-0.5B` model) with zero-shot speaker-cloning from host reference clips (`assets/refs/`).
* Splits script dialogues into smaller, natural sentences before voice synthesis to prevent model speed warp or audio truncation.
* Designed solely around **CosyVoice 2** speaker cloning; the legacy **Edge-TTS** engine has been archived and is disabled.

### 4. Professional Audio Mastering
* Inserts dynamic silent spacing between hosts' voice segments and paragraphs using **pydub**.
* Blends clean host vocals with background music (`bgm_placeholder.wav`) using automated cross-fades.
* Normalizes final master audio volume to standard broadcasting level using **FFmpeg's EBU R128 loudness filter** (`loudnorm`).

### 5. Media Anti-Bloat Branch Strategy
To keep the primary Git branch light and fast to pull, the repository splits content:
* **`main` Branch (Code & Scripts)**: Contains source code, YAML configurations, Markdown files, and `.txt`/`.html` script transcript files. **No binary audio `.mp3` files are allowed**.
* **`gh-pages` Branch (Media & Hosting)**: Serves static player files, RSS feed, and full MP3 files.
* **Monthly Pruning Job (`prune_pages.yml`)**: Automatically backs up the past 30 days of active episodes, cleans the entire `gh-pages` commit history (resetting commits count to 1), and force-pushes the backup back up, physically deleting old audio data to prevent Git size bloat.
---

## 🛠️ Tech Stack & Third-Party Tools

The system relies on a curated set of specialized python libraries and system-level utilities:

### 1. Data Fetching & Scraper Engines
* **`feedparser`** (`v6.0.11`): Decodes and structures incoming RSS and Atom xml payloads.
* **`httpx`**: Asynchronous HTTP client executing concurrent web scraping requests.
* **`readability-lxml`**: High-performance HTML structural text parser used to extract clean, readable text from web pages, bypassing headers, navigation links, and footers.
* **`beautifulsoup4`**: Parsers and cleans up inline HTML entities and custom tag annotations.

### 2. NLP, Clustering & Scoring Engines
* **`sentence-transformers`** (`v3.0.0`): Computes dense vector embeddings from articles using `paraphrase-multilingual-MiniLM-L12-v2` for semantic similarity mapping.
* **`scikit-learn`**: Utilizes `TfidfVectorizer` for keyword weighting and `DBSCAN` for spatial density-based news clustering.
* **`jieba`**: Segments Chinese text to calculate keyword overlap matrices between candidate news pieces.
* **`rapidfuzz`**: Extremely fast C-implemented fuzzy string comparison for redundant title checks.

### 3. LLM Orchestration
* **`openai`**: Leveraged as the standard OpenAI API-compliant interface to query the **MiniMax-M3** flagship reasoning model for Agent scriptwriting.

### 4. Audio Processing & Mastering
* **`pydub`**: Splits vocal chunks, dynamic silent padding, and cross-fades host voices.
* **`FFmpeg`**: System mastering utility for mixing vocal stems with BGM and running EBU R128 standard loudness normalizations.
* **`CosyVoice 2`** (locally installed/GHA setup): Zero-shot speaker cloning model (`CosyVoice2-0.5B`) executing speech synthesis from reference audios. *Note: The legacy `Edge-TTS` engine has been archived and is disabled.*

### 5. Config & Infrastructure
* **`PyYAML`**: Standard YAML loader for configs (`config.yaml`, `sources.yaml`).
* **`python-dotenv`**: Environment config loader.
* **`tenacity`**: Retrying logic with exponential backoff on HTTP/API limits.

---

## 📂 Project Structure & Module Map

```
ai-news-podcast/
├── .github/workflows/          # GHA automation workflows
│   ├── ci.yml                  # Linting & code format QA (Ruff)
│   ├── daily.yml               # Daily content pipeline & audio synthesis GHA
│   └── prune_pages.yml         # Monthly gh-pages history optimization
├── assets/                     # Soundscapes & clone reference clips
│   ├── audio_samples/          # Edge-TTS samples
│   ├── refs/                   # CosyVoice clone reference files (host_a_ref.wav, etc.)
│   └── bgm_placeholder.wav     # Background BGM audio track
├── config/                     # Pipeline configurations
│   ├── config.yaml             # Prompt guidelines, weights, and audio mix options
│   └── sources.yaml            # Monitored RSS/Atom endpoints
├── data/                       # Local data stores
│   ├── briefs/                 # Pipeline output briefs caches (brief_YYYY-MM-DD.json)
│   ├── reports/                # LLM markdown reports outputs (daily_report_YYYY-MM-DD.md)
│   └── episodes.json           # Episode records index
├── docs/                       # Technical documentations & guides
├── scripts/                    # Helper setup scripts
├── site/                       # Static web pages & RSS build outputs
├── src/ai_news_podcast/        # Main Python Package Source
│   ├── cli/                    # CLI commands endpoints
│   ├── pipeline/               # Core pipeline modules
│   │   ├── fetcher.py          # Stage 1: Async scraper
│   │   ├── processor.py        # Stage 2: Deduplication, clustering & scoring
│   │   ├── scriptwriter.py     # Stage 3: LLM Script generator
│   │   ├── tts_engine.py       # Stage 4: CosyVoice TTS synthesis
│   │   └── runner.py           # Pipeline runner coordinator
│   ├── site_builder/           # Stage 5: RSS Feed & web site builder
│   ├── prompts.py              # LLM prompt templates
│   ├── text_utils.py           # TTS text-cleaning helpers
│   └── utils.py                # Helpers & YAML loaders
├── pyproject.toml              # Project dependency setup
└── uv.lock                     # Locked dependency trees
```

---

## 🛠️ Local Installation & Development

### 1. Requirements
* Python 3.11+
* FFmpeg (must be installed on the system path for audio mixing)
* [uv](https://docs.astral.sh/uv/) (highly recommended for fast package installation)

### 2. Installation
Clone the repository and sync the dependencies:
```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
MINIMAX_API_KEY="your-minimax-api-key"
```

### 4. Running the CLI Commands

* **A. Run the entire pipeline (Fetch → Deduplicate → Cluster → Score)**
  ```bash
  uv run podcast-pipeline --date 2026-06-23
  ```
  *(Generates `data/briefs/brief_2026-06-23.json`)*

* **B. Generate Markdown Daily News Report**
  ```bash
  uv run podcast-report --date 2026-06-23
  ```
  *(Generates `data/reports/daily_report_2026-06-23.md`)*

* **C. Generate Daily Podcast Script & Audio**
  ```bash
  # Generate dialogue transcript and web page (skip audio synthesis)
  uv run podcast-daily --date 2026-06-23 --no-audio --base-url http://localhost
  
  # Full generation including audio (requires local CosyVoice env configured)
  uv run podcast-daily --date 2026-06-23 --base-url http://localhost
  ```

---

## ☁️ Automation & Production Hosting

### 1. Setup GitHub Actions Secrets
In your GitHub repository, navigate to **Settings** -> **Secrets and variables** -> **Actions**. 
Add a **New repository secret**:
* **Name**: `MINIMAX_API_KEY`
* **Value**: *Your MiniMax API Key*

### 2. Configure GitHub Pages
1. Push the code to your GitHub repository.
2. Go to **Settings** -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch** as the source.
4. Set the branch to `gh-pages` and folder to `/ (root)`.
5. Your public Podcast RSS Feed URL will be:
   `https://<your-username>.github.io/<repo-name>/feed.xml`

### 3. Rebuild or Dispatch Runs manually
If you ever want to rebuild a specific date (e.g. if the LLM output failed or script got truncated), you can run:
```bash
gh workflow run "Daily Podcast" -f date=YYYY-MM-DD
```
Since GHA contains a safeguard to skip writing if `site/episodes/YYYY-MM-DD.txt` already exists, you can manually fix any script text, push it, and trigger the GHA run, and GHA will synthesize the audio from your corrected script.

## 🤝 Open Source License & Attributions

We want to express our deepest gratitude to the developers and maintainers of the following third-party projects and open-source libraries that make **AI Daily Pioneer** possible:

### 1. Attributions & Third-Party Software Licenses
* **`CosyVoice`** (Licensed under **Apache-2.0**): Used for zero-shot cloning text-to-speech synthesis.
* **`sentence-transformers`** (Licensed under **Apache-2.0**): Used for computing article embeddings.
* **`scikit-learn`** (Licensed under **BSD 3-Clause**): Used for TF-IDF vectorization and DBSCAN clustering.
* **`openai`** (Licensed under **Apache-2.0**): Used to interact with OpenAI-compatible APIs.
* **`httpx`** (Licensed under **BSD 3-Clause**): Used for asynchronous HTTP requests.
* **`pydub`** (Licensed under **MIT**): Used for audio slicing and track mixing.
* **`rapidfuzz`** (Licensed under **MIT**): Used for fast string comparison.
* **`jieba`** (Licensed under **MIT**): Used for Chinese text tokenization.
* **`feedparser`** (Licensed under **BSD 2-Clause**): Used for RSS/Atom XML feed parsing.
* **`readability-lxml`** (Licensed under **Apache-2.0**): Used to extract readable main text from web articles.
* **`beautifulsoup4`** (Licensed under **MIT**): Used for HTML tag stripping and formatting.
* **`PyYAML`** (Licensed under **MIT**): Used for configuration file reading.
* **`tenacity`** (Licensed under **Apache-2.0**): Used for retry operations.
* **`python-dotenv`** (Licensed under **BSD 3-Clause**): Used to load local environment configurations.

### 2. Licensing Compliance & Terms
* This entire project, including all its custom source code, configurations, and files, is licensed under the permissive **MIT License** (see [LICENSE](LICENSE) for details).
* Since we have removed `trafilatura` (GPLv3) and rely entirely on `readability-lxml` and `beautifulsoup4` for web page content extraction, this project has **zero copyleft dependencies**.
* All dependencies declared in this project are licensed under business-friendly, permissive licenses (MIT, BSD, or Apache-2.0), ensuring that the codebase remains fully compatible under the **MIT License** for commercial use, modification, and redistribution.

