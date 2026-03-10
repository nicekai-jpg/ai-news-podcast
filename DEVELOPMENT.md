# Developer Guide

Welcome to the `ai-news-podcast` development guide. This document outlines how to set up the development environment, run tests, and contribute to the codebase.

## Prerequisites

- **Python**: 3.11 or higher.
- **uv**: Recommended for fast package management (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Git**: For version control.

## Setup Environment

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/<your-username>/ai-news-podcast.git
cd ai-news-podcast
uv sync --dev
```

This will create an isolated virtual environment (`.venv`) and install both runtime and development dependencies.

## Project Structure

```text
ai-news-podcast/
├── config/              # Target configurations and RSS source lists
├── data/                # Generated JSON briefs and Markdown reports
├── docs/                # GitHub Pages web root (HTML web and MP3s)
├── src/ai_news_podcast/ # Core python package
│   ├── cli/             # Command-line entry points
│   ├── pipeline/        # Fetch, process, script, and tts modules
│   └── site_builder/    # Static HTML and RSS XML generators
├── tests/               # Unit and integration tests
├── .env                 # Local API Keys (Not checked into source control)
├── pyproject.toml       # Python package requirements
└── README.md            # Main project documentation
```

## Running Tests

The project uses `pytest` for testing. You can run the entire test suite via `uv`:

```bash
uv run pytest tests/ -v
```

### Testing Specific Components

- **Testing the LLM APIs**:
  ```bash
  uv run pytest tests/test_llm.py
  uv run pytest tests/test_rest_llm.py  # Tests direct Ollama integration
  ```
- **Testing text-to-speech**:
  ```bash
  uv run pytest tests/test_tts.py
  ```

## Adding New Features

1. **New News Sources**: Add new RSS feed URLs to `config/sources.yaml`. Ensure they support full-text output or have easily parsed HTML.
2. **New LLM Backends**: If adding a new LLM provider, integrate it within `src/ai_news_podcast/pipeline/scriptwriter.py`. The project currently supports `google-genai`, `openai`, and direct HTTP calls to local `ollama` endpoints.
3. **CLI Commands**: Add new executable scripts in `src/ai_news_podcast/cli/`. We recommend exposing them via the `uv run python -m ...` pattern.

## Code Style & Linting

While we don't strictly enforce a linter yet, please keep the code clean and well-documented. We recommend using `ruff` for formatting:

```bash
uvx ruff check .
uvx ruff format .
```

## GitHub Actions

The core CD pipeline is located in `.github/workflows/daily.yml`. It runs automatically every day using GitHub Actions. If you submit a PR that alters dependencies, ensure `uv.lock` is updated (`uv lock`) so the CI tests correctly resolve the new environment.
