# Clean Up Redundant Data and Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove obsolete configurations, temporary test audio samples, untracked helper scripts, build outputs, and cache directories to clean up the repository structure and save disk space.

**Architecture:** Refactor fallbacks for retired configuration keys (`host_a_voice`, `host_b_voice`, `voice`) in the python codebase, delete unused files/directories, clear build artifacts/caches, and execute verification tests.

**Tech Stack:** Python, pytest

---

### Task 1: Refactor retired configuration fallback logic in Python codebase

**Files:**
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py`
- Modify: `src/ai_news_podcast/cli/podcast_daily.py`

- [x] **Step 1: Modify `src/ai_news_podcast/pipeline/tts_engine.py`** (Done)
- [x] **Step 2: Modify `src/ai_news_podcast/cli/podcast_daily.py`** (Done)

### Task 2: Remove redundant and untracked files

**Files:**
- Delete: `site/episodes/female_test.mp3`
- Delete: `site/episodes/male_test.mp3`
- Delete: `site/episodes/test_minimax_female.mp3`
- Delete: `site/episodes/test_minimax_male.mp3`
- Delete: `site/episodes/chinese_voices/`
- Delete: `site/episodes/benchmark/`
- Delete: `scripts/check_audio.py`
- Delete: `scripts/check_generated_chunks.py`
- Delete: `scripts/slice_and_rename.py`
- Delete: `scripts/transcribe_local.py`
- Delete: `scripts/trim_audio.py`

- [x] **Step 1: Delete redundant files** (Done)

### Task 3: Remove build artifacts and stale caches

**Files:**
- Delete: `dist/`
- Delete: `.pytest_cache/`
- Delete: `.ruff_cache/`
- Delete: All `__pycache__` directories in the repository (contains stale .pyc files)

- [x] **Step 1: Clean build artifacts and caches** (Done)

### Task 4: Verification

- [x] **Step 1: Run pytest to ensure backend logic is healthy** (Done)
