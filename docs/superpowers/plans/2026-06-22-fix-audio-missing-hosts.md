# Fix Audio and Missing Hosts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the missing female host transcript in the browser UI, resolve the fallback parsing bug in the Python TTS engine, and ensure all generated MP3 files and sentence chunk directories are successfully uploaded to the `gh-pages` branch.

**Architecture:** We will update the browser script parsing regex to recognize `host_b` with underscores, restructure the Python XML voice parser to fallback to substring matching if exact matches fail, and temporarily remove `.gitignore` in GitHub Actions before deploying to GitHub Pages so that untracked generated audio/chunk directories are uploaded correctly.

**Tech Stack:** HTML/JS (browser player), Python (TTS engine), YAML (GitHub Actions workflow)

---

### Task 1: Fix player.js UI Speaker Matching

**Files:**
- Modify: `src/ai_news_podcast/site_builder/static/player.js:262-263, 281-282`

- [ ] **Step 1: Write minimal implementation to match host_b**
      Update the browser-side parsing conditions in `player.js` to search for `"host_b"` in the voice tag names.
      Modify both lines 262 and 281.

- [ ] **Step 2: Verify code changes**
      Ensure `isXx` matches both `host-b` and `host_b` formats.

---

### Task 2: Fix tts_engine.py Voice Tag Parsing

**Files:**
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py:42-60`
- Test: `tests/test_tts_engine.py`

- [ ] **Step 1: Write a failing test in test_tts_engine.py**
      Add a test to `tests/test_tts_engine.py` that passes a voice list but uses tag names containing `host_a_professional` or `host_b_professional` to verify that they are correctly mapped to their respective hosts via fallback instead of defaulting to sequential `idx % 2` matching.

- [ ] **Step 2: Verify test fails**
      Run `pytest tests/test_tts_engine.py` to ensure it fails.

- [ ] **Step 3: Modify parse_dialogue_chunks**
      Update `src/ai_news_podcast/pipeline/tts_engine.py` so that if exact matches against `voices` fail, the parser falls back to checking substrings like `"host_a"`, `"host_b"`, `"xiaoxiao"`, `"yunjian"`, etc.

- [ ] **Step 4: Verify test passes**
      Run the tests and verify they pass.

---

### Task 3: Fix GHA Workflow Deployment

**Files:**
- Modify: `.github/workflows/daily.yml:150-156`

- [ ] **Step 1: Add a step to remove .gitignore before deploying**
      In `.github/workflows/daily.yml` inside the `tts` job, add a command step right before `Deploy GitHub Pages` to remove `.gitignore` (`rm -f .gitignore`). This ensures `peaceiris/actions-gh-pages` does not skip deploying ignored audio/chunk files.
