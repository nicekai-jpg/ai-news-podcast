# Fix Host Voice Mapping and Improve Male Voice Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the host gender voice swap bug by resolving voice tag parsing order, and improve the male voice quality by changing the default voice from YunxiNeural to YunjianNeural.

**Architecture:** We will modify `tts_engine.py` to parse standard SSML `<voice>` tags robustly by matching tag names to host configuration rather than relying on alternation order. We will also update the configuration, default fallbacks, and prompt templates to use `zh-CN-YunjianNeural` as the male voice (Host A) instead of `zh-CN-YunxiNeural`.

**Tech Stack:** Python, Edge TTS, Pytest

---

### Task 1: Update Voice Tag Parsing Logic in TTS Engine

**Files:**
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py:19-47`
- Modify: `src/ai_news_podcast/pipeline/tts_engine.py:261-271`

- [ ] **Step 1: Update signature of `parse_dialogue_chunks` and implement robust voice-to-host mapping**

Modify `parse_dialogue_chunks` definition and internal mapping in `src/ai_news_podcast/pipeline/tts_engine.py`:

```python
def parse_dialogue_chunks(text: str, voices: Optional[Tuple[str, str]] = None) -> list[DialogueChunk]:
    """解析对话文本，支持标准的 SSML (XML/HTML) 格式和自定义 [Host A] / [Host B] 格式。"""
    stripped_text = text.strip()
    if stripped_text.startswith("<speak") or "<speak" in stripped_text or "<voice" in stripped_text:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            voice_tags = soup.find_all("voice")
            if voice_tags:
                chunks: list[DialogueChunk] = []
                for idx, voice_tag in enumerate(voice_tags):
                    voice_name = voice_tag.get("name", "").strip()
                    chunk_text = voice_tag.get_text().strip()
                    if chunk_text:
                        cleaned = clean_tts_text(chunk_text)
                        if cleaned:
                            # 默认根据 idx 交替分配
                            host = "B" if idx % 2 == 1 else "A"
                            if voice_name:
                                vn_lower = voice_name.strip().lower()
                                if voices:
                                    if vn_lower == voices[0].strip().lower():
                                        host = "A"
                                    elif vn_lower == voices[1].strip().lower():
                                        host = "B"
                                else:
                                    # 常见中文音色兜底匹配
                                    if "xiaoxiao" in vn_lower or "host_b" in vn_lower or "host-b" in vn_lower:
                                        host = "B"
                                    elif any(x in vn_lower for x in ("yunxi", "yunjian", "yunyang", "host_a", "host-a")):
                                        host = "A"
                            chunks.append(
                                DialogueChunk(
                                    host=host,
                                    text=cleaned,
                                    voice=voice_name if voice_name else None,
                                )
                            )
                if chunks:
                    return chunks
        except (ValueError, OSError):
            pass
```

- [ ] **Step 2: Update `synthesize` in `src/ai_news_podcast/pipeline/tts_engine.py` to pass `voices` and use `zh-CN-YunjianNeural` as default**

Update `synthesize` definition and call:

```python
async def synthesize(
    text: str,
    *,
    backend: str = "edge-tts",
    voices: Tuple[str, str] = ("zh-CN-YunjianNeural", "zh-CN-XiaoxiaoNeural"),
    output_path: Union[str, Path],
    bgm_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    chunks = parse_dialogue_chunks(text, voices=voices)
```

### Task 2: Add and Run Tests for Voice Parsing

**Files:**
- Modify: `tests/test_tts_engine.py`

- [ ] **Step 1: Add a test case for swapped voice parsing order in `tests/test_tts_engine.py`**

Add the following test to `TestParseDialogueChunks`:

```python
    def test_ssml_parsing_swapped_order(self) -> None:
        text = """
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
          <voice name="zh-CN-XiaoxiaoNeural">
            大家好，我是 B (doge)
          </voice>
          <voice name="zh-CN-YunxiNeural">
            听众朋友大家好
          </voice>
        </speak>
        """
        chunks = parse_dialogue_chunks(text)
        assert len(chunks) == 2
        assert chunks[0] == DialogueChunk(
            host="B", text="大家好，我是 B", voice="zh-CN-XiaoxiaoNeural"
        )
        assert chunks[1] == DialogueChunk(
            host="A", text="听众朋友大家好", voice="zh-CN-YunxiNeural"
        )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts_engine.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/ai_news_podcast/pipeline/tts_engine.py tests/test_tts_engine.py
git commit -m "fix: resolve host voice swapping issue and add tests"
```

### Task 3: Improve Default Male Voice to YunjianNeural

**Files:**
- Modify: `config/config.yaml`
- Modify: `src/ai_news_podcast/cli/run_daily.py`
- Modify: `src/ai_news_podcast/prompts.py`
- Modify: `src/ai_news_podcast/pipeline/scriptwriter.py`

- [ ] **Step 1: Update voice configuration in `config/config.yaml`**

Change `host_a_voice` to `zh-CN-YunjianNeural`:

```yaml
  host_a_voice: "zh-CN-YunjianNeural"  # edge / hybrid 降级时使用
```

- [ ] **Step 2: Update default fallback in `src/ai_news_podcast/cli/run_daily.py`**

Change `host_a_voice` default to `"zh-CN-YunjianNeural"`:

```python
    voices = (
        str(tts_cfg.get("host_a_voice") or "zh-CN-YunjianNeural"),
        str(tts_cfg.get("host_b_voice") or tts_cfg.get("voice") or "zh-CN-XiaoxiaoNeural"),
    )
```

- [ ] **Step 3: Update voice references in prompt templates of `src/ai_news_podcast/prompts.py`**

Replace `zh-CN-YunxiNeural` with `zh-CN-YunjianNeural`:

```python
# Line 105:
  - *音色与标签*：使用 `<voice name="zh-CN-YunjianNeural">`，内容中自称为"博文"。
# Line 142:
  <voice name="zh-CN-YunjianNeural">
# Line 148:
  <voice name="zh-CN-YunjianNeural">
```

- [ ] **Step 4: Update voice count checks in `src/ai_news_podcast/pipeline/scriptwriter.py`**

Modify:

```python
        host_a_count = (
            script.count('name="zh-CN-YunxiNeural"')
            + script.count('name="zh-CN-YunjianNeural"')
            + script.count('name="zh-CN-YunyangNeural"')
        )
```

- [ ] **Step 5: Run full test suite to ensure all tests pass**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config/config.yaml src/ai_news_podcast/cli/run_daily.py src/ai_news_podcast/prompts.py src/ai_news_podcast/pipeline/scriptwriter.py
git commit -m "feat: improve default male voice quality by switching from Yunxi to Yunjian"
```
