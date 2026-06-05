# 工程代码优化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 渐进式优化 ai-news-podcast 工程代码，消除重复逻辑、收窄异常捕获、提取硬编码到配置、拆分过大文件、补充测试覆盖，严格保持功能不变。

**架构：** 按模块逐个优化，每步完成后运行 pytest + ruff 验证。先提取公共模块（text_utils.py），再按 pipeline 阶段顺序处理各模块，最后清理 scripts/ 和依赖。

**技术栈：** Python 3.11, ruff, pytest, pydub, edge-tts, httpx

---

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| 创建 | `src/ai_news_podcast/text_utils.py` | 统一的 TTS 文本清洗函数和正则常量 |
| 创建 | `tests/test_text_utils.py` | text_utils 的单元测试 |
| 创建 | `src/ai_news_podcast/prompts.py` | LLM prompt 模板（从 scriptwriter.py 提取） |
| 创建 | `src/ai_news_podcast/pipeline/dedup.py` | 跨期去重逻辑（从 runner.py 提取） |
| 创建 | `tests/test_html_gen.py` | html_gen.py 的单元测试 |
| 修改 | `src/ai_news_podcast/pipeline/scriptwriter.py` | 删除重复清洗逻辑，import text_utils；删除内联 prompt，import prompts |
| 修改 | `src/ai_news_podcast/pipeline/tts_engine.py` | 删除重复清洗逻辑，import text_utils；硬编码提取到配置 |
| 修改 | `src/ai_news_podcast/cli/run_daily.py` | 删除重复正则，import text_utils |
| 修改 | `src/ai_news_podcast/pipeline/fetcher.py` | 收窄异常，硬编码提取到配置 |
| 修改 | `src/ai_news_podcast/pipeline/processor.py` | 收窄异常，补类型标注，硬编码提取到配置 |
| 修改 | `src/ai_news_podcast/pipeline/runner.py` | 收窄异常，硬编码提取到配置，去重逻辑拆到 dedup.py |
| 修改 | `src/ai_news_podcast/site_builder/html_gen.py` | 收窄异常，CSS/JS 提取到静态文件 |
| 修改 | `config/config.yaml` | 新增 entities、tts.audio、processing.dedup 等配置段 |
| 修改 | `scripts/rebuild_site.py` | 修复 ruff 告警 |
| 修改 | `scripts/rebuild_tts.py` | 修复 ruff 告警 |
| 修改 | `pyproject.toml` | 移除未使用的 requests 依赖 |

---

## 任务 1：创建 text_utils.py 公共模块

**文件：**
- 创建：`src/ai_news_podcast/text_utils.py`
- 创建：`tests/test_text_utils.py`

- [ ] **步骤 1：编写 text_utils.py**

将 `scriptwriter.py:63-96` 的 `_sanitize_for_tts` 和 `tts_engine.py:21-44` 的 `_clean_tts_text` 合并为统一的 `clean_tts_text()`。合并逻辑取两者之并集：

```python
"""公共文本清洗工具 — 供 scriptwriter、tts_engine、run_daily 共用。"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 正则常量（消除跨模块重复定义）
# ---------------------------------------------------------------------------

RE_FACT_TAG = re.compile(r"\[(?:FACT|INFERENCE|OPINION)\]\s*")
RE_MOOD_TAG = re.compile(r"\[mood:[^\]]+\]\s*")
RE_EMOJI_PAREN = re.compile(
    r"[（(]\s*(?:doge|狗头|笑|手动狗头|手动|滑稽|哭|捂脸|bushi|划掉)\s*[）)]",
    flags=re.IGNORECASE,
)
RE_FANCY_QUOTES = re.compile(r"[「」『』【】]")
RE_HOST_TAG = re.compile(r"\[(?!(Host\s*A|Host\s*B))[^\]]*\]", flags=re.IGNORECASE)
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_COMMA = re.compile(r"[，,]{2,}")
RE_MULTI_PERIOD = re.compile(r"[。.]{2,}")
RE_EMPTY_PAREN = re.compile(r"[（(]\s*[）)]")


def clean_tts_text(text: str, *, preserve_ssml: bool = True) -> str:
    """清洗 LLM 输出中 TTS 不友好的残留内容。

    合并了原 scriptwriter._sanitize_for_tts 和 tts_engine._clean_tts_text 的全部逻辑。
    当 preserve_ssml=True 时，检测到 SSML 标签则保留 XML 标记。
    """
    if not text:
        return ""

    # 1. 处理转义字符（如模型误输出的 \\n）
    text = text.replace("\\n", "\n")

    # 2. 清除特定标记
    text = RE_FACT_TAG.sub("", text)
    text = RE_MOOD_TAG.sub("", text)
    text = RE_EMOJI_PAREN.sub("", text)
    text = RE_FANCY_QUOTES.sub("", text)

    # 3. 清除非 Host 标签的方括号标注
    text = RE_HOST_TAG.sub("", text)

    # 4. SSML 检测：如果文本包含 SSML 标签则保留 XML，否则清除 HTML 标签
    if preserve_ssml:
        stripped = text.strip()
        is_ssml = (
            stripped.startswith("<speak") or "<speak" in stripped or "<voice" in stripped
        )
        if not is_ssml:
            text = re.sub(r"<[^>]+>", "", text)

    # 5. 清除空括号
    text = RE_EMPTY_PAREN.sub("", text)

    # 6. 压缩重复标点
    text = RE_MULTI_COMMA.sub("，", text)
    text = RE_MULTI_PERIOD.sub("。", text)

    # 7. 规范化空格和换行
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text).strip()
    return text
```

- [ ] **步骤 2：编写 test_text_utils.py**

```python
"""text_utils 公共文本清洗测试。"""

from ai_news_podcast.text_utils import clean_tts_text


class TestCleanTtsText:
    def test_empty_string(self):
        assert clean_tts_text("") == ""

    def test_fact_tag_removal(self):
        assert clean_tts_text("[FACT] 这是事实") == "这是事实"

    def test_inference_tag_removal(self):
        assert clean_tts_text("[INFERENCE] 推论内容") == "推论内容"

    def test_opinion_tag_removal(self):
        assert clean_tts_text("[OPINION] 观点内容") == "观点内容"

    def test_mood_tag_removal(self):
        assert clean_tts_text("[mood:happy] 内容") == "内容"

    def test_emoji_paren_removal(self):
        assert clean_tts_text("好笑(doge) 继续") == "好笑 继续"

    def test_fancy_quotes_removal(self):
        assert clean_tts_text("「引用」内容") == "引用内容"

    def test_host_tag_preserved(self):
        assert "[Host A]" in clean_tts_text("[Host A] 你好")
        assert "[Host B]" in clean_tts_text("[Host B] 你好")

    def test_non_host_bracket_removed(self):
        assert clean_tts_text("[备注] 内容") == "内容"

    def test_escape_newline(self):
        assert clean_tts_text("第一行\\n第二行") == "第一行\n第二行"

    def test_ssml_preserved(self):
        text = '<speak><voice name="v">内容</voice></speak>'
        assert "<speak>" in clean_tts_text(text)

    def test_html_removed_when_not_ssml(self):
        assert clean_tts_text("<b>内容</b>") == "内容"

    def test_multi_comma_compressed(self):
        assert clean_tts_text("内容，，，继续") == "内容，继续"

    def test_multi_newline_compressed(self):
        assert clean_tts_text("第一行\n\n\n\n第二行") == "第一行\n\n第二行"

    def test_idempotent(self):
        text = "[FACT] 测试(mood:happy)「引」\n\n\n内容"
        result = clean_tts_text(text)
        assert clean_tts_text(result) == result
```

- [ ] **步骤 3：运行测试验证通过**

运行：`uv run pytest tests/test_text_utils.py -v`
预期：全部 PASS

- [ ] **步骤 4：替换 scriptwriter.py 中的 _sanitize_for_tts**

在 `scriptwriter.py` 中：
- 添加 `from ai_news_podcast.text_utils import clean_tts_text`
- 将第 490 行的 `script = _sanitize_for_tts(script)` 改为 `script = clean_tts_text(script)`
- 删除 `_sanitize_for_tts` 函数定义（第 63-96 行）

- [ ] **步骤 5：替换 tts_engine.py 中的 _clean_tts_text**

在 `tts_engine.py` 中：
- 添加 `from ai_news_podcast.text_utils import clean_tts_text`
- 将第 62、87、95 行的 `_clean_tts_text(...)` 改为 `clean_tts_text(...)`
- 删除 `_clean_tts_text` 函数定义（第 21-44 行）

- [ ] **步骤 6：替换 run_daily.py 中的内联正则**

在 `run_daily.py` 第 200-204 行，替换为：
```python
from ai_news_podcast.text_utils import clean_tts_text
clean_transcript = clean_tts_text(script_text, preserve_ssml=False) + "\n"
```
删除原来的 4 行内联正则。

- [ ] **步骤 7：替换 scriptwriter.py 第 549 行的 mood 正则**

将 `line = re.sub(r"\[mood:\w+\]\s*", "", line)` 改为使用 `RE_MOOD_TAG`：
```python
from ai_news_podcast.text_utils import RE_MOOD_TAG
line = RE_MOOD_TAG.sub("", line)
```

- [ ] **步骤 8：运行全量测试 + ruff 验证**

运行：`uv run pytest tests/ -v && uv run ruff check src/ tests/`
预期：全部 PASS，零告警

- [ ] **步骤 9：Commit**

```bash
git add src/ai_news_podcast/text_utils.py tests/test_text_utils.py src/ai_news_podcast/pipeline/scriptwriter.py src/ai_news_podcast/pipeline/tts_engine.py src/ai_news_podcast/cli/run_daily.py
git commit -m "refactor: extract shared text_utils module, eliminate duplicate TTS cleaning logic"
```

---

## 任务 2：fetcher.py 优化

**文件：**
- 修改：`src/ai_news_podcast/pipeline/fetcher.py`
- 修改：`config/config.yaml`

- [ ] **步骤 1：将 _JUNK_SUMMARY_PATTERNS 提取到 config.yaml**

在 `config.yaml` 的 `fetch` 段新增：
```yaml
fetch:
  # ... 现有字段 ...
  junk_summary_patterns:
    - "Subscribe to read"
    - "Sign in to read"
    - "Continue reading"
    - "Read more"
    - "Log in to continue"
    - "Premium content"
```

在 `fetcher.py` 中，将 `_JUNK_SUMMARY_PATTERNS` 改为从配置读取，保留原列表作为默认值。

- [ ] **步骤 2：将 _CATEGORY_KEYWORDS 提取到 config.yaml**

在 `config.yaml` 的 `fetch` 段新增 `category_keywords` 字典。在 `fetcher.py` 中改为从配置读取，保留原字典作为默认值。

- [ ] **步骤 3：收窄 fetcher.py 的 5 处 except Exception**

逐个替换：
- 第 68 行 URL 规范化：`except Exception:` → `except (ValueError, UnicodeError):`
- 第 98 行 feed 解析：`except Exception:` → `except (KeyError, ValueError, AttributeError):`
- 第 111 行全文提取：`except Exception:` → `except (OSError, ValueError, UnicodeDecodeError):`
- 第 306 行单条抓取：`except Exception as e:` → `except (httpx.HTTPError, OSError, ValueError) as e:`
- 第 347 行并发任务：`except Exception as e:` → `except (httpx.HTTPError, OSError, ValueError, KeyError) as e:`

- [ ] **步骤 4：运行测试验证**

运行：`uv run pytest tests/test_fetcher.py -v && uv run ruff check src/ai_news_podcast/pipeline/fetcher.py`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/ai_news_podcast/pipeline/fetcher.py config/config.yaml
git commit -m "refactor: fetcher — extract hardcoded patterns to config, narrow exception handling"
```

---

## 任务 3：processor.py 优化

**文件：**
- 修改：`src/ai_news_podcast/pipeline/processor.py`
- 修改：`config/config.yaml`

- [ ] **步骤 1：将 _THESIS_TEMPLATES 提取到 config.yaml**

在 `config.yaml` 的 `processing` 段新增 `thesis_templates` 列表。在 `processor.py` 中改为从配置读取，保留原列表作为默认值。

- [ ] **步骤 2：补全缺失的类型标注**

为以下函数添加返回类型标注：
- `_dedup_title_fuzzy(items: list[RawItem], threshold: float) -> list[RawItem]`
- `_dedup_keyword_overlap(items: list[RawItem], keyword_overlap: float, title_sim: float) -> list[RawItem]`
- `dedup_pipeline(items: list[RawItem], cfg: dict) -> list[RawItem]`
- `cluster_stories(items: list[RawItem], cfg: dict) -> list[dict]`

- [ ] **步骤 3：收窄 processor.py 的 except Exception**

第 113 行 jieba 关键词提取：`except Exception:` → `except (ImportError, RuntimeError):`

- [ ] **步骤 4：运行测试验证**

运行：`uv run pytest tests/test_processor.py -v && uv run ruff check src/ai_news_podcast/pipeline/processor.py`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/ai_news_podcast/pipeline/processor.py config/config.yaml
git commit -m "refactor: processor — extract thesis templates to config, add type annotations, narrow exceptions"
```

---

## 任务 4：scriptwriter.py 优化

**文件：**
- 创建：`src/ai_news_podcast/prompts.py`
- 修改：`src/ai_news_podcast/pipeline/scriptwriter.py`
- 修改：`config/config.yaml`

- [ ] **步骤 1：将 COMPANIES 列表提取到 config.yaml**

在 `config.yaml` 新增 `entities` 段：
```yaml
entities:
  companies:
    - "谷歌"
    - "google"
    - "openai"
    - "微软"
    - "microsoft"
    - "英伟达"
    - "nvidia"
    - "苹果"
    - "apple"
    - "meta"
    - "anthropic"
    - "claude"
    - "字节"
    - "腾讯"
    - "百度"
    - "阿里"
    - "华为"
    - "奥迪"
    - "audi"
    - "特斯拉"
    - "tesla"
```

在 `scriptwriter.py` 中改为从配置读取，保留原列表作为默认值。

- [ ] **步骤 2：将 LLM prompt 模板提取到 prompts.py**

创建 `src/ai_news_podcast/prompts.py`，将 `_build_editor_prompt` 和 `_build_writer_prompt` 中的长字符串模板提取为模块级常量或函数。`scriptwriter.py` 改为 `from ai_news_podcast.prompts import ...`。

- [ ] **步骤 3：收窄 scriptwriter.py 的 except Exception**

第 379 行 LLM 调用：`except Exception as e:` → `except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError) as e:`

- [ ] **步骤 4：运行测试验证**

运行：`uv run pytest tests/test_scriptwriter.py -v && uv run ruff check src/ai_news_podcast/pipeline/scriptwriter.py src/ai_news_podcast/prompts.py`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/ai_news_podcast/prompts.py src/ai_news_podcast/pipeline/scriptwriter.py config/config.yaml
git commit -m "refactor: scriptwriter — extract prompts and entities, narrow exceptions"
```

---

## 任务 5：tts_engine.py 优化

**文件：**
- 修改：`src/ai_news_podcast/pipeline/tts_engine.py`
- 修改：`config/config.yaml`

- [ ] **步骤 1：将 BGM 混音参数提取到 config.yaml**

在 `config.yaml` 的 `tts` 段新增：
```yaml
tts:
  # ... 现有字段 ...
  audio:
    bgm_volume_db: -12
    bgm_fade_in_ms: 2000
    bgm_fade_out_ms: 3000
    vocal_pad_ms: 1000
    loudnorm: "I=-16:LRA=11:TP=-1.5"
    chunk_silence_short: 400
    chunk_silence_long: 800
    chunk_silence_fallback: 100
```

在 `tts_engine.py` 中，将硬编码值改为从配置读取，保留原值作为默认。

- [ ] **步骤 2：收窄 tts_engine.py 的 except Exception**

第 74 行 BeautifulSoup 解析：`except Exception:` → `except (ValueError, OSError):`

- [ ] **步骤 3：运行测试验证**

运行：`uv run pytest tests/test_tts_engine.py -v && uv run ruff check src/ai_news_podcast/pipeline/tts_engine.py`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/ai_news_podcast/pipeline/tts_engine.py config/config.yaml
git commit -m "refactor: tts_engine — extract audio params to config, narrow exceptions"
```

---

## 任务 6：html_gen.py 优化

**文件：**
- 创建：`src/ai_news_podcast/site_builder/static/style.css`
- 创建：`src/ai_news_podcast/site_builder/static/player.js`
- 修改：`src/ai_news_podcast/site_builder/html_gen.py`
- 创建：`tests/test_html_gen.py`

- [ ] **步骤 1：编写 test_html_gen.py 数据逻辑测试**

先为 `format_friendly_date` 和 `build_index_html` 的数据准备逻辑编写测试：

```python
"""html_gen 站点生成测试。"""

import json
import tempfile
from pathlib import Path

from ai_news_podcast.site_builder.html_gen import format_friendly_date, build_index_html


class TestFormatFriendlyDate:
    def test_iso_format(self):
        assert format_friendly_date("2026-06-04") == "2026年6月4日"

    def test_rfc_format(self):
        result = format_friendly_date("Thu, 04 Jun 2026 01:05:50 +0000")
        assert "6月4日" in result

    def test_invalid_returns_original(self):
        assert format_friendly_date("not-a-date") == "not-a-date"


class TestBuildIndexHtml:
    def test_creates_index_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            episodes = [{"id": "2026-06-04", "title": "测试", "pubDate": "Thu, 04 Jun 2026 01:05:50 +0000"}]
            build_index_html(site_dir, "测试播客", episodes, "http://localhost:8000/site")
            assert (site_dir / "index.html").exists()

    def test_index_contains_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            episodes = [{"id": "2026-06-04", "title": "测试", "pubDate": "Thu, 04 Jun 2026 01:05:50 +0000"}]
            build_index_html(site_dir, "测试播客", episodes, "http://localhost:8000/site")
            content = (site_dir / "index.html").read_text()
            assert "测试播客" in content
```

- [ ] **步骤 2：运行测试验证**

运行：`uv run pytest tests/test_html_gen.py -v`
预期：PASS

- [ ] **步骤 3：将 CSS 提取到 static/style.css**

从 `html_gen.py` 的 `build_index_html` 函数中提取所有 `<style>...</style>` 内的 CSS 内容到 `src/ai_news_podcast/site_builder/static/style.css`。在 Python 中改为读取该文件并嵌入。

- [ ] **步骤 4：将 JS 提取到 static/player.js**

从 `html_gen.py` 提取所有 `<script>...</script>` 内的 JavaScript 内容到 `src/ai_news_podcast/site_builder/static/player.js`。在 Python 中改为读取该文件并嵌入。

- [ ] **步骤 5：收窄 html_gen.py 的 4 处 except Exception**

第 16、19、24、27 行的日期解析：`except Exception:` → `except (ValueError, TypeError):`

- [ ] **步骤 6：运行全量测试 + ruff 验证**

运行：`uv run pytest tests/ -v && uv run ruff check src/ tests/`
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add src/ai_news_podcast/site_builder/ tests/test_html_gen.py
git commit -m "refactor: html_gen — extract CSS/JS to static files, add tests, narrow exceptions"
```

---

## 任务 7：runner.py 优化

**文件：**
- 创建：`src/ai_news_podcast/pipeline/dedup.py`
- 修改：`src/ai_news_podcast/pipeline/runner.py`
- 修改：`config/config.yaml`

- [ ] **步骤 1：将跨期去重逻辑拆分到 dedup.py**

从 `runner.py` 提取以下函数到 `src/ai_news_podcast/pipeline/dedup.py`：
- `get_recent_broadcasted_urls`
- `get_recent_broadcasted_texts`
- `extract_semantic_keywords`
- 语义相似度计算相关逻辑

`runner.py` 改为 `from ai_news_podcast.pipeline.dedup import ...`。

- [ ] **步骤 2：将 SentenceTransformer 模型名和 limit 提取到 config.yaml**

在 `config.yaml` 的 `processing.dedup` 段新增：
```yaml
processing:
  dedup:
    # ... 现有字段 ...
    recent_episodes_limit: 14
    semantic_model: "paraphrase-multilingual-MiniLM-L12-v2"
```

在 `runner.py` / `dedup.py` 中改为从配置读取。

- [ ] **步骤 3：收窄 runner.py 的 7 处 except Exception**

逐个替换：
- 第 64、87 行 JSON 解析：→ `except (json.JSONDecodeError, KeyError, ValueError) as e:`
- 第 140 行 URL 读取：→ `except (OSError, ValueError):`
- 第 153、171 行 episodes 加载：→ `except (json.JSONDecodeError, KeyError, ValueError) as e:`
- 第 286 行 SentenceTransformer 加载：→ `except (ImportError, OSError, RuntimeError) as e:`
- 第 346、422 行语义去重：→ `except (ImportError, OSError, ValueError, RuntimeError) as e:`

- [ ] **步骤 4：运行测试验证**

运行：`uv run pytest tests/ -v && uv run ruff check src/ai_news_podcast/pipeline/runner.py src/ai_news_podcast/pipeline/dedup.py`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/ai_news_podcast/pipeline/dedup.py src/ai_news_podcast/pipeline/runner.py config/config.yaml
git commit -m "refactor: runner — extract dedup module, narrow exceptions, extract hardcoded config"
```

---

## 任务 8：scripts/ 和依赖清理

**文件：**
- 修改：`scripts/rebuild_site.py`
- 修改：`scripts/rebuild_tts.py`
- 修改：`pyproject.toml`

- [ ] **步骤 1：修复 rebuild_site.py 的 ruff 告警**

- 删除未使用的 `import json` 和 `import yaml`
- 修复 import 排序（I001）
- 清除空白行中的空白字符（W293）

- [ ] **步骤 2：修复 rebuild_tts.py 的 ruff 告警**

- 删除未使用的 `import yaml`
- 修复 import 排序（I001）

- [ ] **步骤 3：从 pyproject.toml 移除 requests 依赖**

`requests` 在 src/ 中未被使用，从 `dependencies` 列表中移除 `"requests==2.32.3"`。

- [ ] **步骤 4：运行 ruff 验证**

运行：`uv run ruff check src/ tests/ scripts/`
预期：零告警

- [ ] **步骤 5：Commit**

```bash
git add scripts/rebuild_site.py scripts/rebuild_tts.py pyproject.toml
git commit -m "chore: fix scripts lint, remove unused requests dependency"
```

---

## 最终验证

- [ ] **运行全量测试**

运行：`uv run pytest tests/ -v`
预期：全部 PASS

- [ ] **运行 ruff 检查**

运行：`uv run ruff check src/ tests/ scripts/`
预期：零告警

- [ ] **运行 ruff 格式化检查**

运行：`uv run ruff format --check src/ tests/ scripts/`
预期：零差异

- [ ] **最终 Commit（如有格式化修复）**

```bash
git add -A
git commit -m "style: apply ruff formatting"
```
