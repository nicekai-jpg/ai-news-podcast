# 工程代码优化设计

## 背景

项目代码基础质量不错（ruff 零告警），但存在以下共性问题：
- 多处 `except Exception:` 过于宽泛
- 硬编码值散落各模块，应提取到配置
- 跨模块重复逻辑（TTS 文本清洗）
- `html_gen.py`（1173 行）和 `podcastwriter.py`（677 行）过大
- `html_gen.py` 零测试覆盖

## 策略

- **渐进式优化**：按模块逐个处理，每步改动小且可验证
- **严格保持功能不变**：只改内部实现，外部行为完全一致
- **验证方式**：每步完成后运行 `uv run pytest tests/ -v` + `uv run ruff check`

## 优化顺序与内容

### 1. 公共模块提取（基础，后续依赖）

**问题**：TTS 文本清洗逻辑在 3 个文件中重复（`podcastwriter.py`、`tts_engine.py`、`podcast_daily.py`），正则模式也重复定义。

**方案**：创建 `src/ai_news_podcast/text_utils.py`，集中管理：
- `_sanitize_for_tts` / `_clean_tts_text` 合并为统一的 `clean_tts_text()`
- `[FACT|INFERENCE|OPINION]`、`[mood:xxx]` 等正则提取为模块常量
- 三个文件改为 import 使用

**影响**：纯内部重构，外部行为不变。

---

### 2. fetcher.py

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | `_JUNK_SUMMARY_PATTERNS`、`_CATEGORY_KEYWORDS` 硬编码 | 提取到 `config.yaml` |
| 可靠性 | 5 处 `except Exception:` | 收窄为 `httpx.HTTPError`、`ValueError` 等具体异常 |
| 可靠性 | 全文提取失败时日志不够详细 | 增加更详细的日志 |
| 测试 | 现有覆盖较全面 | 补充边界场景 |

---

### 3. processor.py

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | `_AUTHORITY_ORDER`、`_THESIS_TEMPLATES` 硬编码 | 提取到 `config.yaml` |
| 代码质量 | 缺失类型标注（`_dedup_title_fuzzy`、`_dedup_keyword_overlap`、`cluster_stories` 等） | 补全类型标注 |
| 可靠性 | `except Exception:` | 收窄为 `ValueError`、`KeyError` 等 |

---

### 4. podcastwriter.py

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | 677 行过大 | LLM prompt 模板提取到 `src/ai_news_podcast/prompts.py` |
| 代码质量 | `COMPANIES` 列表硬编码 | 提取到 `config.yaml` 的 `entities` 段落 |
| 代码质量 | 重复的文本清洗逻辑 | 使用第 1 步的 `text_utils.py` |
| 可靠性 | LLM 调用失败的 `except Exception:` | 收窄为 `httpx.HTTPError`、`json.JSONDecodeError` |

---

### 5. tts_engine.py

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | 重复的文本清洗逻辑 | 使用第 1 步的 `text_utils.py` |
| 代码质量 | BGM 混音参数、`loudnorm` 参数硬编码 | 提取到 `config.yaml` 的 `tts` 段落 |
| 代码质量 | `_chunk_silence_ms` 中的阈值硬编码 | 提取到配置 |
| 可靠性 | `except Exception:` | 收窄为具体异常 |
| 测试 | 边界场景缺失 | 补充测试 |

---

### 6. html_gen.py（最大改动）

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | 1173 行严重过大 | CSS 提取到 `site_builder/static/style.css`，JS 提取到 `site_builder/static/player.js`，Python 只负责数据逻辑和模板组装 |
| 代码质量 | 内联 HTML 模板 | 提取到 `site_builder/templates/` 目录 |
| 可靠性 | 日期解析的 `except Exception:` | 收窄为 `ValueError` |
| 测试 | 零测试覆盖 | 新增 `test_html_gen.py`，先测数据逻辑，再测模板渲染 |

---

### 7. runner.py

| 类别 | 问题 | 修复 |
|------|------|------|
| 代码质量 | 445 行混合了数据管线、跨期去重、语义相似度计算 | 跨期去重逻辑拆分到 `src/ai_news_podcast/pipeline/dedup.py` |
| 代码质量 | `SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')` 硬编码 | 提取到 `config.yaml` |
| 代码质量 | `limit=14` 硬编码 | 提取到配置 |
| 可靠性 | 多处 `except Exception:` | 收窄为具体异常类型 |

---

### 8. scripts/ 和依赖清理

- `rebuild_site.py`、`rebuild_tts.py` 的 ruff 告警修复（未使用 import、空白行）
- 检查 `requests` 是否实际使用，如未使用则从 `pyproject.toml` 移除
- 统一版本约束策略

## 依赖关系

```
1. text_utils.py（基础，2/4/5 依赖）
2. fetcher.py
3. processor.py
4. podcastwriter.py（依赖 1）
5. tts_engine.py（依赖 1）
6. html_gen.py（独立，最大改动）
7. runner.py（独立）
8. scripts/ 和依赖清理（收尾）
```

## 不做的事

- 不改变 pipeline 的外部行为和输出格式
- 不改变配置文件的整体结构（只新增字段）
- 不引入新的外部依赖
- 不做性能的大幅重构（当前性能已满足需求）
