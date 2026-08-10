# Plan: CosyVoice 原生副语言标记重构

## 1. 背景与目标
- **背景**: 现有工程使用 XML 风格标签（如 `<laughter>`、`<laughing>...</laughing>`）来控制音频情绪，然后在底层通过字符串替换将其转为 CosyVoice 的原生 `[laughter]` 等标记。这导致了 `tts_parser.py` 中需要复杂的标签闭合修复逻辑，且不利于对齐官方的最佳实践。
- **目标**: 废弃中间的 XML 标签层，直接在 Director Agent 的 Prompt 中规定使用 CosyVoice 原生 Token（中括号格式），并引入官方建议的“最佳实践”限制，以提升合成稳定性、简化代码结构。

## 2. 影响范围
| 模块/文件 | 调整内容 |
|---|---|
| `src/ai_news_podcast/prompts.py` | 重写 `DIRECTOR_USER_TEMPLATE` 标签规范与最佳实践；更新 `WRITER_USER_TEMPLATE` 屏蔽词提示。 |
| `src/ai_news_podcast/pipeline/tts_parser.py` | 移除句子切分函数 `split_text_into_sentences` 中修复破裂 XML 标签的冗余代码。 |
| `src/ai_news_podcast/pipeline/tts_engine.py` | 修改正则表达式校验逻辑，从匹配 XML 改为匹配 CosyVoice 的中括号原生 Token。 |
| `src/ai_news_podcast/text_utils.py` | 简化 `RE_HTML_TAG` 过滤规则；更新 `strip_tts_tags`，以清除前端展示时残留的原生 Token。 |
| `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py`| 移除用于将 XML 标签替换为 `[laughter]` 等 Token 的中间转换代码。 |

## 3. 详细执行步骤

- [x] **步骤 1：更新 Prompt 提示词** (`src/ai_news_podcast/prompts.py`)
  - 修改 `WRITER_USER_TEMPLATE` 中关于“纯净输出规范”的约束，加入对中括号情感标签的禁止说明。
  - 完全重写 `DIRECTOR_USER_TEMPLATE`，移除对 `<laughter>` 和 `<strong>` 的说明。
  - 明确引入原子化原生 Token 支持：`[laughter]`, `[breath]`, `[cough]`, `[sigh]`, `[lipsmack]`, `[vocalized-noise]`。
  - 新增最佳实践约束指令（“单句 1-2 个”，“放置于气口或标点前后”，“不可截断成语”等）。

- [x] **步骤 2：精简后端转换与解析逻辑**
  - **后端代码** (`src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py`)：彻底删除将 `<laughter>` 等替换为原生 Token 的映射代码行。
  - **解析代码** (`src/ai_news_podcast/pipeline/tts_parser.py`)：在 `split_text_into_sentences` 函数中，删除遍历修复 `laughing` 和 `strong` 开闭标签的代码块（约 15 行冗余代码）。

- [x] **步骤 3：重构正则表达式校验层**
  - 核心正则：为了防止误伤原有的 `[Host A]` 标记，我们需要定义精准匹配原生 Token 的正则：`r"\[(?:laughter|breath|cough|sigh|lipsmack|vocalized-noise)\]"`
  - **`tts_engine.py`**：将 `synthesize` 函数中检测是否需要触发 Director Agent 的正则表达式（`r"<[^>]+>"`）替换为原生 Token 正则；同时更新 `_annotate_text_in_batches` 中用于统计正文字数（防缩水机制）的正则表达式。
  - **`text_utils.py`**：将 `RE_HTML_TAG` 恢复为通用的 XML 剥离正则 `r"<[^>]+>"`，不再做基于 Token 的白名单保留；在 `strip_tts_tags` 中加入原生 Token 正则替换，确保展示给用户的文本彻底干净。

## 4. 测试与验收
- [x] 运行本地单元测试：`uv run pytest tests/test_tts_engine.py tests/test_text_utils.py -v`。
- [x] 修复可能由于正则表达式变更或 `replace` 逻辑移除而导致失败的旧版测试用例（如包含 XML 标签的 mock 数据需更新为原生 Token）。
- [x] 进行端到端试运行，检查合成结果是否能正常包含 `[breath]` 等口癖，并确保代码不抛出意外的格式异常。
