# AI 新闻播客完整生产计划

> 定位：单人主播叙事独白 · 每天 1 期 · 每期 3–5 条新闻 · 8–15 分钟 · 受众：AI 爱好者
> MVP 不依赖付费 API，可在本地/VPS/GitHub Actions 跑通全链路

---

## 全链路数据流

```
[RSS/Atom 源]
  ↓ 抓取 + 全文提取
[raw_items 库 (SQLite/JSONL)]
  ↓ URL 硬去重 + 标题软去重 + 实体重叠检测
[dedup_items]
  ↓ TF-IDF + DBSCAN 聚类（同事件多源 → 1 story）
[stories]
  ↓ 增补 context_block（3 句事实 + 背景 + 来源列表）
  ↓ 五维打分 → 角色分配 → thesis 提炼
[episode_brief.json]
  ↓ 按模板 A/B 生成脚本（含 [mood:xxx] 情绪标记）
[script.txt]
  ↓ 剥离标签 → 按 mood 分段 TTS → pydub 拼接 → loudnorm
[episode.mp3]
  ↓ 更新 feed.xml + show_notes + transcript
[GitHub Pages / VPS 发布]
```

---

## 第一阶段：新闻资源获取

### 1.1 RSS/Atom 源清单

| 语言 | 来源 | 类型 | Feed URL |
|---|---|---|---|
| 中文 | 机器之心 | 行业/研究 | `https://www.jiqizhixin.com/rss` |
| 中文 | 量子位 | 行业快讯 | `https://www.qbitai.com/feed` |
| 中文 | 智源社区 | 研究/社区 | `https://hub.baai.ac.cn/feed` |
| 英文 | OpenAI News | 官方 | `https://openai.com/news/rss.xml` |
| 英文 | Anthropic News | 官方 | `https://www.anthropic.com/news.rss` |
| 英文 | DeepMind Blog | 官方研究 | `https://deepmind.google/blog/rss.xml` |
| 英文 | Google AI | 官方 | `https://blog.google/technology/ai/rss/` |
| 英文 | Hugging Face Blog | 开源生态 | `https://huggingface.co/blog/feed.xml` |
| 英文 | GitHub Blog AI | 工程实践 | `https://github.blog/category/ai-and-ml/feed/` |
| 英文 | MIT Tech Review AI | 媒体 | `https://www.technologyreview.com/topic/artificial-intelligence/feed/` |
| 英文 | arXiv cs.AI | 论文 | `https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=30` |
| 英文 | arXiv cs.LG | 论文 | `https://export.arxiv.org/api/query?search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=30` |

### 1.2 抓取策略

- 频率：主任务每天跑 1 次（北京时间 08:30）；候选池取过去 36 小时
- HTTP 参数：`timeout=15s`，`connect=5s`，自定义 UA
- 错误处理：
  - 失败重试 2 次（指数退避 2s / 6s）
  - 429 读 Retry-After 并降速
  - 5xx 跳过不阻断
  - 404/410 标记源失效
- 幂等：同一 `normalized_link` 不重复入库；同一 `episode_id` 不重复发布

### 1.3 全文提取（超越 title + description）

- 用 `trafilatura` 从原文链接提取正文前 1200–2000 字符
- fallback：`readability-lxml` + `beautifulsoup4`
- 限速：同域名 1 req/sec，全局并发 ≤4，每期最多抓 80 页

### 1.4 Raw Item 数据结构

```json
{
  "id": "sha256(source_name + '|' + normalized_url)",
  "title": "string",
  "link": "string",
  "normalized_link": "string",
  "source_name": "string",
  "published_at": "RFC3339",
  "summary": "string|null",
  "full_text_snippet": "string|null",
  "category": "model|product|research|open_source|policy|tool|other",
  "language": "zh|en"
}
```

### 1.5 推荐库（免费）

- RSS 解析：`feedparser`
- HTTP 客户端：`httpx`（或 `requests`）
- 全文提取：`trafilatura`
- 兜底提取：`readability-lxml` + `beautifulsoup4`
- 重试：`tenacity`

---

## 第二阶段：信息加工

### 2.1 去重（三层）

1. **URL 硬去重**：规范化后（去 utm_*、统一 https、去尾斜杠）相同即同一条
2. **标题软去重**：`rapidfuzz.token_set_ratio ≥ 92` 且发布时间差 ≤48h → 同事件
3. **实体重叠**：`jieba` 分词取关键词，重叠度 ≥ 0.35 且标题相似度 ≥ 85 → 同事件候选

### 2.2 聚类（同事件多源 → 1 个 story）

- 向量化：TF-IDF（中文用 char n-gram `(2,4)`）
- 聚类：DBSCAN（`metric="cosine"`, `eps=0.35`, `min_samples=2`）
- 未入簇的条目作为独立 story

### 2.3 增补 context_block（每个 story）

```
factual_summary_3s:     从各来源抽取信息量最高 3 句（含数字/机构名/产品名的句子加分）
historical_background:  查 story_memory 表（近 90 天同实体历史），生成 1-2 句背景
sources:                按权威度排序（官方=1 > 一手媒体=2 > 垂直=3 > 聚合=4）
```

### 2.4 打分量表（5 维 × 1–3 分，总分 5–15）

| 维度 | 1 分 | 2 分 | 3 分 |
|---|---|---|---|
| 影响范围 | 局部工具 | 影响一个人群 | 行业级 |
| 新颖性 | 旧闻/常规更新 | 已有趋势新进展 | 范式变化/首次发布 |
| 可解释性 | 高度技术难口播 | 需补背景 | 一句话能讲清 |
| 听众相关性 | 距离远 | 有信息价值 | 直接影响实践 |
| 来源丰富度 | 单源缺正文 | 两源有限细节 | ≥2 源含一手 |

### 2.5 角色分配

| 总分 | 角色 | 数量 | 时长预算 |
|---|---|---|---|
| 12–15 | 🔴 主故事 | 1 条 | 4–6 分钟 |
| 8–11 | 🟡 支撑故事 | 1–2 条 | 2–3 分钟/条 |
| 5–7 | 🟢 快讯 | 1–2 条 | 30–60 秒/条 |
| ≤4 | ⚪ 跳过 | — | — |

### 2.6 Thesis 提炼

- 规则优先：按主故事类别 + 支撑类别组合，从模板库选一句主线
- 无自然主线时降级为 `mode=tool`（"今天给你 2 个能立刻用的 AI 工具/方法"）

### 2.7 输出：episode_brief.json

```json
{
  "episode_id": "2026-02-20",
  "mode": "dots",
  "thesis": "今天的主线是：……",
  "main_story": {
    "story_id": "cluster_12",
    "title": "……",
    "category": "model",
    "score_total": 14,
    "score_breakdown": { "impact_scope": 3, "novelty": 3, "explainability": 2, "listener_relevance": 3, "source_richness": 3 },
    "context_block": {
      "factual_summary_3s": ["[FACT] ……", "[FACT] ……", "[FACT] ……"],
      "historical_background": ["[INFERENCE] ……"],
      "sources": [{ "source_name": "OpenAI News", "url": "https://..." }]
    }
  },
  "supporting_stories": [],
  "quick_hits": [],
  "constraints": { "target_minutes": 10, "story_count": 4 }
}
```

---

## 第三阶段：播客脚本生产

### 3.1 两套模板

#### Mode A「连点成线」（推荐，有论点时用）

```
[0:00–0:40]  Hook：反直觉事实或问题开场（150–180 字）          [mood:hook]
[0:40–1:10]  Thesis：明确主线论点（120–160 字）                [mood:calm]
[1:10–6:30]  主故事：事实→意外点→背景→解读→行动建议           [mood:根据内容切换]
             （1200–1500 字）
[6:30–8:30]  支撑故事 1：事实→与主线关联→一句评论              [mood:根据内容切换]
             （450–550 字）
[8:30–10:00] 支撑故事 2 或快讯区（300–450 字）                [mood:calm]
[10:00–11:30] Quick hits（300–450 字）                       [mood:calm]
[11:30–12:10] Closing：回扣论点+开放问题（150–220 字）         [mood:closing]
```

#### Mode B「工具优先」（素材零散时兜底）

```
[0:00–0:35]  Hook："假设你今天要用 AI 做 X……"                [mood:hook]
[0:35–4:30]  工具/方法 1：是什么→能干什么→跟现有比→谁适合用    [mood:excited/calm]
[4:30–8:00]  工具/方法 2                                     [mood:根据内容切换]
[8:00–11:00] 快讯 3–4 条                                     [mood:calm]
[11:00–12:00] Closing                                        [mood:closing]
```

### 3.2 写作风格指南

#### 语言与节奏
- 口语化但有节奏；句长 12–28 字为主，每段至少 1 个短句做节奏点
- 数字必须给来源或标 `[INFERENCE]`
- 专有名词首次出现加 6–12 字解释
- 每个故事用"问题句"开头、"落点句"收尾
- 转场用内容逻辑，不用"接下来是第 X 条新闻"

#### 禁用词（命中则校验失败）
- "废话不多说""众所周知""颠覆/炸裂/史诗级""据网友称""让我们拭目以待"
- "大家都在用/全网都在传"（除非有数据来源）
- "大家好欢迎收听""感谢收听我们下期再见"

#### 不确定性处理
- `[FACT]`：来源可追溯的事实 → 肯定语气
- `[INFERENCE]`：合理推断 → 必须用"可能/目前信息有限"
- `[OPINION]`：个人观点 → 必须用"我认为/在我看来"
- 主故事必须口播点名至少 1 次主来源

### 3.3 反幻觉检查清单（7 条）

1. 数字/日期是否为 `[FACT]` 且有来源
2. 引用是否在原文中存在
3. 产品/模型命名大小写与版本号一致
4. 因果句是否标注 `[INFERENCE]`
5. 是否出现绝对化词汇
6. 主故事段落是否口播点名来源
7. 故事数 3–5、总字数 1800–3900（≈ 8–15 分钟）

### 3.4 脚本生成方式

- **MVP 默认**：规则拼装（从 context_block 按模板填充，完全确定性）
- **可选升级**：本地 Ollama（免费）或 OpenAI API（付费），用结构化 prompt 生成

### 3.5 Show Notes 模板

```markdown
# AI 日报｜{{date}}｜{{episode_title}}

> 时长：{{duration}}
> 说明：本期内容含事实 [FACT]、推断 [INFERENCE] 与观点 [OPINION]，请以原文为准。

## 今日主线
- {{thesis}}

## 主故事
- {{main_title}}
- 关键点：{{bullet1}} / {{bullet2}} / {{bullet3}}
- 来源：{{source1_url}} / {{source2_url}}

## 支撑故事
- {{supporting_1_title}} — 来源：{{url}}
- {{supporting_2_title}} — 来源：{{url}}

## 快讯
- {{quick_1_title}}：{{url}}
- {{quick_2_title}}：{{url}}

## 免责声明
本播客为信息整理与个人解读，不构成投资/法律建议。
```

---

## 第四阶段：录制成 MP3

### 4.1 情绪标记系统 `[mood:xxx]`

脚本生成阶段在文本中插入情绪标记，TTS 阶段按标记切段，每段使用不同的语音参数：

| 脚本标记 | 场景 | edge-tts (rate/pitch) | CosyVoice 指令 |
|---|---|---|---|
| `[mood:excited]` | 兴奋/重大突破 | `rate="+15%"` `pitch="+10%"` | `"用兴奋的语气播报"` |
| `[mood:serious]` | 严肃/政策分析 | `rate="-5%"` `pitch="-5%"` | `"用严肃的新闻主播语气说"` |
| `[mood:calm]` | 冷静/常规叙述 | `rate="+0%"` `pitch="-2%"` | `"用平稳冷静的语气叙述"` |
| `[mood:emphasis]` | 强调某句话 | `rate="-10%"` `pitch="+5%"` | `"放慢语速，加重语气强调"` |
| `[mood:hook]` | 开场 Hook | `rate="+8%"` `pitch="+8%"` | `"用吸引注意力的悬念语气说"` |
| `[mood:closing]` | 收尾总结 | `rate="-3%"` `pitch="-3%"` | `"用收束总结的语气说"` |

#### 脚本标记示例

```
[mood:hook]
如果我告诉你，这周最重要的 AI 新闻不是某个新模型——你会怎么想？

[mood:calm]
我的观察是：开源 AI 正在完成一次转变。今天这几条新闻，恰好从不同角度印证了这一点。

[mood:excited]
根据 OpenAI 官方博客，他们刚刚发布了全新的推理架构！这是过去一年最大的一次能力跃升。

[mood:serious]
但我们也需要冷静地看到，DeepMind 的安全报告指出了一个关键问题——

[mood:emphasis]
当 AI Agent 真的能自主操作时，安全性就不再是学术讨论了。

[mood:closing]
这就是今天的内容。如果你觉得有收获，我们明天继续。
```

### 4.2 TTS 引擎选择

#### 免费方案对比

| 排名 | 名称 | 许可证 | 情感控制方式 | 中文质量 | 硬件需求 | 适合场景 |
|---|---|---|---|---|---|---|
| ⭐1 | **CosyVoice 3.0**（阿里） | Apache 2.0 | 自然语言指令控制情感 | 接近真人 | GPU 12GB+ | 新闻播报主体 |
| ⭐2 | **ChatTTS** | CC BY-NC 4.0 | 口语标记 `[laugh]`/`[uv_break]` | 非常自然（对话风） | GPU 8GB+ / CPU（慢） | 轻松过渡段 |
| 3 | **GPT-SoVITS v3** | MIT | 参考音频情感克隆 | 优秀 | GPU 12GB+ | 固定主播音色 |
| 4 | **Fish Speech 1.5** | Apache 2.0 | Dual-AR Prosody | 很高 | GPU 16GB+ | 表现力强但资源重 |
| 5 | **EmotiVoice**（网易有道） | Apache 2.0 | 情感提示词 (happy/sad/angry) | 良好 | **CPU 可跑** | 无 GPU 兜底 |
| 兜底 | **edge-tts**（现有） | 免费服务 | prosody rate/pitch 组合 | 良好 | 无需本地算力 | 最简方案 |

#### 硬件可行性

| 环境 | CosyVoice | ChatTTS | EmotiVoice | edge-tts |
|---|---|---|---|---|
| VPS（无 GPU） | ❌ 太慢 | ❌ 太慢 | ✅ 可用 | ✅ 可用 |
| Mac 本地（M 系列） | ✅ MPS 加速 | ✅ MPS 加速 | ✅ | ✅ |
| 带 GPU 云实例 | ✅ 最佳 | ✅ | ✅ | ✅ |

#### CosyVoice 3.0 关键信息

- GitHub：`FunAudioLLM/CosyVoice`
- HuggingFace 模型：`FunAudioLLM/CosyVoice-300M-Instruct`
- Instruct 用法：
  ```python
  model.inference_instruct("今天的天气真不错。", "用兴奋的语气说", "speaker_1")
  ```
- 支持中文方言、多语言、3 秒声音克隆
- 2025 年 12 月发布 3.0：Pronunciation Inpainting，指令驱动语速/情感/音量控制

#### ChatTTS 关键信息

- GitHub：`2noise/ChatTTS`
- 口语标记：`[laugh]`、`[uv_break]`
- 自动添加口语填充词（"嗯""啊"）使语音更自然
- 注意：CC BY-NC 4.0 许可证限制商用

### 4.3 TTS 合成流程

```
1. 按 [mood:xxx] 标记把脚本切成多个 chunk
2. 剥离 [FACT]/[INFERENCE]/[OPINION] 标签
3. 每个 chunk 调用 TTS（edge-tts 用对应 rate/pitch；CosyVoice 用对应指令）
4. 用 pydub 在 chunk 之间插入 300–800ms 静音
5. 拼接完成后 loudnorm 标准化
```

### 4.4 音频后期

```bash
# 响度标准化（-16 LUFS，播客标准）
ffmpeg -i input.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 44100 -ac 2 normalized.wav

# 可选：拼接片头/片尾
# 导出 MP3 128kbps
ffmpeg -i normalized.wav -codec:a libmp3lame -b:a 128k -ar 44100 -ac 2 episode.mp3
```

### 4.5 重试/降级策略

- 每个 chunk 失败重试 3 次（2s / 6s / 15s）
- edge-tts 失败切换备用音色（Xiaoxiao → Yunxi → Xiaoyi）
- 最终兜底：EmotiVoice（CPU 可跑）

### 4.6 输出产物

```
output/2026-02-20/episode.mp3
output/2026-02-20/transcript.txt
output/2026-02-20/show_notes.md
output/2026-02-20/show_notes.html
```

---

## 第五阶段：发布与分发

- 更新 `feed.xml`（保留最近 30 期）
- 托管：GitHub Pages（`docs/`）或 VPS（nginx）
- 订阅地址：`https://<domain>/feed.xml`
- 调度：GitHub Actions 定时（北京 08:30）或 VPS cron

---

## 附录

### A) 推荐落地阶段

| 阶段 | TTS | 脚本 | 说明 |
|---|---|---|---|
| 阶段一（立刻可用） | edge-tts + prosody 分段合成 | 规则拼装 | 无额外硬件 |
| 阶段二（推荐升级） | CosyVoice 3.0 | 规则拼装 / 本地 Ollama | Mac 本地或 GPU 云实例 |
| 阶段三（最佳效果） | CosyVoice + ChatTTS 混合 | LLM 生成 | 主体用 CosyVoice，过渡段用 ChatTTS |

### B) 成本分析

| 模块 | MVP（免费） | 可选付费升级 |
|---|---|---|
| 抓取/解析 | feedparser + httpx | 付费新闻 API |
| 全文提取 | trafilatura | 商业爬虫 |
| 摘要/脚本 | 规则拼装 / 本地 Ollama | OpenAI / Claude API |
| TTS | edge-tts / CosyVoice | ElevenLabs / Azure TTS |
| 托管 | GitHub Pages | 专业播客托管 |

### C) 风险矩阵

| 风险 | 缓解 |
|---|---|
| RSS 源失效 | 多源冗余 + 健康检查 + 自动降级 |
| 反爬 429 | 域名限速 + 退避 + 只保留 summary |
| 同事件重复讲 | 三层去重 + 聚类 |
| 幻觉/编数字 | [FACT] 标注 + 7 条校验清单 |
| edge-tts 封禁 | 重试 + 换音色 + CosyVoice/EmotiVoice 兜底 |
| 版权风险 | 只做摘要+评论，标注来源链接 |

### D) 示例：一期完整脚本大纲

假设当天新闻：
1. OpenAI 发布新推理能力更新（来源：OpenAI News）
2. Hugging Face 上线新评测工具（来源：HF Blog）
3. DeepMind 发布对齐安全研究（来源：DeepMind Blog）
4. 国内某厂开源推理框架更新（来源：量子位）

**模式**：A「连点成线」
**论点**：推理能力在变强，但真正决定体验的是评测与工程化

```
[mood:hook]
模型更聪明了，但为什么你的产品体验不一定变好？

[mood:calm]
今天的主线是：能力外溢 → 评测与工程决定落地

[mood:excited]
主故事：OpenAI 更新推理能力（3 事实句 + 背景 + 影响拆解 + 行动建议）

[mood:calm]
支撑故事：Hugging Face 评测工具如何把评测标准化（2 步上手路径）

[mood:serious]
快讯 1：DeepMind 安全研究结论一句话 + 对行业的提醒

[mood:calm]
快讯 2：国内开源框架更新一句话

[mood:closing]
回扣：从"能跑"到"能用"到"安全地用"，这个转变正在发生。
```
