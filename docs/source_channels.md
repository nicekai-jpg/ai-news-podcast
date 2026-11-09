# 多通道信息采集设计（目标态 · v4）

> 规则（按最新确认）：
> 1. **每个通道只收 AI 相关的源；一个通道 = 一种独立抓取/解析机制**；
> 2. **同一源出现在多个通道时全部保留**，不做跨通道去重（去重交给 Stage 1）；
> 3. **Atom 与 RSS 是两种格式，拆成两个通道**；
> 4. 删除：官方 API 通道、Google News 关键词通道；
> 5. wechat2rss（0 成本）作为独立通道纳入。

## 1. 通道总表

| 通道 | 抓取机制 | 格式 | 源数量 | 说明 |
|---|---|---|---|---|
| C1 RSS | 直连 RSS 2.0 URL | RSS 2.0 | 15 | 官方博客、媒体、arXiv 分类 RSS |
| C2 Atom | 直连 Atom 1.0 URL | Atom 1.0 | 15 + 仓库 + 频道 | arXiv 查询、GitHub Releases、YouTube |
| C3 RSSHub | 临时 RSSHub 实例 + 路由 | 以 RSS 2.0 为主 | 54 | 没有稳定原生 feed 的 AI 官方/媒体/社区 |
| C4 网页抓取 | HTML 解析 | HTML | 13 | 厂商官网、政策公告、Telegram |
| C5 wechat2rss | 公共公众号 feed | RSS 2.0 | 19 | 免费名单中的 AI 相关号（0 成本） |
| ~~官方 API~~ | — | — | 删除 | 通用平台，不是 AI 专用源 |
| ~~Google News~~ | — | — | 删除 | 关键词搜索重复率高、噪声大 |

> C6 私有桥（需要微信登录态：wewe-rss / we-mp-rss / wechat2rss 私有部署）：**可选模块，另立项**，不进主流水线。

## 2. C1 RSS 通道（RSS 2.0，直连）

| 分组 | 源 | 采集内容 |
|---|---|---|
| 官方博客 | OpenAI | 官方新闻/研究 |
| 官方博客 | DeepMind | 官方博客 |
| 官方博客 | Google AI | 官方博客 |
| 官方博客 | NVIDIA 官方博客 | 公司动态 |
| 官方博客 | Microsoft Research AI | 研究动态 |
| 官方博客 | Apple ML Research | 研究动态 |
| 官方博客 | HuggingFace Blog | 博客/模型动态 |
| 官方博客 | Qwen 通义千问技术博客 | 技术博客 |
| 中文媒体 | 量子位 | 中文 AI 媒体 |
| 中文媒体 | 少数派 | 科技媒体 |
| 英文媒体 | TechCrunch AI | 英文科技媒体 |
| 英文媒体 | Synced Review | 英文 AI 媒体（陈旧，待观察） |
| 英文媒体 | VentureBeat AI | 英文 AI 媒体（429，待观察） |
| arXiv RSS | arXiv cs.AI | 论文列表（`rss.arxiv.org/rss/cs.AI`） |
| arXiv RSS | arXiv cs.LG | 论文列表（`rss.arxiv.org/rss/cs.LG`） |

## 3. C2 Atom 通道（Atom 1.0，直连）

### 3.1 arXiv 查询（`export.arxiv.org/api/query`，Atom 1.0）

| 查询 | 内容 |
|---|---|
| `abs:DeepSeek OR ti:DeepSeek` | DeepSeek 相关论文 |
| `abs:GLM-4 OR ti:GLM-4 OR abs:ZhipuAI` | 智谱相关论文 |
| `abs:Moonshot OR ti:Kimi` | Kimi 相关论文 |
| `ti:MiniMax` | MiniMax 相关论文 |
| `abs:Anthropic OR ti:Claude` | Anthropic/Claude 相关论文 |
| `abs:xAI OR ti:Grok` | xAI/Grok 相关论文 |
| `abs:Mistral OR abs:Mixtral OR ti:Llama` | Mistral/Llama 相关论文 |
| `abs:Hunyuan` | 腾讯混元相关论文 |
| `abs:ByteDance OR abs:Seed-TTS` | 字节相关论文 |
| `abs:InternLM` | 上海 AI Lab 相关论文 |
| `abs:ERNIE OR abs:PaddlePaddle` | 百度相关论文 |
| `cat:cs.CL` | 计算与语言分类 |
| `cat:cs.CV` | 计算机视觉分类 |
| `cat:cs.RO` | 机器人分类 |

### 3.2 GitHub Releases（Atom，`github.com/{repo}/releases.atom`）

关注 AI 仓库（**2026-11-08 实测**，8/10 可用）：

| 仓库 | 状态 | 最新 release |
|---|---|---|
| `deepseek-ai/deepseek-harness` | ✅ 10 条 | 2026-11-04 |
| `deepseek-ai/DeepGEMM` | ✅ 10 条 | 2026-07-20 |
| `deepseek-ai/DeepEP` | ✅ 1 条 | 2025-09-16 |
| `vllm-project/vllm` | ✅ 10 条 | 2026-11-02 |
| `ggml-org/llama.cpp` | ✅ 10 条 | 2026-11-08 |
| `huggingface/transformers` | ✅ 10 条 | 2026-11-06 |
| `langchain-ai/langchain` | ✅ 10 条 | 2026-11-05 |
| `ollama/ollama` | ✅ 10 条 | 2026-11-06 |
| `deepseek-ai/DeepInfer` | ⚠️ 无 release（空 feed） | — |
| `QwenLM/Qwen3` | ⚠️ 无 release（空 feed） | — |

### 3.3 YouTube 频道（Atom，`youtube.com/feeds/videos.xml?channel_id=`）

| 频道 | channel_id | 实测 |
|---|---|---|
| OpenAI | `UCXZCJLdBC09xxGZ6gcdrc6A` | ✅ 15 条，最新 2026-11-06 |
| Google DeepMind | `UCP7jMXSY2xbc3KCAE0MHQ-A` | ✅ 15 条，最新 2026-10-27 |
| Anthropic | `UCrDwWp7EBBv4NwvScIpBDOA` | ✅ 15 条，最新 2026-09-23 |

## 4. C3 RSSHub 通道（最终筛选清单，以 4.8 为准）

> 与 C1/C2 重复的源也保留（按规则不跨通道去重）。

### 4.1 AI 公司/研究院官方（17）

OpenAI、Anthropic、DeepMind、DeepSeek、Google、Meta AI、Cohere、Cline、Cursor、Windsurf、Ollama、智源 BAAI、中国人工智能学会、国科大 AI、Google PAIR、MIT HAN Lab、Qwen（通义）。

### 4.2 AI 垂直媒体/工具（16）

AI工具集、AIbase、AI财经社、极术社区、量子位、AI 博客、PixelsTech、DeepLearning.AI、Perplexity、BFL、LangChain、Kiro、HuggingFace、Podwise、Civitai、Chub。

### 4.3 泛科技媒体（14）

TechCrunch、The Verge、WIRED、36Kr、IT之家、极客公园、虎嗅、钛媒体、雷峰网、爱范儿、InfoQ、少数派、澎湃、MIT 科技评论中文。

### 4.4 社区/榜单（7）

GitHub、Hacker News、Product Hunt、知乎、微博、B 站、小宇宙。

### 4.5 2026-11-08 本地实测结果

54 个源逐条验证（本地临时 RSSHub 实例）：**可用 40 个，移除 14 个**。

| 移除原因 | 源 | 详情 |
|---|---|---|
| 抓取失败 | DeepMind | 路由超时（C1 原生 RSS 已覆盖，可作 failover） |
| 抓取失败 | Qwen | 路由 404（C1 原生 RSS 已覆盖） |
| 抓取失败 | Perplexity | 503 |
| 抓取失败 | GitHub Trending | 503 |
| 抓取失败 | Product Hunt | 503 |
| 抓取失败（需 cookie） | 微博热搜 | 503 |
| 抓取失败（需 cookie） | B站排行 | 503 |
| 抓取失败 | 小宇宙 | 503 |
| 内容陈旧（>90 天） | AI财经社 | 最新 2023-03 |
| 内容陈旧（>90 天） | AI 博客 | 最新 2025-12 |
| 内容陈旧（>90 天） | Google PAIR | 最新 2025-09 |
| 内容陈旧（>90 天） | MIT HAN Lab | 最新 2026-02 |
| 内容陈旧（>90 天） | Meta AI | 最新 2026-07 |
| 内容陈旧（>90 天） | Windsurf | 最新 2026-05 |

可用 40 个：AI 官方 11、AI 垂直 13、泛科技 14、社区 2（HN、知乎日报）。

### 4.6 加固后的扫描方法与结果（2026-11-09）

**五层扫描法**（可复现）：

1. **版本锚定**：以实际部署镜像 `ghcr.io/diygod/rsshub:latest` 内的
   `/app/assets/build/routes.json` 为准（2087 命名空间 / 4059 路由）；
   docs 的 routes.json 基本同步；jsDelivr 文件列表（1557）版本不可靠。记录镜像 digest。
2. **全量特征扫描**：对全部命名空间计算 AI 信号 —— 词组级词表（英文按词边界、中文按子串）
   + AI 品牌/域名白名单 + 路由级路径/参数/示例 + 官方分类 + heat；并用 TF-IDF 相似度
   对全部命名空间排序交叉验证，人工复核候选。
3. **Radar 规则扫描**：解析镜像内 `radar-rules.json`（1563 个站点），按域名/标题扫 AI 关键词，
   发现通用站点里的 AI 栏目（如 21财经 AI 智讯、极客公园 AI 新浪潮、高校 AI 新闻网）。
4. **路由参数级探测**：对泛科技媒体逐路由测试 AI 专用参数（tag/category/hub），
   如 IT之家 `/ithome/tag/AI`、量子位 `/qbitai/tag/大模型`、少数派 `/sspai/tag/AI`、
   WIRED `/wired/tag/artificial-intelligence`、NYT `/nytimes/rss/Technology`、Nature `/nature/news`。
5. **独立清单反查**：200+ 条已知 AI 源（媒体/Newsletter/公司官方/研究/社区）逐个核对，
   不存在的登记为“RSSHub 缺失”。

**新增且实测可用的 AI 源（原 54 未包含）**：

| 类别 | 源 | 路由 | 实测 |
|---|---|---|---|
| AI 媒体 | The Decoder | `/the-decoder/news` | ✅ 30 条，11-08 |
| AI 媒体 | The Gradient | `/thegradient/posts` | ⚠️ 11 条，2026-02（陈旧） |
| AI 媒体 | Cool Papers（cs.AI） | `/papers/category/arxiv/cs.AI` | ✅ 50 条，11-05 |
| AI 媒体 | MIT Technology Review AI | `/technologyreview/artificial-intelligence` | ✅ 10 条，11-06 |
| AI 媒体 | Forward Future | `/forwardfuture/daily` | ✅ 6 条，11-05 |
| 官方 | NVIDIA Newsroom | `/nvidia/newsroom` | ✅ 20 条，11-08 |
| 官方 | NVIDIA Developer Blog | `/nvidia/developer` | ✅ 20 条，11-06 |
| 官方 | Claude Blog | `/claude/blog` | ✅ 11 条，11-05 |
| 官方 | Claude Code Changelog | `/claude/code/changelog` | ✅ 25 条，11-06 |
| 官方 | Qwen Research | `/qwen/blog` | ✅ 40 条，09-20（替代 404 的 qwenlm） |
| 官方 | DeepSeek 更新日志 | `/deepseek/changelog/zh` | ✅ 22 条，10-17 |
| 官方 | Transformer Circuits | `/transformer-circuits` | ✅ 57 条，09-30 |
| 官方 | Inception Labs | `/inceptionlabs/blog` | ✅ 13 条，10-15 |
| 官方 | Thinking Machines Lab | `/thinkingmachines/news` | ✅ 14 条，11-05 |
| 社区 | 掘金 AI 编程 | `/juejin/aicoding` | ✅ 12 条，11-08 |
| 可选 | 扣子商店 | `/coze/store/project` | ✅ 20 条，11-08 |
| 可选 | 快科技 | `/mydrivers/bcid/801` | ✅ 20 条，11-08 |
| 可选 | Bloomberg Tech | `/bloomberg/technology` | ✅ 6 条，11-08 |
| 可选 | Zread 热门 | `/zread/trending` | ✅ 16 条（无日期） |
| 参数路由 | IT之家 AI 标签 | `/ithome/tag/AI` | ✅ 20 条 |
| 参数路由 | 量子位 大模型标签 | `/qbitai/tag/大模型` | ✅ 10 条，10-20 |
| 参数路由 | 少数派 AI 标签 | `/sspai/tag/AI` | ✅ 50 条，11-07 |
| 参数路由 | NYT Technology | `/nytimes/rss/Technology` | ✅ 26 条，11-09 |
| 参数路由 | Nature News | `/nature/news` | ✅ 25 条，11-06 |

**仍不可用**：Arena `/arena/leaderboard`、字节点击 `/byteclicks`、The Verge AI hub、
极客公园 AI 栏目、21财经 AI 智讯、爱范儿 AI 分类（均 503）。

**RSSHub 缺失清单（独立清单反查结果）**：机器之心、新智元、VentureBeat、Synced、
The Information、Techmeme、Import AI、Simon Willison、Latent Space、Interconnects、
One Useful Thing、TLDR AI、The Batch、The Neuron、Ben's Bites、AI Supremacy、Mistral、xAI、
智谱、Kimi/Moonshot、MiniMax、文心/ERNIE、混元、阶跃、零一、商汤、讯飞、Stability AI、
Midjourney、Runway、ElevenLabs、Suno、Groq、Replicate、Together、OpenRouter、vLLM、
llama.cpp、Papers with Code、OpenReview、Semantic Scholar、Stanford HAI、MIT CSAIL、BAIR、LessWrong。

**结论**：RSSHub 当前镜像下，AI 相关可抓取源约 **60–70 个**（含参数路由与可选项）；
纯 AI 垂直/官方约 **45–50 个**；原 54 清单的实质缺口约 13–15 个。

### 4.7 六路并集扫描最终清单（2026-11-09）

方法：以镜像 registry 的 **2087 个命名空间**为全集，六种方法**并联取并集**（关键词/词表、
Radar 规则、维护者共现、分类人工遍历、外部新闻域名反查、路由参数探测）→ 148 个候选 →
人工复核 → 本地逐条实测。**不是层层过滤，而是分别查找后合并。**

**六路各自贡献（并集前）**

| 方法 | 候选数 | 仅此法发现 | 主要作用 |
|---|---|---|---|
| M1 关键词/词表 | 87 | 22 | 主力召回；独有项含大学/财经等噪声，需人工剔除 |
| M2 Radar 规则 | 42 | 0 | 佐证与交叉验证（与 M1 高度重叠） |
| M3 维护者共现 | 85 | 0 | 仅作排序信号，噪声大，不单独产出 |
| M4 分类人工遍历 | 26 | 2 | 补关键词盲区（如 ASML、半导体协会、Engadget） |
| M5 外部新闻域名反查 | 54 | 15 | 发现通用媒体与“RSSHub 缺失”的 AI 媒体（unite.ai 等） |
| M6 路由参数探测 | 7 | 4 | 发现通用媒体的 AI 栏目（NYT/Nature/Bloomberg/WIRED） |

合并后共 148 个候选，人工复核 + 实测后得到下述分级清单。

**一级：AI 垂直/官方（核心，✅ 实测可用）**

| 源 | 路由 | 状态 |
|---|---|---|
| OpenAI | `/openai/news` | ✅ |
| Anthropic | `/anthropic/news` | ✅ |
| Claude Blog | `/claude/blog` | ✅ |
| Claude Code Changelog | `/claude/code/changelog` | ✅ |
| DeepSeek | `/deepseek/news`、`/deepseek/changelog/zh` | ✅ |
| HuggingFace | `/huggingface/blog`、`/daily-papers` | ✅ |
| Ollama | `/ollama/blog` | ✅ |
| Qwen Research | `/qwen/blog` | ✅ |
| Cohere | `/cohere/blog` | ✅ |
| LangChain | `/langchain/blog` | ✅ |
| Kiro | `/kiro/blog` | ✅ |
| Inception Labs | `/inceptionlabs/blog` | ✅ |
| Thinking Machines | `/thinkingmachines/news` | ✅ |
| Transformer Circuits | `/transformer-circuits` | ✅ |
| The Decoder | `/the-decoder/news` | ✅ |
| Cool Papers | `/papers/category/arxiv/cs.AI` | ✅ |
| MIT Technology Review AI | `/technologyreview/artificial-intelligence` | ✅ |
| NVIDIA Newsroom | `/nvidia/newsroom` | ✅ |
| NVIDIA Developer | `/nvidia/developer` | ✅ |
| Google Research | `/google/research` | ✅ |
| 智源 BAAI | `/baai/hub` | ✅ |
| DeepLearning.AI | `/deeplearning/the-batch` | ✅ |
| BFL AI | `/bfl/announcements` | ✅ |
| Cursor | `/cursor/blog` | ✅ |
| Cline | `/cline/blog` | ✅ |
| Coze | `/coze/store/project` | ✅ |
| Civitai | `/civitai/models` | ✅ |
| Chub | `/chub/characters` | ✅ |
| Podwise | `/podwise/explore` | ✅ |
| Zread | `/zread/trending` | ✅ |
| Forward Future | `/forwardfuture/daily` | ✅ |
| Augment Code | `/augmentcode/blog` | ✅ |
| DeepL | `/deepl/blog/en` | ✅ |
| Mathpix | `/mathpix/blog` | ✅ |
| AI工具集 | `/ai-bot/daily-ai-news` | ✅ |
| AIbase | `/aibase/news` | ✅ |
| 极术社区 | `/aijishu/channel/ai` | ✅ |
| 量子位 | `/qbitai/category/资讯` | ✅ |

**陈旧/失败（保留观察或移除）**：Meta AI（2026-07）、Windsurf（2026-05）、The Gradient（2026-02）、
MIT HAN Lab（2026-02）、Google PAIR（2025-09）、AI 博客（2025-12）、AI财经社（2023-03）；
Arena、Perplexity、字节点击、ASML、21财经 AI 智讯（均 503）；品玩/安全内参（404）。

**二级：泛科技媒体里的 AI 栏目（参数路由，实测可用）**

| 源 | 路由 | 状态 |
|---|---|---|
| IT之家 AI 标签 | `/ithome/tag/AI` | ✅ |
| 量子位 大模型标签 | `/qbitai/tag/大模型` | ✅（10-20） |
| 少数派 AI 标签 | `/sspai/tag/AI` | ✅ |
| NYT Technology | `/nytimes/rss/Technology` | ✅ |
| Nature News | `/nature/news` | ✅ |
| WIRED AI | `/wired/tag/artificial-intelligence` | ✅ |
| Bloomberg Technology | `/bloomberg/technology` | ✅ |

**三级：泛科技/产业媒体（AI 报道占比高，按需接入）**
36Kr、极客公园、虎嗅、钛媒体、雷峰网、爱范儿、InfoQ、澎湃、cnBeta、OFweek、Engadget、
Scientific American、TrendForce、半导体协会、TechFlow、开源中国、掘金 AI 编程、快科技、
蓝点网、数字尾巴、MIT 科技评论中文、华尔街见闻。

**RSSHub 缺失（外部域名反查发现，只能官网直抓/公众号）**：unite.ai、alphaxiv.org、
quasa.io、ajudaily.com、mezha.net，以及机器之心、新智元、VentureBeat、Synced、The Information、
Import AI、Simon Willison 等（详见 4.6）。

### 4.8 C3 定稿清单（v1.0，2026-11-09）

> **状态：已定稿（v1.0，2026-11-09）**。4.1–4.4 仅作历史盘点留档，实现时以本节为准。
> 规模：**66 个源** = 一级 38 + 二级 6 + 三级 22。
> 验证环境：本地部署镜像 `ghcr.io/diygod/rsshub:latest`（端口 1205），66 个源（38+6+22）全部逐条实测。

**一级：AI 垂直/官方（38 个，实测可用）**

| 源 | 路由 | 实测 |
|---|---|---|
| OpenAI | `/openai/news` | ✅ |
| Anthropic | `/anthropic/news` | ✅ |
| Claude Blog | `/claude/blog` | ✅ |
| Claude Code Changelog | `/claude/code/changelog` | ✅ |
| DeepMind | `/deepmind/blog` | ⚠️ 路由超时（C1 原生 RSS 覆盖） |
| DeepSeek | `/deepseek/news`、`/deepseek/changelog/zh` | ✅ |
| HuggingFace | `/huggingface/blog`、`/daily-papers`、`/models/:group` | ✅ |
| Ollama | `/ollama/blog` | ✅ |
| Qwen Research | `/qwen/blog` | ✅ |
| Cohere | `/cohere/blog` | ✅ |
| LangChain | `/langchain/blog` | ✅ |
| Kiro | `/kiro/blog` | ✅ |
| Inception Labs | `/inceptionlabs/blog` | ✅ |
| Thinking Machines | `/thinkingmachines/news` | ✅ |
| Transformer Circuits | `/transformer-circuits` | ✅ |
| The Decoder | `/the-decoder/news` | ✅ |
| Cool Papers | `/papers/category/arxiv/cs.AI` | ✅ |
| MIT Technology Review AI | `/technologyreview/artificial-intelligence` | ✅ |
| NVIDIA Newsroom | `/nvidia/newsroom` | ✅ |
| NVIDIA Developer | `/nvidia/developer` | ✅ |
| 智源 BAAI | `/baai/hub` | ✅ |
| DeepLearning.AI | `/deeplearning/the-batch` | ✅ |
| BFL AI | `/bfl/announcements` | ✅ |
| Cursor | `/cursor/blog` | ✅ |
| Cline | `/cline/blog` | ✅ |
| Coze | `/coze/store/project` | ✅ |
| Civitai | `/civitai/models` | ✅ |
| Chub | `/chub/characters` | ✅ |
| Podwise | `/podwise/explore` | ✅（无日期） |
| Zread | `/zread/trending` | ✅（无日期） |
| Forward Future | `/forwardfuture/daily` | ✅ |
| Augment Code | `/augmentcode/blog` | ✅ |
| DeepL | `/deepl/blog/en` | ✅ |
| Mathpix | `/mathpix/blog` | ✅ |
| AI工具集 | `/ai-bot/daily-ai-news` | ✅ |
| AIbase | `/aibase/news` | ✅ |
| 极术社区 | `/aijishu/channel/ai` | ✅ |
| 量子位 | `/qbitai/category/资讯` | ✅ |
| Google Research | `/google/research` | ✅ |
| PixelsTech | `/pixelstech/feed/ai` | ✅ |

**二级：泛科技媒体的 AI 栏目（6 个参数路由，实测可用）**

| 源 | 路由 | 实测 |
|---|---|---|
| IT之家 AI 标签 | `/ithome/tag/AI` | ✅ |
| 少数派 AI 标签 | `/sspai/tag/AI` | ✅ |
| NYT Technology | `/nytimes/rss/Technology` | ✅ |
| Nature News | `/nature/news` | ✅ |
| WIRED AI | `/wired/tag/artificial-intelligence` | ✅ |
| Bloomberg Technology | `/bloomberg/technology` | ✅ |

**三级：泛科技/产业（22 个，可选）**

36Kr、极客公园、虎嗅、钛媒体、雷峰网、爱范儿、InfoQ、澎湃、cnBeta、OFweek、Engadget、
Scientific American、TrendForce、半导体协会、开源中国、掘金（AI 编程）、快科技、蓝点网、
数字尾巴、MIT 科技评论中文、Benedict Evans、Gwern。

**排除（不再接入）**：
- 陈旧（7）：AI 博客、AI 财经社、得物技术、Meta AI、MIT HAN Lab、The Gradient、Google PAIR；
- 不可用（7）：21财经、Arena、ASML、字节点击、Perplexity、品玩、安全内参；
- 噪声（70）：高校/研究机构 17、财经/咨询 13、政府/官媒/综合新闻 17、区块链 4、技术工具社区 8、生活体育购物 11；
- **暂缓（学术机构公告类）**：CAAI（中国人工智能学会）、国科大 AI（中国科学院大学人工智能学院）——内容以会议/招生/竞赛通知为主，新闻性弱，先不接入。

**验证状态说明**：66 个保留源全部在本地镜像实测过（HTTP 200 + 条目数 + 最新日期）。

**定稿变更记录**：

- 相对早期 54 清单，新增 14 个（The Decoder、NVIDIA×2、Claude Blog、Claude Code Changelog、
  Qwen Research、Cool Papers、MIT TR AI、Inception Labs、Thinking Machines、Forward Future、
  Transformer Circuits、Augment Code、DeepL、Mathpix 等）；
- 移除：CAAI、国科大 AI（学术机构公告类，暂缓）；
- 暂缓/排除：陈旧 7、不可用 7、噪声 70、三级可选 22（三级保留在清单内，但接入优先级最低）；
- DeepMind 路由超时但保留（C1 原生 RSS 覆盖）；Podwise/Zread 无发布时间。
例外：DeepMind 路由超时（C1 原生覆盖）；Podwise/Zread 无发布时间；CAAI/国科大以学会/学院通知为主。

## 5. C4 网页抓取通道（定稿：7 个 SSR 源，v1.0）

| 分组 | 源 | 采集内容 |
|---|---|---|
| 厂商官网 | DeepSeek | 官方新闻、更新日志 |
| 厂商官网 | 智谱 | 官网新闻/changelog |
| 厂商官网 | Kimi（月之暗面） | 官网新闻/changelog |
| 厂商官网 | MiniMax | 官网新闻/changelog |
| 厂商官网 | 字节 Seed | 官网新闻/研究发布 |
| 厂商官网 | 百度文心 | 官网新闻/changelog |
| 厂商官网 | 腾讯混元 | 官网新闻/changelog |
| 厂商官网 | 阶跃星辰 | 官网新闻/changelog |
| 厂商官网 | 零一万物 | 官网新闻/changelog |
| 厂商官网 | 商汤 | 官网新闻/changelog |
| 厂商官网 | 讯飞 | 官网新闻/changelog |
| 政策公告 | 工信部 / 网信办等 | AI 政策公告 |
| 社区频道 | Telegram 公开频道 | AI 团队/媒体快讯 |

### 5.0 C4 定稿清单（v1.0，2026-11-09）

**第一版接入（7 个，全部 SSR/静态 HTML，可提取标题/链接/日期）**

| 源 | 抓取地址 | 实测 |
|---|---|---|
| DeepSeek 官方新闻 | `/zh-cn/updates`（sitemap 动态发现 + changelog） | ✅ 已实现 |
| 智谱新闻 | `zhipuai.cn/zh/news` | ✅ 17 条 |
| MiniMax 新闻 | `minimax.cn/news` | ✅ 8 条，2026-11-09 |
| 字节 Seed 研究/博客 | `seed.bytedance.com/research`、`/blog_list/foundation` | ✅ 7 条 |
| 商汤新闻 | `sensetime.com/cn/news` | ✅ 12 条，2026-11-04 |
| 工信部 | `miit.gov.cn`（政策文件/新闻栏目） | ✅ 需细化栏目 URL |
| 网信办 | `cac.gov.cn`（政策法规/网信政务栏目） | ✅ 需细化栏目 URL |

**待验证（1）**：零一万物 `lingyiwanwu.com/newsroom.html`（有链接、无日期）。

**二期/暂不接入（5，SPA 需浏览器渲染）**：Kimi、文心、腾讯混元、阶跃星辰、讯飞。
可选用 Playwright 渲染（GHA 免费）或放弃、靠公众号/媒体覆盖。

**Telegram**：待选定可信 AI 频道后接入；`@OpenAI` 为山寨号，不可用。

### 5.1 C4 实测结果（2026-11-09）

**可抓（SSR/静态 HTML，能提取标题+链接+日期）**

| 源 | 抓取地址 | 实测样本 |
|---|---|---|
| DeepSeek 更新日志 | `/zh-cn/updates` | ✅ 已实现（sitemap 动态发现 + changelog） |
| 智谱新闻 | `zhipuai.cn/zh/news` | ✅ 17 条，可提取标题/日期/链接 |
| MiniMax 新闻 | `minimax.cn/news` | ✅ 8 条，最新 2026-11-09 |
| 字节 Seed 研究/博客 | `seed.bytedance.com/research`、`/blog_list/foundation` | ✅ 7 条，最新 2026-08-05 |
| 商汤新闻 | `sensetime.com/cn/news` | ✅ 12 条，最新 2026-11-04 |
| 工信部 | `miit.gov.cn` | ✅ SSR 有日期，需细化到“政策文件/新闻”栏目 |
| 网信办 | `cac.gov.cn` | ✅ SSR 有日期，需细化到“政策法规/网信政务”栏目 |
| 零一万物新闻室 | `lingyiwanwu.com/newsroom.html` | ⚠️ 有 9 条链接、无日期，待验证 |

**不可抓（SPA/JS 渲染，纯 HTTP 拿不到列表）**

| 源 | 现状 | 后续方案 |
|---|---|---|
| Kimi/月之暗面 | SPA，正文仅 928 字符 | 浏览器渲染（puppeteer）或找内部 API |
| 文心一言 | SPA，无内容 | 同上 |
| 腾讯混元 | SPA，无内容 | 同上 |
| 阶跃星辰 | SPA，无内容 | 同上 |
| 讯飞新闻 | SPA，无内容 | 同上 |

**Telegram**：`t.me/s/<channel>` 可解析（200），但需人工选定可信 AI 频道；示例 `@OpenAI` 是山寨号（1568 订阅、内容为 “Free AI credits”），不可用。

**SPA 进一步探测（2026-11-09）**

| 探测项 | 结果 |
|---|---|
| 内部 API | 腾讯混元页面暴露 `/api/new-portal/*`（登录/配置类），未发现 news 列表接口；其余 4 家初始 HTML/JS 未见 API 路径 |
| 文章页直抓 | Kimi/阶跃/文心/混元/讯飞的 sitemap 基本不含新闻 URL；Google News `site:` 查询有结果但链接是 Google 跳转（返回同意页 600KB），无法直接解析；Bing `site:` 结果不可靠 |
| 文档站替代 | Kimi 开放平台文档（SSR，261KB，2026-10-23）、阶跃开放平台文档（SSR，112KB，2026-11-05）可抓，但内容是 API 文档非新闻；混元/百度千帆文档 JS 渲染仅 864/2831 字节正文；讯飞文档 403 |
| 结论 | 5 家 SPA 要么用 Playwright 渲染（GHA 免费，二期），要么放弃，靠公众号/媒体覆盖 |

**结论**：C4 第一版先接 SSR 可抓的 6–7 个（DeepSeek/智谱/MiniMax/字节 Seed/商汤/工信部/网信办）；
SPA 的 5 家（Kimi/文心/混元/阶跃/讯飞）需要浏览器渲染，作为二期；
Telegram 待选定频道；政策公告需细化到具体栏目 URL。

## 6. C5 wechat2rss 通道（0 成本，17 个 AI 相关号）

> **2026-11-08 逐个实测**：19/19 全部 HTTP 200，均为 RSS 2.0 且每条带 `content:encoded` 全文（无需再抓文章页）。
> 单 feed 体积 0.7–4.7 MB（含全文），落盘时只保留摘要（≤500 字）与元数据，避免仓库膨胀。

| 状态 | 公众号 |
|---|---|
| ✅ 活跃（≤7 天，16 个） | 机器之心、新智元、量子位、极客公园、PaperWeekly、我爱计算机视觉、夕小瑶科技说、字节跳动技术团队、阿里技术、阿里云开发者、阿里巴巴中间件、腾讯技术工程、美团技术团队、小米技术、爱奇艺技术产品团队、得物技术 |
| ⚠️ 偏旧（19 天，1 个） | 哔哩哔哩技术（最新 2026-10-20） |
| ❌ 陈旧（建议移除，2 个） | 浅黑科技（最新 2026-06-01）、机器学习初学者（最新 2025-08-16） |


| 公众号 | feed id | 内容 |
|---|---|---|
| 机器之心 | `51e92aad2728acdd1fda7314be32b16639353001` | AI 媒体（全文） |
| 新智元 | `ede30346413ea70dbef5d485ea5cbb95cca446e7` | AI 媒体（全文） |
| 量子位 | `7131b577c61365cb47e81000738c10d872685908` | AI 媒体（全文） |
| 极客公园 | `1a5aec98e71c707c8ca092bc2c255b9d4bac477d` | 科技媒体 |
| PaperWeekly | `3be891c2f4e526629ab055a297cc2cd6c1f0a563` | 论文解读 |
| 我爱计算机视觉 | `b81ffcfff1107b5265cd7e39de610dc7ca72caf4` | CV 方向 |
| 夕小瑶科技说 | `a1cd365aa14ed7d64cabfc8aa086da40ecaba34d` | AI 行业评论 |
| 字节跳动技术团队 | `4025ea55575daf8bfd8227e68b28d9638b073267` | 技术博客（AI 偶发） |
| 阿里技术 | `6e1f9b775f7a5841ac1a94310f0478b45a02ec01` | 技术博客 |
| 阿里云开发者 | `c74ed6db00cfbf16f2a048a165b4453f982681f0` | 技术博客 |
| 阿里巴巴中间件 | `ede34e7a9aff5e93ad159ebfd27075710a963f54` | 技术博客 |
| 腾讯技术工程 | `9685937b45fe9c7a526dbc32e4f24ba879a65b9a` | 技术博客 |
| 美团技术团队 | `eb4d04149424a874693a51c6fdda0dba8673f5e4` | 技术博客 |
| 小米技术 | `20bc9c3251b3c4f73d3b53aa1f1ab853d05d4cbc` | 技术博客 |
| 哔哩哔哩技术 | `434235d4815fdb8447ff3127fc053ceb8b3aada6` | 技术博客 |
| 爱奇艺技术产品团队 | `16a4ec12a83a52e1f6e941bce030a4d64ee26c47` | 技术博客 |
| 得物技术 | `f3a42bd249ec6e8834ae761d8d0f85a949950944` | 技术博客 |

> **已移除（陈旧）**：机器学习初学者（2025-08 后停更）、浅黑科技（2026-06 后停更）。
>
> 说明：厂商官方 AI 号（DeepSeek、OpenAI、Anthropic、Kimi、智谱、通义、豆包、混元、文心、商汤、旷视、讯飞、阶跃、零一、智源）**不在免费名单**，需要 C6 私有桥（登录态/可选，另立项）。

## 7. 统一数据结构

```json
{
  "channel": "c1_rss|c2_atom|c3_rsshub|c4_web|c5_wechat2rss",
  "source": "anthropic_news",
  "source_label": "Anthropic News",
  "category": "official|news|research|community|podcast|video",
  "title": "……",
  "link": "https://……",
  "published_at": "2026-11-08T00:00:00+00:00",
  "summary": "……（≤500 字）",
  "full_text": "……（采集时一次性抓到的正文；feed 自带 content:encoded 则直接用，否则采集时抓文章页抽取）",
  "full_text_status": "ok|failed|truncated",
  "extra": {"score": 120, "comments": 14}
}
```

落盘：

```
data/cache/
  c1_rss/{source}.json
  c2_atom/{source}.json
  c3_rsshub/{source}.json
  c4_web/{source}.json
  c5_wechat2rss/{account}.json
```

## 8. 数据留存与传递（一次性全量抓取，零二次抓取）

**原则：采集阶段一次性把数据抓完整（含正文），之后所有分类/整合/归纳/加工只读采集产物，不再访问网络、不做二次抓取。**

### 8.1 采集阶段必须抓完整

- 每条 item 除元数据外，必须落 `full_text`：
  - feed 带 `content:encoded` / `content` → 直接使用（wechat2rss、部分官方博客）；
  - 只有摘要 → **采集时当场抓文章页**，用 readability 抽正文；
  - 抓取失败 → 标记 `full_text_status=failed`，**不重试第二次抓取**，下游按摘要降级处理。
- 这样正文在采集时一次拿到；下游选材/写作不再按链接回源（可能已失效、被反爬、或需登录）。

### 8.2 全文存哪里（不能进 git）

体积估算：约 1450 条/天 × 5–10KB 正文 ≈ **7–15MB/天**，年增 2.5–5GB。全存 git 会让 clone / `actions/checkout` 不可用。

| 方案 | 成本 | 说明 |
|---|---|---|
| **采集作为 daily.yml 的 Job-0，全文走 artifact 传给 Stage 1**（推荐） | 0 元 | 同一次 workflow run 内传递，零二次抓取；artifact 免费保留 90 天，可审计；git 不存全文 |
| 采集独立 workflow + artifact，Stage 1 按 run-id 下载 | 0 元 | 需要 `actions:read` token；多一次跨 run 依赖 |
| 对象存储（R2/S3）做长期归档 | 低（可选） | 只有需要长期保存全文时才用 |

### 8.3 各层分工

| 层 | 内容 | 位置 | 用途 |
|---|---|---|---|
| 采集产物 | 元数据 + 摘要 + **全文** | GHA artifact（90 天） | Stage 1 唯一输入，零网络 |
| 摘要快照 | 元数据 + 摘要（≤500 字） | `data/cache/`（git） | 审计/回看，不作为素材来源 |
| 节目产物 | brief / 脚本 / 音频 | 现有流程 | 不变 |

### 8.4 Stage 1 的变化

- 不再自己抓 RSS/文章页，改为读取采集产物（artifact 解压后的 JSON）；
- 去重、聚类、打分、选材全部基于本地 `full_text`；
- 采集失败/全文缺失的条目按摘要降级，仍可参与聚类打分。

## 9. 调度与失败处理

- 每天 16:00 UTC（北京 00:00）执行一次；
- C3 需要临时 RSSHub 容器；起不来则 C3 源跳过，其余通道照常；
- 每源 try/except + 重试；单源失败保留旧快照；
- 单次 commit（`[skip ci]`），避免多通道并发提交冲突；
- 手动触发支持 `dry_run` 与单通道调试。

## 10. 容量预算

| 通道 | 估算条数/天 |
|---|---|
| C1 RSS | ~300 |
| C2 Atom | ~250 |
| C3 RSSHub | ~600 |
| C4 网页 | ~100 |
| C5 wechat2rss | ~200 |
| **合计** | **~1450 条/天（去重前）** |

- 快照约 1 MB/天；压缩进 git 约 0.3 MB/天（保留 7 天窗口）；
- 单轮耗时 3–6 分钟（含 RSSHub 镜像启动）。

## 11. 去重（只在 Stage 1 做）

- 通道之间不互相排除（重复源允许存在）；
- 采集端只做 URL 规范化、同源内 URL 去重、时效过滤、条数上限；
- Stage 1 负责 URL 去重、跨集去重、语义去重、DBSCAN 聚类。

## 12. 跨期去重（重复新闻过滤）

目标：一条新闻第一天被选中播报过，第 N 天又被其他源报出来（同一事件、不同链接），必须识别为“已覆盖”并过滤，不再选中。

### 12.1 比较用的历史数据从哪来

**用历史采集 artifact，不重新抓取。** 每天采集 job 会把“全量新闻（含全文）”打成 artifact 保留（`retention-days` 设 30 天，≥ 去重窗口）。Stage 1 启动时：

1. 下载**今天**的 artifact（`fulltext-{date}.json.gz`）；
2. 通过 GitHub API 找到**最近 30 天**的采集 run，下载它们的 artifact；
3. 用这些历史 artifact 构建“历史新闻库”（title / link / published_at / summary / full_text）；
4. 叠加历史 `brief_{date}.json` / `episodes.json` 的“**已选中**”标记：把历史新闻库里的条目与当天 brief 的 stories 链接/title 关联，标出哪些被播报过。

这样比较依据来自真实抓取时的原文，不会因为文章后来 404/反爬/登录而失效，也不产生二次抓取。

### 12.2 判定顺序

```
今天的条目
  → 在历史新闻库中 URL 精确命中，且该历史条目“已选中”？   → 过滤（重复）
  → 语义相似度 ≥ 阈值（用 full_text），且历史条目“已选中”？
        → 无新进展（数字/日期/版本/动作无变化） → 过滤（重复报道）
        → 有新进展                              → 保留（事件更新）
  → 命中“历史出现过但未被选中”？ → 保留（可降权），不算已覆盖
  → 通过后进入当日聚类/打分/选材
```

### 12.3 阶段 1 的实现要点

- **artifact 保留期**：设 30 天（GitHub 上限 90 天），保证去重窗口内的历史都能下载；
- **下载方式**：`GITHUB_TOKEN`（`actions:read`）调用 REST API 列 run → 取 `fulltext-*` artifact → 下载 zip 解压；  
  每天约 30 × 2–5MB ≈ **60–150MB** 下载量，GHA 网络可承受；
- **历史新闻库规模**：30 天 × ~1450 条 ≈ **4.3 万条**，网址/标题去重 + TF-IDF 相似度可接受；装 embedding 时可选抽样；
- **兜底**：如果某天 artifact 缺失/过期/run 失败，用 git 里的“覆盖账本”（`data/coverage/ledger.json`，只记已选中的故事摘要）兜底，保证不会完全失去跨期记忆。

### 12.4 覆盖账本（可选，git 里的小型长期记忆）

- 每期只记“已选中的故事”：代表标题、链接、实体、关键词、摘要；
- 永久保留，约 3MB/年；
- 作用：artifact 过期后的兜底、快速查询、审计；
- 不存全文（全文只在 artifact 里）。

### 12.5 新进展保护

同一条新闻第 N 天再出现，可能是新进展（发布→开源、融资金额更新、新版本号、监管动作）。语义命中后检查是否出现新数字/新日期/新版本/新动作词，有则保留，避免误杀。

### 12.6 注意

- 比较只发生在 Stage 1 本地，输入是下载下来的历史 artifact，不发网络请求；
- “已选中”的范围默认 main + supporting + quick 都算（覆盖过的都算），可配置；
- 历史上“出现过但没选”的条目不算已覆盖，不因此过滤（**确认：保留**）。

### 12.7 已确认参数

| 参数 | 值 |
|---|---|
| 去重窗口 | **30 天** |
| 出现过但未选中 | **保留**（不过滤） |
| 覆盖账本 | **要**（git 里，长期兜底） |
| artifact 保留 | **30 天** |
