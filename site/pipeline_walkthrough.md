# 🎙️ AI News Podcast 自动化新闻处理管线全景图解

本项目是一个全自动的 AI 前沿资讯整理与播音系统。以下是该系统从“原始资讯抓取”到“全球分发部署”的完整流程图解。

---

## 🎨 系统整体架构流水线 (Overall Architecture Pipeline)

以下是本项目从“资讯抓取”到“全网发布”的端到端整体架构与数据流向流程图：

```mermaid
graph TD
    subgraph "1. 输入配置 (Inputs)"
        Config_Sources[config/sources.yaml<br/>RSS新闻源]
        Config_System[config/config.yaml<br/>系统与打分参数]
    end

    subgraph "2. 核心资讯处理管线 (Core Pipeline)"
        Ingestion[RSS 抓取 & Trafilatura 正文提取]
        Deduplication[三层高精度去重:<br/>URL + Title + Jieba 关键词]
        Clustering[DBSCAN 密度聚类与代表新闻选定]
        Scoring[五维打分模型 & 角色与内容路由决策]
        Generation[LLM 双人对话剧本创作与文字日报生成]
    end

    subgraph "3. 语音合成与构建 (Audio Synthesis & Concat)"
        TTS_Select{TTS 选型映射决策}
        TTS_Local[Edge-TTS GHA 本地合成<br/>(标准旁白/普通文本)]
        TTS_ECS[Tencent Cloud 2C2G ECS<br/>ONNX 队列合成 (情感/中英混读)]
        FFmpeg_Conc[ffmpeg 无损拼接 & loudnorm 响度均衡]
    end

    subgraph "4. 部署与托管 (Distribution & Hosting)"
        Site_Build[静态页面 index.html & feed.xml 生成]
        Publish[GitHub Pages CDN 全球分发]
    end

    Config_Sources & Config_System --> Ingestion
    Ingestion --> Deduplication
    Deduplication --> Clustering
    Clustering --> Scoring
    Scoring --> Generation
    
    Generation -->|文字日报 & 旁白| TTS_Local
    Generation -->|情感/中英混读台词| TTS_Select
    TTS_Select -->|缓存未命中| TTS_ECS
    TTS_Select -->|缓存命中| FFmpeg_Conc
    TTS_Local & TTS_ECS --> FFmpeg_Conc
    FFmpeg_Conc --> Site_Build
    Site_Build --> Publish
```

## 🛠️ 技术实现流程详解 (Flowcharts)

### 1. 资讯抓取与正文提取 (Fetch & Ingestion)

系统每天定时被触发，首先进入资讯获取模块（`fetcher.py`），解析源配置文件并抓取正文：

```mermaid
graph TD
    A[config/sources.yaml] -->|并发抓取| B(RSS Feed 抓取)
    B -->|URL 提取| C(URL 规范化)
    C -->|哈希计算| D(生成 sha256 唯一ID)
    C -->|双引擎正文提取| E{优先使用 trafilatura}
    E -->|提取成功且长文本| F[保留 1200-2000字核心段落]
    E -->|短文本或提取失败| G[回退使用 readability-lxml 提取]
    G --> F
    F --> H[生成 RawItem 列表]
```

---

### 2. 三层高精度去重 (Three-Layer Deduplication)

清洗抓取出的新闻池，防止重复话题在同一天里反复提及：

```mermaid
graph TD
    A[原始新闻池 RawItem 列表] --> B[第一层: 规范化 URL 绝对排重]
    B --> C[第二层: RapidFuzz 标题相似度比对]
    C -->|比对 48小时内标题 / 相似度大于等于 92%| D[判定为重复事件 - 过滤]
    C --> E[第三层: Jieba 中文分词提取关键词]
    E -->|比对关键词重合度大于等于 35% 且 标题相似大于等于 85%| D
    E --> F[获得精炼后的 RawItem 新闻池]
```

---

### 3. DBSCAN 智能聚类与代表选定 (Clustering)

把分散在不同媒体、不同视角的同类报道智能归为一个“事件簇”：

```mermaid
graph TD
    A[精炼后的 RawItem 新闻池] --> B[合并标题与摘要作为特征文本]
    B --> C[提取 TF-IDF 字符级别 N-Gram 矩阵]
    C --> D[计算文本余弦相似度矩阵 Cosine Distance]
    D --> E[传入 DBSCAN 密度聚类算法]
    E --> F{是否为噪声点/离散事件?}
    F -->|是| G[该新闻自成一个新闻簇]
    F -->|否| H[聚合相似报道归入同一个新闻簇 Cluster]
    G --> I[从每个簇中选择全文最丰富/最长的文章作为代表]
    H --> I
    I --> J[生成最终的 Cluster 列表]
```

---

### 4. 五维打分与角色分派 (Scoring & Role Assignment)

给所有筛选出来的事件簇进行多维度量化评估，决定其是否上播或刊登：

```mermaid
graph TB
    subgraph 评分维度 (各项 1-3分)
        A[<b>社会热度</b>: 报道的媒体源数量]
        B[<b>技术创新</b>: 含突破/新模型/开源等词]
        C[<b>信息丰富</b>: 信源长度与细节]
        D[<b>受众相关</b>: AI核心话题与中文支持]
        E[<b>信源权威</b>: 官方博客与学术论文权重高]
    end

    A & B & C & D & E --> F[<b>综合得分 (5-15分)</b>]
    F --> G{划分等级与角色}
    G -->|12-15分| H[🔴 <b>核心主打新闻</b>: 详细拆解剖析]
    G -->|8-11分| I[🟡 <b>支线支撑新闻</b>: 辅助技术印证]
    G -->|5-7分| J[🟢 <b>行业速报简讯</b>: 结尾一句话略过]
    G -->|小于5分| K[⚪ <b>噪音过滤 (Skip)</b>: 软文、广告、水文彻底废弃]
```

---

### 5. 双模同步生成与语音合成 (Script, TTS & Publish)

最后，将角色分配好的新闻事件，通过两条独立的支线进行文字和音频维度的二次创作与部署：

```mermaid
graph TD
    A[打分分配完毕的新闻树] --> B(<b>文字日报支线</b>)
    A --> C(<b>播客节目支线</b>)

    B --> D[LLM 媒体编辑风格撰写]
    D --> E[生成结构化科技日报 Markdown]
    E --> F[每日 reports 目录保存]

    C --> G[LLM 播客剧本双人对话体创作]
    G --> H[CosyVoice2 逐句零样本克隆 → 拼接 → BGM → loudnorm]
    H --> I[音轨小块并行并发 ➔ pydub 无缝拼接]
    I --> J[响度归一化 Loudness Normalization]
    J --> K[生成成品播客 MP3 音频]

    F & K --> L[自动生成 index.html 与 feed.xml 订阅源]
    L --> M[GitHub Actions 自动构建与部署发布]
    M --> N[🚀 线上 gh-pages 静态网站全球分发上线]
```

---

## 📌 两个板块的互补定位

* **科技日报 (Tech Report)**：是供用户 **阅读** 的文字载体，适合希望快速预览、浏览具体条目、或者通过点击链接查看原始报道出处的读者。
* **播客节目 (Podcast Episodes)**：是供用户 **聆听** 的音频载体，由两位 AI 主播用轻松的口语化对谈风格来播报，适合通勤、驾车、做家务等“闭眼收听”的场景。
* **双向联动**：我们在主页中将其进行了深度联动，点击任何卡片下的“阅读日报”都可以快速转跳到文字大本营，达成“声文并茂”的立体体验。
