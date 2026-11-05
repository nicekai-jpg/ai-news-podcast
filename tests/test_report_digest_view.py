"""report_digest_view 单测：日报 md → 结构化 report 块。"""

from ai_news_podcast.presentation.site_builder.report_digest_view import parse_report_markdown

STANDARD_MD = """# ✨ 科技新闻日报 | 2026年10月15日

**导语**：今日全球科技圈呈现"智能体规模化落地"的清晰主线。

---

## 🚀 重磅解读

### 一、NVIDIA 押注法国，智能体计算挺进欧洲纵深

**事件背景**：在欧盟加速推进 AI 主权战略的关键节点。

**核心突破点**：agentic computing 的产业级落地。

**对行业与用户的深远影响**：欧洲不再甘于仅做监管者。

### 二、万级智能体长时任务

**事件背景**：Third Moment 披露。

**核心突破点**：首个万级长时案例。

**行业影响**：从对话工具到虚拟员工。

---

## ⚡ 前沿情报

- **DevFest 2026 强势回归**：Google 宣布启动全球报名。
- **SentinelOne 提速千倍**：恶意软件分析缩至小时级。

---

> **AI 小编深度点评**：今日新闻共同勾勒出三段式跃迁。
"""


class TestParseStandard:
    def test_title_and_lead(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert r["title"].startswith("✨ 科技新闻日报")
        assert "智能体规模化落地" in r["lead"]

    def test_deep_dives_three_points(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert len(r["deep_dives"]) == 2
        first = r["deep_dives"][0]
        assert first["heading"].startswith("一、NVIDIA")
        assert first["background"].startswith("在欧盟")
        assert first["breakthrough"].startswith("agentic computing")
        assert first["impact"].startswith("欧洲不再")

    def test_loose_impact_label_variants(self) -> None:
        md = STANDARD_MD.replace("**对行业与用户的深远影响**", "**行业影响**")
        assert parse_report_markdown(md)["deep_dives"][0]["impact"].startswith("欧洲不再")

    def test_frontier_items(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert [i["title"] for i in r["frontier"]] == [
            "DevFest 2026 强势回归",
            "SentinelOne 提速千倍",
        ]

    def test_editor_verdict(self) -> None:
        assert parse_report_markdown(STANDARD_MD)["editor_verdict"].startswith("今日新闻共同")

    def test_char_count_and_minutes(self) -> None:
        r = parse_report_markdown(STANDARD_MD)
        assert r["char_count"] == len(STANDARD_MD)
        assert r["reading_minutes"] >= 1


class TestParseVariants:
    def test_missing_sections_fall_back_to_raw_paragraphs(self) -> None:
        md = "# 日报\n\n**导语**：一句话。\n\n## 别的栏目\n\n正文一段。\n"
        r = parse_report_markdown(md)
        assert r["deep_dives"] == [] and r["frontier"] == []
        assert any("正文一段" in p for p in r["raw_paragraphs"])

    def test_empty_returns_none(self) -> None:
        assert parse_report_markdown("") is None

    def test_lead_with_colon_inside_bold(self) -> None:
        """2026-11-05 起的写法：`**导语：** 正文`（冒号在粗体内）。"""
        md = "# ✨ 科技新闻日报 | 2026年11月5日\n\n**导语：** 今日主线是长周期执行。\n\n## 别的栏目\n\n正文。\n"
        r = parse_report_markdown(md)
        assert r["lead"] == "今日主线是长周期执行。"

    def test_radar_section_is_not_news(self) -> None:
        md = STANDARD_MD + "\n## 📡 项目雷达 | 2026年10月15日\n\n### 🥇 主推 [o/r](https://x)\n"
        r = parse_report_markdown(md)
        assert all("项目雷达" not in d["heading"] for d in r["deep_dives"])


class TestNewBulletFormat:
    """2026-10-16 之后日报改用 "- **事件背景**：…" 单行 bullet，必须同样解析成结构化卡片。"""

    MD = """# ✨ 科技新闻日报 | 2026年10月27日

**导语**：今日焦点是无障碍交互与推理效率。

## 🚀 重磅解读

### 1. DeepMind 推出 SL2T：AI 无障碍走向产品能力

- **事件背景**：语音助手大多围绕“说出来”设计，聋人主要通过手语交流。
- **核心突破点**：DeepMind 发布 **SL2T** 模型，把能力封装成产品功能。
- **行业与用户影响**：降低听障人士在公共服务中的沟通成本。

### 2. MOONSHOT 多目标剪枝：单位算力产出

- **事件背景**：模型参数膨胀带来推理成本上升。
- **核心突破点**：Kimi 发布 **MOONSHOT 框架**，同时压缩视觉与语言模型。
- **行业与用户影响**：云端推理价格与端侧部署门槛将下降。

## ⚡ 前沿情报

- **GHARP 推动实时 3D 头像**：Apple 发布 GHARP，实时生成高斯头部动画。

> 今日主线是 AI 竞争从“更大”转向“更有效率”。
"""

    def test_bullet_points_parsed(self) -> None:
        report = parse_report_markdown(self.MD)
        assert report is not None
        assert report["lead"].startswith("今日焦点")
        assert [dive["heading"] for dive in report["deep_dives"]] == [
            "一、DeepMind 推出 SL2T：AI 无障碍走向产品能力",
            "二、MOONSHOT 多目标剪枝：单位算力产出",
        ]
        first = report["deep_dives"][0]
        assert first["background"].startswith("语音助手大多")
        assert "SL2T" in first["breakthrough"]
        assert first["impact"].startswith("降低听障人士")
        assert "**" not in first["breakthrough"]

    def test_frontier_and_verdict_parsed(self) -> None:
        report = parse_report_markdown(self.MD)
        assert report is not None
        assert report["frontier"] == [
            {
                "title": "GHARP 推动实时 3D 头像",
                "summary": "Apple 发布 GHARP，实时生成高斯头部动画。",
            }
        ]
        assert report["editor_verdict"].startswith("今日主线")
        assert next(item["title"] for item in report["toc"]).startswith("一、DeepMind")


class TestProseFallback:
    """没有结构化要点的老日报：段落挂在 deep dive 的 raw_sections 上，避免卡片只剩标题。"""

    MD = """# 日报

## 🚀 重磅解读

### 一、某个重要事件

这一段是没有小标题的正文，解释事件背景与影响。

第二段继续补充行业影响。
"""

    def test_prose_attached_to_deep_dive(self) -> None:
        report = parse_report_markdown(self.MD)
        assert report is not None
        dive = report["deep_dives"][0]
        assert dive["background"] is None
        assert dive["raw_sections"] == [
            "这一段是没有小标题的正文，解释事件背景与影响。",
            "第二段继续补充行业影响。",
        ]
