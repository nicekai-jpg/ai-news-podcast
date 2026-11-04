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

    def test_radar_section_is_not_news(self) -> None:
        md = STANDARD_MD + "\n## 📡 项目雷达 | 2026年10月15日\n\n### 🥇 主推 [o/r](https://x)\n"
        r = parse_report_markdown(md)
        assert all("项目雷达" not in d["heading"] for d in r["deep_dives"])
