"""Stage 3b CLI entry: podcast-report.

Generate daily tech news report from brief.
"""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_news_podcast.cli.base import AsyncCommand
from ai_news_podcast.config.models import AppConfig
from ai_news_podcast.pipeline.llm_client import call_llm
from ai_news_podcast.pipeline.material import build_material_text
from ai_news_podcast.utils import read_yaml


def build_report_prompt(brief: dict, date_str: str) -> str:
    """Build the LLM prompt for the daily tech news report."""

    material = build_material_text(brief, max_stories=5, strategy="pure_score")
    return f"""你是顶级科技智库与科技媒体的主编，需要根据以下今日的 AI 和科技新闻素材，撰写一份结构严谨、视角深刻的「科技新闻日报」。

## 格式与结构要求（必须严格遵守）
请严格使用标准的 Markdown 格式输出，包含以下结构：

1. **大标题**：必须为 `# ✨ 科技新闻日报 | {date_str}`
2. **核心导语**：
   - 必须以 `**导语**：` 开头，用 2-3 句话高度概括今日全球科技圈最核心的技术突破与行业演进主线。
3. **今日重磅解读**：
   - 使用二级标题 `## 🚀 重磅解读`。
   - 挑选 2 个最重要的事件进行深度拆解，包含：事件背景、核心突破点、对行业生态与用户的深刻影响。
4. **前沿情报与产品动向**：
   - 使用二级标题 `## ⚡ 前沿情报`。
   - 精炼总结其余 2-3 条短新闻，使用列表格式，包含加粗关键词和简析。
5. **AI 小编深度点评**：
   - 在文末必须包含一个引用块（以 `> ` 开头），写一段 150-250 字的专业总结，洞察今日所有新闻背后的底层商业逻辑与未来竞争局势。

## 写作风格与要求
- **深度洞察**：拒绝机械罗列，突出“为什么重要”与“背后逻辑”。
- **专业流畅**：使用准确、专业的科技与商业术语。
- **排版优雅**：灵活使用加粗、列表、引用块提升阅读体验。
- **字数**：控制在 1000 - 1800 字左右。
- **禁止思考过程**：绝对不要包含 `<think>` 等思考标记，不要在开头结尾带 ``` 代码块。

## 今日素材
{material}

请直接输出 Markdown 文本。"""


class ReportCommand(AsyncCommand):
    """Generate daily tech news report."""

    description = "生成科技新闻日报 (Stage 3b)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
        parser.add_argument("--outdir", default="data/reports")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        import re

        date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
        report_id = date_str
        date_display = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y年%m月%d日")

        brief_path = root / "data" / "briefs" / f"brief_{date_str}.json"
        if not brief_path.exists():
            print(f"Brief not found: {brief_path}")
            return 1

        brief = read_yaml(brief_path)
        if not brief.get("stories"):
            print("No stories in brief. Aborting.")
            return 1

        outdir = root / args.outdir
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"Stage 3b: Generating daily report for {date_str} ...")
        prompt = build_report_prompt(brief, date_display)
        report_md = call_llm(prompt, cfg.llm)

        if report_md:
            # Strip <think>...</think> if present
            report_md = re.sub(r"<think>.*?</think>", "", report_md, flags=re.DOTALL).strip()

        if not report_md:
            report_md = f"# ✨ 科技新闻日报 | {date_display}\n\n**导语**：由于大模型服务暂时不可用，以下是由系统自动整理的新闻速览：\n\n"
            stories = brief.get("stories", [])
            active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
            active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
            active = active[:5]
            for i, story in enumerate(active, 1):
                title = story.get("representative_title", "无标题")
                context = story.get("context", {})
                summaries = context.get("factual_summary", [])
                report_md += f"## {i}. {title}\n"
                for s in summaries:
                    report_md += f"- **{s}**\n"
                report_md += "\n"

        if report_md.startswith("```markdown"):
            report_md = report_md[11:].strip()
        if report_md.startswith("```"):
            report_md = report_md[3:].strip()
        if report_md.endswith("```"):
            report_md = report_md[:-3].strip()

        report_path = outdir / f"daily_report_{report_id}.md"
        report_path.write_text(report_md, encoding="utf-8")

        print(f"Daily report saved to: {report_path}")
        return 0


# Backward compatibility for tests
async def main() -> int:
    """Legacy entrypoint for tests."""
    import argparse
    import sys

    from ai_news_podcast.config.loader import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--outdir", default="data/reports")
    ap.add_argument("--date", default=None)
    args = ap.parse_args(sys.argv[1:])

    root = Path(__file__).resolve().parents[3]
    config_path = root / args.config
    cfg = load_config(config_path) if config_path.exists() else AppConfig()

    return await ReportCommand().execute_async(args, cfg, root)


def entrypoint() -> int:
    return ReportCommand().run()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
