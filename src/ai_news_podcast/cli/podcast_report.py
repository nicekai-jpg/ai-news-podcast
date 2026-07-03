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
    return f"""你是专业的科技媒体编辑，需要根据以下今日的 AI 和科技新闻素材，写一份专业的「科技新闻日报」。

## 格式与结构要求
请严格使用 Markdown 格式输出，包含以下几个部分：

1. **大标题**：如 `# 🌍 科技新闻日报 | {date_str}`
2. **导语**：简短概括今天最核心的科技趋势（1-2句话）。
3. **分类新闻**：将新闻素材按主题分类（如：AI前沿、大公司动态、产品与开源、行业数据等）。
4. **新闻条目**：每条新闻使用列表格式，加粗小标题，然后是一段精炼的中文总结。
5. **AI 洞察小结**（可选）：在文末用引用的方式（`> `）输出一段你对今天科技动态的总评析。

## 写作要求
- **全部用中文**：所有英文内容必须翻译为中文，保持用词专业、准确。
- **融合提炼**：不要机械地罗列每条素材，如果有相关的素材可以合并写。
- **客观中立**：保持科技新闻的客观感，语言精炼。
- **字数**：总字数控制在 800 - 1500 字左右。

## 今日素材
{material}

请直接输出 Markdown 文本，不要在开头和结尾带多余的解释，不要带 ```markdown 这样的代码块标记。"""


class ReportCommand(AsyncCommand):
    """Generate daily tech news report."""

    description = "生成科技新闻日报 (Stage 3b)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
        parser.add_argument("--outdir", default="data/reports")

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
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

        if not report_md:
            report_md = f"# 🌍 科技新闻日报 | {date_display}\n\n> 由于大模型服务暂时不可用，以下是由系统自动整理的新闻速览：\n\n"
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
