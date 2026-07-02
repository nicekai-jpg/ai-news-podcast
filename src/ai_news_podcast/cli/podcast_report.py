import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.pipeline.llm_client import call_llm
from ai_news_podcast.pipeline.material import build_material_text
from ai_news_podcast.utils import read_yaml

log = logging.getLogger("daily_report")


def build_report_prompt(brief: dict, date_str: str) -> str:
    material = build_material_text(brief, max_stories=5, strategy="pure_score")

    return f"""你是专业的科技媒体编辑，需要根据以下今日的 AI 和科技新闻素材，写一份专业的「科技新闻日报」。

## 格式与结构要求
请严格使用 Markdown 格式输出，包含以下几个部分：

1. **大标题**：如 `# 🌍 科技新闻日报 | {date_str}`
2. **导语**：简短概括今天最核心的科技趋势（1-2句话）。
3. **分类新闻**：将新闻素材按主题分类（如：AI前沿、大公司动态、产品与开源、行业数据等）。每个分类下使用子标题（如 `## 💡 AI 前沿与政策风向`）。
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


def generate_report(
    brief: dict,
    *,
    date_str: str,
    report_id: str,
    outdir: Path,
    llm_cfg: dict,
) -> Path:
    """Generate and save the daily tech news report.

    Returns
    -------
    Path: Path to the generated report file.
    """
    prompt = build_report_prompt(brief, date_str)
    log.info("Generated prompt string length: %d characters", len(prompt))

    report_md = call_llm(prompt, llm_cfg)

    if not report_md:
        log.warning("Failed to generate report from LLM. Generating a basic fallback report.")
        report_md = f"# 🌍 科技新闻日报 | {date_str}\n\n> 由于大模型服务暂时不可用，以下是由系统自动整理的新闻速览：\n\n"

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

    # Clean up markdown markers
    if report_md.startswith("```markdown"):
        report_md = report_md[11:].strip()
    if report_md.endswith("```"):
        report_md = report_md[:-3].strip()

    # Save report
    report_path = outdir / f"daily_report_{report_id}.md"
    report_path.write_text(report_md, encoding="utf-8")

    log.info("✨ Daily report generated successfully! Saved to: %s", report_path)
    return report_path


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="生成科技新闻日报 (Stage 3b)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD (默认今天)")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)

    date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    report_id = date_str
    date_display = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y年%m月%d日")

    # 读取已有的 brief
    brief_path = root / "data" / "briefs" / f"brief_{date_str}.json"
    if not brief_path.exists():
        log.error("Brief not found: %s", brief_path)
        return 1

    brief = read_yaml(brief_path)
    if not brief.get("stories"):
        log.warning("No stories in brief. Aborting.")
        return 1

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 生成报告
    log.info("Stage 3b: Generating daily report for %s ...", date_str)
    llm_cfg = cfg.get("llm", {})
    generate_report(
        brief,
        date_str=date_display,
        report_id=report_id,
        outdir=outdir,
        llm_cfg=llm_cfg,
    )

    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
