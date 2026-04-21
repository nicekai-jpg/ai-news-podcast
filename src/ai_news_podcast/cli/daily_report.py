import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process
from ai_news_podcast.utils import load_sources, read_yaml

log = logging.getLogger("daily_report")


def build_report_prompt(brief: dict, date_str: str) -> str:
    stories = brief.get("stories", [])
    active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
    active.sort(key=lambda s: s.get("total_score", 0), reverse=True)

    # 恢复取前 15 条新闻（原生 API 配合大 context 可以扛住）
    active = active[:15]

    material = ""
    for i, story in enumerate(active, 1):
        title = story.get("representative_title", "无标题")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])

        material += f"【素材{i}】\n标题：{title}\n摘要：\n"
        for s in summaries:
            material += f"  - {s}\n"
        material += "\n"

    prompt = f"""你是专业的科技媒体编辑，需要根据以下今日的 AI 和科技新闻素材，写一份专业的「科技新闻日报」。

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

    return prompt


def _call_llm_ollama_direct(prompt: str, model: str) -> str:
    """使用 requests 直接调用 Ollama 原生 API，并通过 stream 避免超时切断连接"""
    url = "http://192.168.7.7:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_ctx": 16384,  # 给足上下文窗口长度
            "temperature": 0.7,
        },
    }
    log.info(f"发起原生 Ollama 流式调用 (模型: {model}, 节点: {url})")
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)  # 连接超时控制在120s
        resp.raise_for_status()

        full_text = []
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        full_text.append(chunk)
                        # 这里可以打印一下进度，避免觉得卡死
                        print(chunk, end="", flush=True)
                except Exception:
                    pass
        print()  # 换行
        text = "".join(full_text)
        log.info(f"Ollama 流式调用成功，返回 {len(text)} 字符")
        return text
    except Exception as e:
        log.error(f"Ollama 原生流式调用失败: {e}")
        return None


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sources", default="config/sources.yaml")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = read_yaml(root / args.config)
    sources = load_sources(root / args.sources)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})

    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    connect_timeout = int(fetch_cfg.get("connect_timeout", 5))
    user_agent = str(fetch_cfg.get("user_agent") or "ai-news-podcast/0.1")
    max_items_per_feed = int(fetch_cfg.get("max_items_per_feed", 30))
    max_pages = int(processing_cfg.get("max_pages", 80))

    now = datetime.now()
    date_str = now.strftime("%Y年%m月%d日")
    report_id = now.strftime("%Y-%m-%d")

    # 1. 抓取新闻
    log.info("Stage 1: Fetching RSS feeds...")
    raw_items = await fetch_all(
        sources,
        timeout_seconds=timeout_seconds,
        connect_timeout=connect_timeout,
        user_agent=user_agent,
        max_items_per_feed=max_items_per_feed,
        max_pages=max_pages,
    )

    if not raw_items:
        log.warning("No items fetched. Aborting.")
        return 1

    # 2. 数据处理与评分
    log.info("Stage 2: Processing and scoring items...")
    brief = process(raw_items, processing_cfg=processing_cfg)

    # 3. 构造 Prompt 并调用 LLM
    log.info("Stage 3: Generating report via LLM (using smaller local model)...")
    prompt = build_report_prompt(brief, date_str)

    log.info(f"Generated prompt string length: {len(prompt)} characters")

    # 彻底弃用 _call_llm 里的 openai 库包装，直接用原生 API 强行生成
    report_md = _call_llm_ollama_direct(prompt, "qwen3.5:27b")

    if not report_md:
        log.warning("Failed to generate report from LLM. Generating a basic fallback report.")
        report_md = f"# 🌍 科技新闻日报 | {date_str}\n\n> 由于大模型服务暂时不可用，以下是由系统自动整理的新闻速览：\n\n"

        stories = brief.get("stories", [])
        active = [s for s in stories if isinstance(s, dict) and s.get("role") != "skip"]
        active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
        active = active[:15]

        for i, story in enumerate(active, 1):
            title = story.get("representative_title", "无标题")
            context = story.get("context", {})
            summaries = context.get("factual_summary", [])
            report_md += f"## {i}. {title}\n"
            for s in summaries:
                report_md += f"- **{s}**\n"
            report_md += "\n"

    # 清理顶部可能的 ```markdown 标记
    if report_md.startswith("```markdown"):
        report_md = report_md[11:].strip()
    if report_md.endswith("```"):
        report_md = report_md[:-3].strip()

    # 4. 保存报告
    report_path = outdir / f"daily_report_{report_id}.md"
    report_path.write_text(report_md, encoding="utf-8")

    log.info(f"✨ Daily report generated successfully! Saved to: {report_path}")
    print("\n" + "=" * 50)
    print(f"📄 报告已生成: {report_path}")
    print("=" * 50 + "\n")
    print(report_md)
    print("\n" + "=" * 50)

    return 0


def entrypoint() -> int:
    """Synchronous entrypoint for console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(entrypoint())
