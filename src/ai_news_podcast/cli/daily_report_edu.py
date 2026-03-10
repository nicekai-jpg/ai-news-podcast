import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 添加 src 目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_news_podcast.cli.run_daily import _read_yaml, _load_sources, _episode_id
from ai_news_podcast.pipeline.fetcher import fetch_all
from ai_news_podcast.pipeline.processor import process
from ai_news_podcast.pipeline.scriptwriter import _call_llm

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

    prompt = f"""你是「国家宏观产业与高考升学战略规划首席专家」，你需要根据以下抓取到的今日宏观政策、产业动态及教育改革资讯，写一份顶级硬核的「前瞻升学与报考内参」。

## 核心任务与分析逻辑
不要机械地罗列新闻！你要使用**“自上而下（Top-Down）”**的战略推演逻辑，将宏观新闻转化为高中生及家长的“报考志愿指引”。

## 强制输出结构（Markdown格式）

1. **大标题**：如 `# 🎯 前瞻升学与报考内参 | {date_str}`
2. **核心风向提要**：用一段话总结今天涉及的核心国家战略与升学趋势。
3. **战略推演分析**（按主题域展开，至少推演2-3个主题）：
   针对每一个重大新闻/产业风口，必须严格按照以下三步体例进行撰写：
   
   ✅ **【国家/产业风向标】**：（总结今天国家发布了哪些重要政策？或哪些“卡脖子”硬科技/新质生产力迎来了爆发？）
   ✅ **【学科与专业映射】**：（基于上述风向，哪类大学的哪些具体“专业/交叉学科/新工科/新医科”将会受国家大幅度资金倾斜和重用？比如人工智能催生了智能科学与技术、自动化类需求）
   ✅ **【升学赛道实操建议】**：（明确给出对于当前初高中生，想进入这些重点专业，应该走哪条路？重点结合**“强基计划（注重基础学科）”**、**“综合评价招生（注重综合素质）”**、**“国家专项/高校专项”**或是**“普通统招的高发捡漏批次”**给出明确报考策略）

4. **百字避坑指南**：在文末用引用的方式（`> `），指出近期需要避免报考或正在撤销的夕阳专业方向，或是规划上的常见误区。

## 写作要求
- **语境权威**：你是资深规划师，用词极其专业、冷静、甚至一针见血。
- **结合素材推演**：必须紧密围绕今天抓取的情报素材来发散，若素材不足，可基于你的常识进行发散，但逻辑必须严密。
- **绝不废话**：直指核心，总字数控制在 1000 - 1500 字，全中文。

## 今日素材（包括宏观政策、产业与教育）
{material}

请直接输出 Markdown 文本，不要带有 ```markdown 的代码块。"""
    
    return prompt

def _call_llm_ollama_direct(prompt: str, model: str) -> str:
    """使用 requests 直接调用 Ollama 原生 API，并通过 stream 避免超时切断连接"""
    url = "http://192.168.7.7:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_ctx": 8192,  # 降低上下文窗口，避免 27b 模型爆显存导致 Ollama 崩溃
            "temperature": 0.7
        }
    }
    log.info(f"发起原生 Ollama 流式调用 (模型: {model}, 节点: {url})")
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=300)  # 连接超时控制在300s
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
                except Exception as e:
                    pass
        print() # 换行
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
    ap.add_argument("--config", default="config/config_edu.yaml")
    ap.add_argument("--sources", default="config/sources_edu.yaml")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    cfg = _read_yaml(root / args.config)
    sources = _load_sources(root / args.sources)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fetch_cfg = cfg.get("fetch", {})
    processing_cfg = cfg.get("processing", {})
    llm_cfg = cfg.get("llm", {})

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
        report_md = f"# 🎯 前瞻升学与报考内参 | {date_str}\n\n> 由于大模型服务暂时不可用，以下是由系统自动整理的资讯速览：\n\n"
        
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
    report_path = outdir / f"education_report_{report_id}.md"
    report_path.write_text(report_md, encoding="utf-8")
    
    log.info(f"✨ Daily report generated successfully! Saved to: {report_path}")
    print("\n" + "="*50)
    print(f"📄 报告已生成: {report_path}")
    print("="*50 + "\n")
    print(report_md)
    print("\n" + "="*50)

    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
