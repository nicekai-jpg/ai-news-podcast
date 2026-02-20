"""Stage 3 — 脚本生产模块（LLM 融合写作版）

职责：episode_brief → 通义千问 LLM 消化融合 → 连贯中文播客脚本（含 mood 标记）+ Show Notes。
英文内容在 LLM 阶段直接翻译为中文，TTS 朗读零障碍。
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 禁用词
# ---------------------------------------------------------------------------

DEFAULT_BANNED_WORDS = [
    "废话不多说",
    "众所周知",
    "颠覆",
    "炸裂",
    "重磅",
    "王炸",
    "杀疯了",
    "遥遥领先",
    "细思极恐",
]


def check_banned_words(text: str, banned: list[str] | None = None) -> list[str]:
    banned = banned or DEFAULT_BANNED_WORDS
    return [w for w in banned if w in text]


def _replace_banned_words(text: str, banned: list[str] | None = None) -> str:
    banned = banned or DEFAULT_BANNED_WORDS
    for w in banned:
        text = text.replace(w, "")
    return text


# ---------------------------------------------------------------------------
# TTS 文本清洗（LLM 产出后的最终保险）
# ---------------------------------------------------------------------------


def _sanitize_for_tts(text: str) -> str:
    """清洗 LLM 输出中 TTS 不友好的残留内容。"""
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    text = re.sub(
        r"[（(][^）)]{0,10}(?:doge|狗头|笑|手动|滑稽|哭|捂脸)[^）)]{0,5}[）)]", "", text
    )
    text = re.sub(r"[「」『』【】]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。.]{2,}", "。", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# 新闻素材 → LLM prompt 构建
# ---------------------------------------------------------------------------


def _build_material_text(brief: dict[str, Any], max_stories: int = 8) -> str:
    """把 episode_brief 中评分最高的新闻素材整理为结构化文本，供 LLM 消化。

    只取 main + supporting + quick 中评分最高的前 max_stories 条，
    避免素材过多导致 LLM 超时或输出质量下降。
    """
    stories = brief.get("stories", [])
    active = [s for s in stories if s.get("role") != "skip"]
    active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
    active = active[:max_stories]

    sections: list[str] = []
    for i, story in enumerate(active, 1):
        role = story.get("role", "quick")
        role_label = {"main": "重要", "supporting": "次要", "quick": "简讯"}.get(
            role, "简讯"
        )
        title = story.get("representative_title", "无标题")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        background = context.get("historical_background", "")
        sources = context.get("sources_ranked", [])
        items = story.get("items", [])

        part = f"【素材{i}】（{role_label}）\n标题：{title}\n"

        if summaries:
            part += "摘要：\n"
            for s in summaries:
                part += f"  - {s}\n"

        if items:
            best = max(items, key=lambda x: len(x.get("full_text_snippet", "")))
            snippet = best.get("full_text_snippet", "")
            if snippet and len(snippet) > 50:
                part += f"详情：{snippet[:800]}\n"
            src_name = best.get("source_name", "")
            if src_name:
                part += f"来源：{src_name}\n"

        if background:
            part += f"背景：{background}\n"

        if sources:
            src_names = "、".join(s["name"] for s in sources[:3])
            part += f"综合来源：{src_names}\n"

        sections.append(part)

    return "\n".join(sections)


def _build_llm_prompt(
    material: str,
    episode_date: datetime,
    podcast_title: str,
    style_cfg: dict[str, Any],
) -> str:
    """构建给 LLM 的完整 prompt。"""
    total_range = style_cfg.get("total_chars", [1800, 3900])
    min_chars, max_chars = total_range[0], total_range[1]
    banned = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    banned_str = "、".join(banned)
    date_str = _cn_date(episode_date)

    return f"""你是「{podcast_title}」的主播，需要根据以下新闻素材，写一篇连贯、自然、有深度的中文播客脚本。

## 核心要求

1. **融合写作，不是罗列**：不要逐条复述新闻，而是把相关素材融合在一起，找到内在联系，讲出一个有主线的故事。
2. **全部用中文**：所有英文内容必须翻译为中文。英文专有名词（如公司名、产品名、技术术语）用中文表述，必要时首次出现可括号标注英文原名，之后只用中文。例如：「大型语言模型（LLM）」之后直接说「大语言模型」。
3. **口语化**：这是播客脚本，不是新闻稿。用说话的方式写，句子短（12-28字），节奏明快。可以用「你知道吗」「说白了」「换句话说」等口语衔接。
4. **有观点有态度**：不要干巴巴地陈述事实，适当加入分析和看法，但要标明哪些是事实哪些是推测。
5. **禁用词**：不要使用以下词汇：{banned_str}

## 脚本结构

用 [mood:xxx] 标记来控制语气情绪，可用的标记有：hook、excited、serious、calm、emphasis、closing。

请按以下结构组织脚本：

**开场** [mood:hook]：用一个引人入胜的问题或悬念开场，吸引听众继续听下去。提到今天是{date_str}。不要说「欢迎收听」这种套话。

**主体**：把最重要的 1-2 个话题展开讲透（用 [mood:excited] 或 [mood:serious]），次要话题简要带过（用 [mood:calm]）。话题之间用自然过渡衔接，不要用「第一条、第二条」这种机械编号。

**快讯**（如果有简讯类素材）[mood:emphasis]：用 2-3 句话快速带过。

**收尾** [mood:closing]：总结今天的核心观点，给听众一个思考的角度。自然结束，不要说「感谢收听」。

## 字数要求

总字数控制在 {min_chars}-{max_chars} 字之间（不含 mood 标记）。

## 今日素材

{material}

## 输出格式

直接输出脚本文本，每个段落前用 [mood:xxx] 标记情绪。不要输出任何解释、标题、分隔线或 markdown 格式。"""


# ---------------------------------------------------------------------------
# DashScope (通义千问) LLM 调用 — OpenAI 兼容格式
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, llm_cfg: dict[str, Any]) -> str | None:
    """调用通义千问 DashScope API 生成播客脚本。失败返回 None。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY 未设置，无法调用 LLM")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai 包未安装，请安装: pip install openai")
        return None

    model_name = llm_cfg.get("model", "qwen-plus")
    temperature = llm_cfg.get("temperature", 0.7)
    max_tokens = llm_cfg.get("max_output_tokens", 8192)
    base_url = llm_cfg.get(
        "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    timeout = llm_cfg.get("timeout", 120)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(
                "调用通义千问 (%s), attempt %d/%d, timeout %ds",
                model_name,
                attempt + 1,
                max_retries,
                timeout,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的中文播客脚本撰稿人，擅长将科技新闻素材融合为连贯、生动、有深度的口语化播客文章。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )
            text = response.choices[0].message.content
            if text and len(text.strip()) > 200:
                logger.info("通义千问返回 %d 字符", len(text))
                return text.strip()
            logger.warning(
                "通义千问返回内容过短 (%d 字符)，重试", len(text) if text else 0
            )
        except Exception as e:
            logger.warning("通义千问调用失败 (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    logger.error("通义千问全部 %d 次重试失败", max_retries)
    return None


# ---------------------------------------------------------------------------
# 纯模板兜底（LLM 不可用时）
# ---------------------------------------------------------------------------


def _build_fallback(
    brief: dict[str, Any],
    episode_date: datetime,
    podcast_title: str,
) -> str:
    """无 LLM 时的极简兜底脚本。"""
    stories = brief.get("stories", [])
    active = [s for s in stories if s.get("role") != "skip"]

    lines: list[str] = []
    lines.append(
        f"[mood:hook] 今天是{_cn_date(episode_date)}，来看看AI领域有什么新动向。"
    )

    for i, story in enumerate(active[:6]):
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        summary = summaries[0] if summaries else ""

        if i == 0:
            lines.append(f"[mood:excited] 今天最值得关注的是，{title}。{summary}。")
        else:
            lines.append(f"[mood:calm] 另外，{title}。{summary}。")

    lines.append(f"[mood:closing] 以上就是今天的AI动态，{podcast_title}，明天见。")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def generate_script(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str = "脑活素 AI 新闻播客",
    script_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """
    生成播客脚本。先尝试 LLM 融合写作，失败时退回模板兜底。

    Returns: (script_text, warnings)
    """
    cfg = script_cfg or {}
    style_cfg = cfg.get("style", {})
    banned_words = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    llm_cfg = llm_cfg or {}

    material = _build_material_text(brief)
    logger.info("素材文本 %d 字符，准备调用 LLM", len(material))

    script = None
    mode_used = "fallback"

    if material.strip():
        prompt = _build_llm_prompt(material, episode_date, podcast_title, style_cfg)
        raw = _call_llm(prompt, llm_cfg)
        if raw:
            script = raw
            mode_used = "LLM"
            logger.info("LLM 脚本生成成功")

    if not script:
        logger.warning("LLM 不可用，使用模板兜底")
        script = _build_fallback(brief, episode_date, podcast_title)
        mode_used = "fallback"

    script = _sanitize_for_tts(script)
    script = _replace_banned_words(script, banned_words)
    script = _normalize_mood_tags(script)

    warnings: list[str] = []

    found_banned = check_banned_words(script, banned_words)
    if found_banned:
        warnings.append(f"仍含禁用词: {found_banned}")

    total_range = style_cfg.get("total_chars", [1800, 3900])
    clean_text = re.sub(r"\[mood:\w+\]\s*", "", script)
    char_count = len(clean_text.replace("\n", "").replace(" ", ""))
    if char_count < total_range[0]:
        warnings.append(f"脚本字数 {char_count} 低于下限 {total_range[0]}")
    elif char_count > total_range[1]:
        warnings.append(f"脚本字数 {char_count} 超过上限 {total_range[1]}")

    if not re.search(r"\[mood:\w+\]", script):
        warnings.append("脚本缺少 [mood:xxx] 标记，TTS 将使用默认语气")

    logger.info(
        "脚本生成完毕 (Mode: %s), %d 字, %d 条警告",
        mode_used,
        char_count,
        len(warnings),
    )
    return script, warnings


def _normalize_mood_tags(text: str) -> str:
    """规范化 mood 标记格式，确保 TTS 引擎能正确解析。"""
    valid_moods = {"hook", "excited", "serious", "calm", "emphasis", "closing"}
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = re.match(r"\[mood:(\w+)\]\s*(.*)", line)
        if m:
            mood, content = m.group(1), m.group(2)
            if mood not in valid_moods:
                mood = "calm"
            if content.strip():
                result.append(f"[mood:{mood}] {content.strip()}")
        else:
            if line.strip():
                result.append(f"[mood:calm] {line.strip()}")

    return "\n".join(result) + "\n" if result else "[mood:calm] 暂无内容。\n"


# ---------------------------------------------------------------------------
# Show Notes — Markdown
# ---------------------------------------------------------------------------


def generate_show_notes(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    stories = brief.get("stories", [])
    thesis = brief.get("thesis", "")
    active = [s for s in stories if s.get("role") != "skip"]

    lines: list[str] = []
    lines.append(f"# {episode_title}")
    lines.append("")
    lines.append(f"**日期**: {_cn_date(episode_date)}")
    lines.append("")

    if thesis:
        lines.append(f"> {thesis}")
        lines.append("")

    for role, label in [
        ("main", "🔴 主要报道"),
        ("supporting", "🟡 支撑消息"),
        ("quick", "🟢 快讯"),
    ]:
        role_stories = [s for s in active if s.get("role") == role]
        if not role_stories:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for story in role_stories:
            title = story.get("representative_title", "")
            items = story.get("items", [])
            context = story.get("context", {})
            summaries = context.get("factual_summary", [])
            total = story.get("total_score", 0)

            lines.append(f"### {title}")
            lines.append("")
            if summaries:
                for s in summaries:
                    lines.append(f"- {s}")
                lines.append("")
            if items:
                lines.append("**来源链接：**")
                lines.append("")
                for item in items[:5]:
                    name = item.get("source_name", "")
                    link = item.get("link", "")
                    lines.append(f"- [{name}]({link})")
                lines.append("")
            lines.append(f"*综合评分: {total}/15*")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*本期由 AI 自动生成，数据截至 {_cn_date(episode_date)}*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Show Notes — HTML
# ---------------------------------------------------------------------------


def generate_show_notes_html(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    stories = brief.get("stories", [])
    active = [s for s in stories if s.get("role") != "skip"]

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    items_html: list[str] = []
    for story in active:
        title = _esc(story.get("representative_title", ""))
        story_items = story.get("items", [])
        role_emoji = story.get("role_emoji", "")
        if story_items:
            link = story_items[0].get("link", "")
            source = _esc(story_items[0].get("source_name", ""))
            items_html.append(
                f'<li>{role_emoji} <a href="{link}">{title}</a> <small>({source})</small></li>'
            )

    date_text = _cn_date(episode_date)
    safe_title = _esc(episode_title)
    body = "\n".join(items_html)

    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{safe_title}</title>\n"
        "  <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;"
        "max-width:860px;margin:24px auto;padding:0 16px;line-height:1.6}"
        "li{margin:12px 0}small{color:#555}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{safe_title}</h1>\n"
        f"<p>{date_text}</p>\n"
        "<ol>\n"
        f"{body}\n"
        "</ol>\n"
        "</body>\n"
        "</html>\n"
    )
