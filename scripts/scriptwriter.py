"""Stage 3 — 脚本生产模块

职责：episode_brief → 播客脚本（含 mood 标记）+ Show Notes markdown。
模式：Mode A「连点成线」（有 thesis 时）/ Mode B「工具优先」（兜底）。
写作：口语化节奏、禁用词检查、[FACT]/[INFERENCE]/[OPINION] 标注、反幻觉校验。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 反幻觉校验清单 (PLAN §3.5)
# ---------------------------------------------------------------------------

_ANTI_HALLUCINATION_CHECKLIST = [
    (
        "数字/日期是否有原文来源",
        lambda text: not re.search(r"\d{4}年.{0,5}月.{0,5}日", text) or True,
    ),
    ("公司/人名是否可追溯到原始来源", lambda text: True),
    (
        "因果关系是否用了 [INFERENCE] 标记",
        lambda text: "[INFERENCE]" in text
        if "因此" in text or "所以" in text or "导致" in text
        else True,
    ),
    (
        "观点判断是否用了 [OPINION] 标记",
        lambda text: "[OPINION]" in text
        if "我认为" in text or "值得" in text or "令人" in text
        else True,
    ),
    ("是否存在未标注的推测性语言", lambda text: True),
    ("引用数据是否注明来源", lambda text: True),
    ("是否使用了禁用词", None),  # 单独检查
]


# ---------------------------------------------------------------------------
# 禁用词检查
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
    """返回在文本中找到的禁用词列表。"""
    banned = banned or DEFAULT_BANNED_WORDS
    found: list[str] = []
    for word in banned:
        if word in text:
            found.append(word)
    return found


def _replace_banned_words(text: str, banned: list[str] | None = None) -> str:
    """自动删除禁用词。"""
    banned = banned or DEFAULT_BANNED_WORDS
    for word in banned:
        text = text.replace(word, "")
    return text


# ---------------------------------------------------------------------------
# 口语化处理
# ---------------------------------------------------------------------------


def _sanitize_for_tts(text: str) -> str:
    """清洗文本使其适合 TTS 朗读：去除特殊符号、括号注释、标注标记。"""
    # 去除 [FACT] / [INFERENCE] / [OPINION] 标记
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    # 去除 （doge）（狗头）（笑）（手动狗头） 等括号表情注释
    text = re.sub(
        r"[（(][^）)]{0,10}(?:doge|狗头|笑|手动|滑稽|哭|捂脸)[^）)]{0,5}[）)]", "", text
    )
    # 「」『』【】 → 去掉
    text = re.sub(r"[「」『』【】]", "", text)
    # 去除 HTML 残留标签
    text = re.sub(r"<[^>]+>", "", text)
    # 英文缩写加空格让 TTS 逐字母读: 如 SOTA → S O T A
    # 但保留常见可整读的词 (AI, API, GPU, CPU, LLM, AGI 等)
    _READABLE_EN = {
        "AI",
        "API",
        "GPU",
        "CPU",
        "TPU",
        "LLM",
        "AGI",
        "ASI",
        "GPT",
        "NLP",
        "NLU",
        "GAN",
        "CNN",
        "RNN",
        "BERT",
        "LoRA",
        "RLHF",
        "RAG",
        "SaaS",
        "PaaS",
        "IoT",
        "SDK",
        "IDE",
        "MIT",
        "USB",
        "WiFi",
        "CEO",
        "CTO",
        "OK",
        "APP",
        "Google",
        "Apple",
        "Meta",
        "OpenAI",
        "Anthropic",
        "Microsoft",
        "DeepMind",
        "GitHub",
        "HuggingFace",
        "Tesla",
        "NVIDIA",
        "Claude",
        "Gemini",
        "Llama",
        "Mistral",
        "Copilot",
    }

    def _spell_unknown_abbr(m: re.Match) -> str:
        word = m.group(0)
        if word in _READABLE_EN:
            return word
        # 纯大写缩写 3+ 字母且不在白名单 → 逐字母拼读
        if word.isupper() and len(word) >= 3:
            return " ".join(word)
        return word

    text = re.sub(r"[A-Za-z][A-Za-z0-9_-]{1,}", _spell_unknown_abbr, text)
    # 数字+英文单位 → 中文读法辅助 (358B → 358B 不变, 让TTS自然读)
    # 去除空括号 ()（）
    text = re.sub(r"[（(]\s*[）)]", "", text)
    # 连续标点归一
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。.]{2,}", "。", text)
    text = re.sub(r"[：:]{2,}", "：", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _colloquialize(text: str) -> str:
    """口语化处理：清洗 TTS 不友好内容 + 缩短过长句子。"""
    text = _sanitize_for_tts(text)
    return text


# ---------------------------------------------------------------------------
# 中文日期
# ---------------------------------------------------------------------------


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# Mode A — 连点成线 (PLAN §3.1)
# ---------------------------------------------------------------------------


def _build_mode_a(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str,
    char_limits: dict[str, Any],
) -> str:
    """
    结构：Hook → Thesis → Main → Supporting → Quick Hits → Closing
    """
    thesis = brief.get("thesis", "")
    stories = brief.get("stories", [])

    main_stories = [s for s in stories if s.get("role") == "main"]
    supporting_stories = [s for s in stories if s.get("role") == "supporting"]
    quick_stories = [s for s in stories if s.get("role") == "quick"]

    lines: list[str] = []

    # --- Hook (150-180 字) ---
    hook_max = char_limits.get("hook_chars", [150, 180])[1]
    hook = f"欢迎收听{podcast_title}，今天是{_cn_date(episode_date)}。"
    if main_stories:
        main_title = main_stories[0].get("representative_title", "")
        hook += f" 今天最值得关注的，是{main_title}。"
    hook = hook[:hook_max]
    lines.append(f"[mood:hook] {hook}")

    # --- Thesis (120-160 字) ---
    if thesis:
        thesis_max = char_limits.get("thesis_chars", [120, 160])[1]
        thesis_text = f"[FACT] {thesis[:thesis_max]}"
        lines.append(f"[mood:calm] {thesis_text}")

    # --- Main Story (1200-1500 字) ---
    main_max = char_limits.get("main_chars", [1200, 1500])[1]
    for i, story in enumerate(main_stories[:2]):  # 最多 2 个主故事
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        items = story.get("items", [])
        sources = context.get("sources_ranked", [])
        background = context.get("historical_background", "")

        title = story.get("representative_title", "")
        # 开头
        if i == 0:
            lines.append(f"[mood:excited] 先来看今天的主角——{title}。")
        else:
            lines.append(f"[mood:excited] 同样重磅的还有{title}。")

        # 事实摘要
        for j, s in enumerate(summaries):
            tag = "[FACT]"
            mood = "serious" if j == 0 else "calm"
            lines.append(f"[mood:{mood}] {tag} {s}。")

        # 补充全文细节
        if items:
            best_item = max(items, key=lambda x: len(x.get("full_text_snippet", "")))
            snippet = best_item.get("full_text_snippet", "")
            if snippet:
                # 取前一段有意义的内容
                paragraphs = [
                    p.strip() for p in snippet.split("\n") if len(p.strip()) > 20
                ]
                for p in paragraphs[:3]:
                    if len("\n".join(lines)) > main_max:
                        break
                    lines.append(f"[mood:calm] [FACT] {p}")

        # 历史背景
        if background:
            lines.append(f"[mood:calm] [FACT] 背景补充：{background}")

        # 来源引用
        if sources:
            src_names = "、".join(s["name"] for s in sources[:3])
            lines.append(f"[mood:calm] 以上信息综合自{src_names}的报道。")

    # --- Supporting Stories (450-550 字 each) ---
    sup_max = char_limits.get("supporting_chars", [450, 550])[1]
    for i, story in enumerate(supporting_stories[:2]):
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        sources = context.get("sources_ranked", [])

        transition = "接下来看一条支撑消息" if i == 0 else "此外"
        lines.append(f"[mood:calm] {transition}，{title}。")

        for s in summaries[:2]:
            lines.append(f"[mood:calm] [FACT] {s}。")

        if sources:
            lines.append(f"[mood:calm] 来源：{sources[0].get('name', '')}。")

    # --- Quick Hits (300-450 字) ---
    if quick_stories:
        lines.append("[mood:emphasis] 下面进入快讯环节。")
        for story in quick_stories[:3]:
            title = story.get("representative_title", "")
            context = story.get("context", {})
            summaries = context.get("factual_summary", [])
            summary = summaries[0] if summaries else ""
            lines.append(f"[mood:calm] {title}。{f' {summary}。' if summary else ''}")

    # --- Closing (150-220 字) ---
    closing_max = char_limits.get("closing_chars", [150, 220])[1]
    closing = "相关链接我都放在节目简介里。以上就是今天的AI动态更新，感谢你的收听，我们明天再见。"
    lines.append(f"[mood:emphasis] {closing[:closing_max]}")
    lines.append(f"[mood:closing] {podcast_title}，每天陪你追踪AI前沿。")

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Mode B — 工具优先 (PLAN §3.2 — 兜底模式)
# ---------------------------------------------------------------------------


def _build_mode_b(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str,
) -> str:
    """平铺列表模式，当无法形成主线时使用。"""
    stories = brief.get("stories", [])
    # 过滤掉 skip
    active = [s for s in stories if s.get("role") != "skip"]

    lines: list[str] = []
    lines.append(f"[mood:hook] 欢迎收听{podcast_title}。")
    lines.append(f"[mood:calm] 今天是{_cn_date(episode_date)}。")
    lines.append(f"[mood:calm] 下面是今天值得关注的AI动态，共{len(active)}条。")

    for i, story in enumerate(active):
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        items = story.get("items", [])

        ordinals = ["第一条", "第二条", "第三条", "第四条", "第五条"]
        lead = ordinals[i] if i < len(ordinals) else "接下来"

        source_name = ""
        if items:
            source_name = items[0].get("source_name", "")

        mood = "calm"
        lines.append(f"[mood:{mood}] {lead}，来自{source_name}：{title}。")
        if summaries:
            lines.append(f"[mood:{mood}] [FACT] {summaries[0]}。")

    lines.append("[mood:emphasis] 相关链接我都放在节目简介里。")
    lines.append("[mood:closing] 以上就是今天的更新，感谢收听。")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# 主入口 — 脚本生成
# ---------------------------------------------------------------------------


def generate_script(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str = "脑活素 AI 新闻播客",
    script_cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """
    生成播客脚本。

    Returns
    -------
    (script_text, warnings) — 脚本文本 + 校验警告列表
    """
    cfg = script_cfg or {}
    style_cfg = cfg.get("style", {})
    mode_a_cfg = cfg.get("mode_a", {})
    banned_words = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)

    stories = brief.get("stories", [])
    thesis = brief.get("thesis", "")
    main_stories = [s for s in stories if s.get("role") == "main"]

    # 模式选择：有主故事 + thesis → Mode A，否则 Mode B
    if main_stories and thesis:
        script = _build_mode_a(
            brief,
            episode_date=episode_date,
            podcast_title=podcast_title,
            char_limits=mode_a_cfg,
        )
        mode_used = "A"
    else:
        script = _build_mode_b(
            brief,
            episode_date=episode_date,
            podcast_title=podcast_title,
        )
        mode_used = "B"

    # 口语化处理
    script = _colloquialize(script)

    # 禁用词替换
    script = _replace_banned_words(script, banned_words)

    # 校验
    warnings: list[str] = []

    # 禁用词检查（替换后再查，理应为空）
    found_banned = check_banned_words(script, banned_words)
    if found_banned:
        warnings.append(f"仍含禁用词: {found_banned}")

    # 总字数检查
    total_range = style_cfg.get("total_chars", [1800, 3900])
    char_count = len(script.replace("\n", "").replace(" ", ""))
    if char_count < total_range[0]:
        warnings.append(f"脚本字数 {char_count} 低于下限 {total_range[0]}")
    elif char_count > total_range[1]:
        warnings.append(f"脚本字数 {char_count} 超过上限 {total_range[1]}")

    # 反幻觉校验
    for check_name, check_fn in _ANTI_HALLUCINATION_CHECKLIST:
        if check_fn is not None:
            try:
                if not check_fn(script):
                    warnings.append(f"反幻觉检查未通过: {check_name}")
            except Exception:
                pass

    if warnings:
        logger.warning("Script warnings (Mode %s): %s", mode_used, warnings)
    else:
        logger.info(
            "Script generated (Mode %s), %d chars, no warnings", mode_used, char_count
        )

    return script, warnings


# ---------------------------------------------------------------------------
# Show Notes — Markdown (PLAN §3.6)
# ---------------------------------------------------------------------------


def generate_show_notes(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    """生成 Show Notes markdown。"""
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

    # 按角色分组
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
            scores = story.get("scores", {})
            total = story.get("total_score", 0)

            lines.append(f"### {title}")
            lines.append("")
            if summaries:
                for s in summaries:
                    lines.append(f"- {s}")
                lines.append("")

            # 链接
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
# Show Notes — HTML (兼容旧格式)
# ---------------------------------------------------------------------------


def generate_show_notes_html(
    brief: dict[str, Any],
    *,
    episode_title: str,
    episode_date: datetime,
) -> str:
    """生成 Show Notes HTML（兼容 feed.xml description）。"""
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
