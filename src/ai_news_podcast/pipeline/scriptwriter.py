"""Stage 3 — 脚本生产模块（LLM 多 Agent 融合写作版）

职责：episode_brief → 主编Agent选定头条/大纲 → 撰稿Agent转化为双人对白播客脚本（Host A, Host B）。
英文内容在 LLM 阶段直接翻译为中文，输出格式严格遵循对话剧本要求，供后续混音使用。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from ai_news_podcast.prompts import DEFAULT_BANNED_WORDS, build_editor_prompt, build_writer_prompt
from ai_news_podcast.text_utils import RE_MOOD_TAG, clean_tts_text

logger = logging.getLogger(__name__)


def check_banned_words(text: str, banned: list[str] | None = None) -> list[str]:
    banned = banned or DEFAULT_BANNED_WORDS
    return [w for w in banned if w in text]


def _replace_banned_words(text: str, banned: list[str] | None = None) -> str:
    """更体面地替换禁用词，避免句子断线。"""
    banned = banned or DEFAULT_BANNED_WORDS
    # 映射表：将某些词替换为中性词，其余直接移除
    replacements = {
        "炸裂": "非常",
        "王炸": "王牌",
        "杀疯了": "非常激烈",
        "遥遥领先": "保持领先",
        "细思极恐": "值得深思",
        "重磅": "重要",
    }
    for w in banned:
        target = replacements.get(w, "")
        text = text.replace(w, target)
    return text


# ---------------------------------------------------------------------------
# TTS 文本清洗（LLM 产出后的最终保险）
# ---------------------------------------------------------------------------


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# 新闻素材 → LLM prompt 构建
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 默认实体列表（当 config.yaml 中未配置 entities.companies 时使用）
# ---------------------------------------------------------------------------

_DEFAULT_COMPANIES = [
    "谷歌",
    "google",
    "openai",
    "微软",
    "microsoft",
    "英伟达",
    "nvidia",
    "苹果",
    "apple",
    "meta",
    "anthropic",
    "claude",
    "字节",
    "腾讯",
    "百度",
    "阿里",
    "华为",
    "奥迪",
    "audi",
    "特斯拉",
    "tesla",
]


def _load_companies_from_config() -> list[str]:
    """从 config.yaml 读取 entities.companies，读取失败时返回默认值。"""
    try:
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        companies = cfg.get("entities", {}).get("companies", [])
        return companies if companies else _DEFAULT_COMPANIES
    except (OSError, yaml.YAMLError, KeyError):
        return _DEFAULT_COMPANIES


def _get_story_entities(story: dict[str, Any]) -> set[str]:
    """提取新闻故事中的公司/品牌实体词，用于多样性去重。"""
    title = str(story.get("representative_title", "")).lower()
    entities = set()
    COMPANIES = _load_companies_from_config()
    for c in COMPANIES:
        if c in title:
            # 标准化实体名
            norm = c
            if c in ("google", "谷歌"):
                norm = "谷歌"
            elif c in ("microsoft", "微软"):
                norm = "微软"
            elif c in ("nvidia", "英伟达"):
                norm = "英伟达"
            elif c in ("apple", "苹果"):
                norm = "苹果"
            elif c in ("audi", "奥迪"):
                norm = "奥迪"
            elif c in ("tesla", "特斯拉"):
                norm = "特斯拉"
            entities.add(norm)
    return entities


def _build_material_text(brief: dict[str, Any], max_stories: int = 8) -> str:
    """把 episode_brief 中最重要的新闻素材整理为结构化文本。

    采用动态实体多样性重排算法（类似 MMR 推荐算法）：
    按分数降序挑选，但如果候选新闻包含已选中实体的关键词，则对其施加分数惩罚（每冲突一次扣 3 分），
    从而防止同一家公司的多条新闻霸榜，保证节目题材的丰富度。
    """
    stories = brief.get("stories", [])
    active: list[dict[str, Any]] = [
        s for s in stories if isinstance(s, dict) and s.get("role") != "skip"
    ]

    # 动态多样性选择循环
    selected: list[dict[str, Any]] = []
    entity_counts: dict[str, int] = {}
    candidates = [dict(s) for s in active]

    while len(selected) < max_stories and candidates:
        # 对剩余候选计算有效得分 (Effective Score)
        for c in candidates:
            orig_score = c.get("total_score", 0)
            c_entities = _get_story_entities(c)
            # 重复选择的实体每个惩罚 3 分
            penalty = sum(3 * entity_counts.get(ent, 0) for ent in c_entities)
            c["_temp_score"] = orig_score - penalty

        # 按临时有效得分重新排序并选择最高者
        candidates.sort(key=lambda x: x.get("_temp_score", 0), reverse=True)
        best = candidates.pop(0)
        selected.append(best)

        # 更新已选实体的计数器
        for ent in _get_story_entities(best):
            entity_counts[ent] = entity_counts.get(ent, 0) + 1

    top_active = selected

    sections: list[str] = []
    for i, story in enumerate(top_active, 1):
        role = story.get("role", "quick")
        role_label = {"main": "重要", "supporting": "次要", "quick": "简讯"}.get(role, "简讯")
        title = story.get("representative_title", "无标题")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        background = context.get("historical_background", "")
        sources = context.get("sources_ranked", [])

        part = f"【素材{i}】（{role_label}）\n标题：{title}\n"

        if summaries:
            part += "摘要：\n"
            for sm in summaries:
                part += f"  - {sm}\n"

        if background:
            part += f"背景：{background}\n"

        if sources:
            src_names = "、".join(src["name"] for src in sources[:3])
            part += f"综合来源：{src_names}\n"

        sections.append(part)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# DashScope (通义千问) LLM 调用 — OpenAI 兼容格式
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, llm_cfg: dict[str, Any]) -> str | None:
    """使用通用 OpenAI 兼容协议调用大语言模型。"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("缺少 openai 库，请运行 uv pip install openai")
        return None

    # 从配置中获取目标环境变量名、URL和模型
    env_key_name = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(env_key_name, "").strip()

    if not api_key:
        logger.error(f"未找到对应的环境变量 {env_key_name} 用于 API 鉴权。请在 .env 中设置。")
        return None

    model_name = llm_cfg.get("model", "deepseek-chat")
    base_url = llm_cfg.get("base_url")
    timeout = llm_cfg.get("timeout", 60)

    # 构造客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(
                "发起 LLM 调用 (模型: %s, 节点: %s)",
                model_name,
                base_url or "官方 OpenAI",
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_output_tokens", 2048),
            )

            text = response.choices[0].message.content
            if text:
                logger.info("LLM 调用成功，返回 %d 字符", len(text))
                return text.strip()

        except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as e:
            # openai SDK 内部基于 httpx 抛出异常，httpx.HTTPError 可覆盖网络层错误；
            # json.JSONDecodeError 覆盖响应解析失败；OSError/ValueError 覆盖环境与参数异常；
            # RuntimeError 覆盖 openai SDK 内部运行时错误。
            logger.warning("LLM 调用失败 (第 %d 次重试): %s", attempt + 1, e)
            time.sleep((attempt + 1) * 2)

    return None


# ---------------------------------------------------------------------------
# 纯模板兜底（LLM 不可用时）
# ---------------------------------------------------------------------------


def _build_fallback(
    brief: dict[str, Any],
    episode_date: datetime,
    podcast_title: str,
) -> str:
    """无 LLM 时的极简双人对谈兜底脚本。"""
    stories = brief.get("stories", [])
    active: list[dict[str, Any]] = [
        s for s in stories if isinstance(s, dict) and s.get("role") != "skip"
    ]

    lines: list[str] = []
    lines.append(
        f"[Host A] 听众朋友们好，欢迎收听{podcast_title}，我是博文。今天是{_cn_date(episode_date)}。"
    )
    lines.append("[Host B] 大家好，我是晓晓。今天这AI圈的瓜，那可是相当有意思。")

    for i, story in enumerate(active):
        if i >= 5:
            break
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        summary = summaries[0] if summaries else "细节不详，但极其重要。"

        if i == 0:
            lines.append(f"[Host A] 咱们先聊聊今天第一个大头条。那就是，{title}。")
            lines.append("[Host B] 晓晓也一直在关注这个，博文，这个事影响真的很大吗？")
            lines.append(f"[Host A] 确实。简单来说，{summary}")
        elif i == 1:
            lines.append(f"[Host B] 没错。那今天的第二个大头条，是关于：{title}。")
            lines.append(f"[Host A] 对，这个也备受关注，主要进展是：{summary}")
        else:
            lines.append(f"[Host B] 嗯，除了这两个头条，我看到快讯中 {title} 也有新动态。")
            lines.append(f"[Host A] 没错！关于这个，主要是 {summary}")

    lines.append("[Host B] 哇，每天都能看到新东西在冒出来。")
    lines.append(
        "[Host A] 科技变化就是快，那以上就是今天的 AI 重点资讯。感谢各位收听，咱们明天见！"
    )
    lines.append("[Host B] 拜拜！")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def generate_script(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str = "AI 每日先锋",
    script_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """
    生成播客脚本 (双Agent双播客机制)。
    Step 1: Editor Agent 提炼大纲
    Step 2: Writer Agent 编写对谈剧本
    失败时退回退化版的双人对谈模板。
    """
    cfg = script_cfg or {}
    style_cfg = cfg.get("style", {})
    banned_words = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    llm_cfg = llm_cfg or {}

    material = _build_material_text(brief)
    logger.info("素材文本 %d 字符，准备启动 Multi-Agent 编剧流程", len(material))

    script = None
    mode_used = "fallback"

    if material.strip():
        # --- Node 1: Editor ---
        editor_prompt = build_editor_prompt(material, episode_date)
        raw_editor = _call_llm(editor_prompt, llm_cfg)

        if raw_editor:
            logger.info("Editor Agent 生成大纲成功，准备注入 Writer 节点")
            # --- Node 2: Writer ---
            writer_prompt = build_writer_prompt(raw_editor, episode_date, podcast_title, style_cfg)
            raw_writer = _call_llm(
                writer_prompt, dict(llm_cfg, temperature=0.85)
            )  # 稍微提高Writer的temperature增加活泼度
            if raw_writer:
                script = raw_writer
                mode_used = "Multi-Agent"
                logger.info("Writer Agent 剧本生成成功")
            else:
                logger.warning("Writer Agent 生成失败，降级到 Fallback 模式")
        else:
            logger.warning("Editor Agent 生成失败，降级到 Fallback 模式")

    if not script:
        script = _build_fallback(brief, episode_date, podcast_title)

    script = clean_tts_text(script)
    script = _replace_banned_words(script, banned_words)
    script = _normalize_host_tags(script)

    warnings: list[str] = []

    found_banned = check_banned_words(script, banned_words)
    if found_banned:
        warnings.append(f"仍含禁用词: {found_banned}")

    # 简单统计Host A和Host B的出场次数
    stripped_script = script.strip()
    is_ssml = (
        stripped_script.startswith("<speak")
        or "<speak" in stripped_script
        or "<voice" in stripped_script
    )

    if is_ssml:
        host_a_count = (
            script.count('name="zh-CN-YunxiNeural"')
            + script.count('name="zh-CN-YunjianNeural"')
            + script.count('name="zh-CN-YunyangNeural"')
        )
        host_b_count = script.count('name="zh-CN-XiaoxiaoNeural"')
        total_turns = host_a_count + host_b_count
        if host_a_count == 0 or host_b_count == 0:
            warnings.append("生成剧本未严格包含 <voice> 切换标记")
    else:
        host_a_count = script.count("[Host A]")
        host_b_count = script.count("[Host B]")
        total_turns = host_a_count + host_b_count
        if host_a_count == 0 or host_b_count == 0:
            warnings.append("生成剧本未严格包含 [Host A] 和 [Host B] 双人对谈标记")

    logger.info(
        "双人剧本生成完毕 (Mode: %s), 对话回合数: %d 轮 (A:%d, B:%d)",
        mode_used,
        total_turns,
        host_a_count,
        host_b_count,
    )
    return script, warnings


def _normalize_host_tags(text: str) -> str:
    """清理并确保 Host 标签格式绝对标准，形如 `[Host A] 说话内容`。如果是 SSML 格式则不进行转换。"""
    stripped_text = text.strip()
    if stripped_text.startswith("<speak") or "<speak" in stripped_text or "<voice" in stripped_text:
        return text

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 宽泛匹配，防止大模型加了一些冒号或者空格: [Host A]: 或者 Host A:
        line = re.sub(r"^\[?Host\s*A\]?:?\s*", "[Host A] ", line, flags=re.IGNORECASE)
        line = re.sub(r"^\[?Host\s*B\]?:?\s*", "[Host B] ", line, flags=re.IGNORECASE)

        # 移除段首原本系统的 mood 标记残留
        line = RE_MOOD_TAG.sub("", line)

        # 如果依然没有包含标准的前缀，强行挂给 Host A 作为过渡，避免 TTS 崩溃
        if not line.startswith("[Host A]") and not line.startswith("[Host B]"):
            line = f"[Host A] {line}"

        if line.strip():
            result.append(line)

    return "\n\n".join(result) + "\n"


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
            context = story.get("context", {})
            summaries = context.get("factual_summary", [])
            total = story.get("total_score", 0)

            lines.append(f"### {title}")
            lines.append("")
            if summaries:
                for s in summaries:
                    lines.append(f"- {s}")
                lines.append("")
            if story.get("items"):
                lines.append("**来源链接：**")
                lines.append("")
                for item in story.get("items", [])[:5]:
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
        role_emoji = story.get("role_emoji", "")
        if story.get("items"):
            link = story.get("items", [])[0].get("link", "")
            source = _esc(story.get("items", [])[0].get("source_name", ""))
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
