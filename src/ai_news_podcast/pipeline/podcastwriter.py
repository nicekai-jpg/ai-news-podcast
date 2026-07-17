"""Stage 3 — 脚本生产模块（LLM 多 Agent 融合写作版）

职责：episode_brief → 主编Agent选定头条/大纲 → 撰稿Agent转化为双人对白播客脚本（Host A, Host B）。
英文内容在 LLM 阶段直接翻译为中文，输出格式严格遵循对话播客要求，供后续混音使用。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from ai_news_podcast.pipeline.llm_client import call_llm as _call_llm
from ai_news_podcast.pipeline.material import build_material_text as _build_material_text
from ai_news_podcast.prompts import DEFAULT_BANNED_WORDS, build_editor_prompt, build_writer_prompt
from ai_news_podcast.text_utils import RE_MOOD_TAG, clean_tts_text, contains_thinking_process

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

# ---------------------------------------------------------------------------
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
        f"[Host A] 听众朋友们好，欢迎收听{podcast_title}，我是苏晴。今天是{_cn_date(episode_date)}。"
    )
    lines.append("[Host B] 大家好，我是周航。今天这AI圈的瓜，那可是相当有意思。")

    for i, story in enumerate(active):
        if i >= 5:
            break
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        summary = summaries[0] if summaries else "细节不详，但极其重要。"

        if i == 0:
            lines.append(f"[Host A] 咱们先聊聊今天第一个大头条。那就是，{title}。")
            lines.append("[Host B] 苏晴也一直在关注这个，周航，这个事影响真的很大吗？")
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


def generate_podcast(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str = "AI 每日先锋",
    writer_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """
    生成播客脚本 (双Agent双播客机制)。
    Step 1: Editor Agent 提炼大纲
    Step 2: Writer Agent 编写对谈播客
    失败时退回退化版的双人对谈模板。
    """
    cfg = writer_cfg or {}
    style_cfg = cfg.get("style", {})
    banned_words = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    llm_cfg = llm_cfg or {}

    material = _build_material_text(brief, max_stories=5)
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
                logger.info("Writer Agent 播客生成成功")
                # 检测 LLM 是否输出了思考过程而非对话
                if contains_thinking_process(script):
                    logger.warning("Writer Agent 输出了思考过程而非对话，降级到 Fallback 模式")
                    script = None
                    mode_used = "fallback"
            else:
                logger.warning("Writer Agent 生成失败，降级到 Fallback 模式")
        else:
            logger.warning("Editor Agent 生成失败，降级到 Fallback 模式")

    if not script:
        script = _build_fallback(brief, episode_date, podcast_title)

    script, warnings, _ = _post_process_script(
        script, brief, episode_date, podcast_title, banned_words, mode_used
    )
    return script, warnings


def _post_process_script(
    script: str,
    brief: dict[str, Any],
    episode_date: datetime,
    podcast_title: str,
    banned_words: list[str],
    mode_used: str,
) -> tuple[str, list[str], str]:
    """清洗、验证脚本，必要时回退到 fallback 模板。"""
    script = clean_tts_text(script)
    if not script.strip():
        logger.warning("clean_tts_text 后脚本为空，强制降级到 Fallback 模板")
        script = _build_fallback(brief, episode_date, podcast_title)
        mode_used = "fallback (forced: empty after cleaning)"
    script = _replace_banned_words(script, banned_words)
    script = _normalize_host_tags(script)
    script = _ensure_proper_ending(script)

    warnings: list[str] = []
    host_a_count = script.count("[Host A]")
    host_b_count = script.count("[Host B]")
    total_turns = host_a_count + host_b_count

    if total_turns == 0:
        logger.warning("脚本生成后不含任何 [Host A]/[Host B] 标记，强制降级到 Fallback 模板")
        script = _build_fallback(brief, episode_date, podcast_title)
        script = clean_tts_text(script)
        script = _replace_banned_words(script, banned_words)
        script = _normalize_host_tags(script)
        script = _ensure_proper_ending(script)
        mode_used = "fallback (forced: no host markers)"
        warnings.append("LLM 输出不含 Host 标记，已强制降级到 Fallback 模板")
        host_a_count = script.count("[Host A]")
        host_b_count = script.count("[Host B]")
        total_turns = host_a_count + host_b_count

    found_banned = check_banned_words(script, banned_words)
    if found_banned:
        warnings.append(f"仍含禁用词: {found_banned}")

    if host_a_count == 0 or host_b_count == 0:
        warnings.append("生成播客未严格包含 [Host A] 和 [Host B] 双人对谈标记")

    logger.info(
        "双人播客生成完毕 (Mode: %s), 对话回合数: %d 轮 (A:%d, B:%d)",
        mode_used,
        total_turns,
        host_a_count,
        host_b_count,
    )
    return script, warnings, mode_used


def _normalize_host_tags(text: str) -> str:
    """清理并确保 Host 标签格式绝对标准，形如 `[Host A] 说话内容`。"""
    lines = text.split("\n")
    result: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
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


def _ensure_proper_ending(script: str) -> str:
    """检测脚本是否被截断，如果是则追加标准结束语。"""
    stripped = script.strip()
    if not stripped:
        return script

    # 检查是否以完整的结束语结尾 (忽略尾部标点符号)
    endings = ["拜拜", "再见", "明天见", "下期见"]
    check_str = re.sub(r"[^\w\u4e00-\u9fff]+$", "", stripped)
    has_ending = any(check_str.endswith(e) for e in endings)

    if has_ending:
        return script

    # 脚本被截断，追加标准结束语
    logger.warning("脚本被截断，自动追加标准结束语")
    closing_lines = """\n[Host A] 好了听众朋友们，以上就是今天的AI每日先锋。感谢收听，我们明天见！\n[Host B] 拜拜！"""
    return stripped + closing_lines + "\n"
