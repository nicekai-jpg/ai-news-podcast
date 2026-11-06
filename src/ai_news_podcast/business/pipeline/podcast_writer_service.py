"""Stage 3 — 脚本生产模块（LLM 多 Agent 融合写作版）

职责：episode_brief → 主编Agent选定头条/大纲 → 撰稿Agent转化为双人对白播客脚本（Host A, Host B）。
英文内容在 LLM 阶段直接翻译为中文，输出格式严格遵循对话播客要求，供后续混音使用。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from ai_news_podcast.business.pipeline.llm_client_service import call_llm as _call_llm
from ai_news_podcast.business.pipeline.material_service import (
    build_material_text as _build_material_text,
)
from ai_news_podcast.business.pipeline.material_service import build_radar_text as _build_radar_text
from ai_news_podcast.data.prompts_dao import (
    DEFAULT_BANNED_WORDS,
    build_editor_prompt,
    build_writer_prompt,
)
from ai_news_podcast.data.text_utils_dao import (
    RE_MOOD_TAG,
    clean_tts_text,
    contains_thinking_process,
    script_quality_stats,
)

logger = logging.getLogger(__name__)

# 章节标记由 writer 按 Stage 3 提示词单独成行输出。clean_tts_text 会把非 Host 的
# 中括号标记整段删掉（RE_NON_HOST_BRACKET），因此清洗前先用哨兵占位保住标题，
# 再由 _normalize_host_tags 还原成 `[CHAPTER] 标题`（不补 [Host A] 前缀）。
_CHAPTER_LINE_RE = re.compile(r"^\[CHAPTER\]\s*(.+)$")
_CHAPTER_SENTINEL = "__CHAPTER_MARKER__"
_CHAPTER_SENTINEL_RE = re.compile(r"^" + _CHAPTER_SENTINEL + r"\s*(.+)$")

# 双主持对话：任一主持人占比低于该阈值视为口径失衡（2026-11-05 事故：Host A 228 句 / Host B 4 句）
_MIN_HOST_RATIO = 0.15


def _host_balance_ok(script: str, min_ratio: float = _MIN_HOST_RATIO) -> bool:
    """Host A/B 轮次是否大致均衡；没有 Host 标记时返回 True，交给「无标记」分支处理。"""
    host_a = script.count("[Host A]")
    host_b = script.count("[Host B]")
    total = host_a + host_b
    if total == 0:
        return True
    return min(host_a, host_b) / total >= min_ratio


# 质量闸门：长度、轮次、每位主持人的字符占比、未标记行占比
_MIN_SCRIPT_CHARS = 1200
_MAX_SCRIPT_CHARS = 5200
_MIN_TURNS = 10
_MIN_HOST_CHAR_RATIO = 0.25
_MAX_UNMARKED_RATIO = 0.2


def _host_char_counts(script: str) -> tuple[int, int]:
    """按 [Host X] 标记粗略统计每位主持人的实际台词字符数（去空白）。"""
    counts = {"A": 0, "B": 0}
    current: str | None = None
    for raw_line in script.splitlines():
        line = raw_line.strip()
        match = re.match(r"^\[Host ([AB])\]\s*(.*)$", line)
        if match:
            current = match.group(1)
            line = match.group(2)
        if current:
            counts[current] += len(re.sub(r"\s+", "", line))
    return counts["A"], counts["B"]


def _script_stats(script: str) -> dict[str, float]:
    host_a, host_b = _host_char_counts(script)
    total = host_a + host_b
    unmarked = 0
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[CHAPTER]"):
            continue
        if not re.match(r"^\[Host [AB]\]", line):
            unmarked += 1
    lines = [ln for ln in script.splitlines() if ln.strip()]
    return {
        "chars": len(re.sub(r"\s+", "", script)),
        "turns": script.count("[Host A]") + script.count("[Host B]"),
        "host_a_chars": host_a,
        "host_b_chars": host_b,
        "host_char_ratio": (min(host_a, host_b) / total) if total else 0.0,
        "unmarked_ratio": (unmarked / len(lines)) if lines else 0.0,
        "chapters": script.count("[CHAPTER]"),
    }


def _script_looks_valid(script: str) -> bool:
    """Writer 输出是否像一份可用的双主持对话（而非计划/单人口播/短稿/超长稿）。"""
    if contains_thinking_process(script) or not _host_balance_ok(script):
        return False
    stats = _script_stats(script)
    if stats["turns"] < _MIN_TURNS:
        return False
    if not _MIN_SCRIPT_CHARS <= stats["chars"] <= _MAX_SCRIPT_CHARS:
        return False
    if stats["host_char_ratio"] < _MIN_HOST_CHAR_RATIO:
        return False
    return stats["unmarked_ratio"] <= _MAX_UNMARKED_RATIO


_MAX_SALVAGE_CHARS = 5200


def _salvage_dialogue(script: str) -> str:
    """模型常把「写稿计划」和正式台词混在一起输出（2026-11-06 实测 19k 字符里混着
    `Need no ...`/`Let's count rough.` 这类规划行，正文被切成很多段）。

    这里取**最长的一段连续对白**（[Host A]/[Host B]/[CHAPTER] 行），它通常就是模型
    最后写出的完整成稿；碎片化的草稿段不采用，宁可走 fallback 也不拼一份前后错乱的稿。
    """
    runs: list[list[str]] = []
    current: list[str] = []
    for raw_line in str(script).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\[(?:Host [AB]|CHAPTER)\]", line):
            current.append(line)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return ""
    return "\n".join(max(runs, key=len)) + "\n"


def _trim_script(script: str, max_chars: int = _MAX_SALVAGE_CHARS) -> str:
    """超长稿按行截断到目标长度，并补结束语；优先在章节/整行边界切。"""
    kept: list[str] = []
    total = 0
    for raw_line in script.splitlines():
        if total + len(raw_line) + 1 > max_chars:
            break
        kept.append(raw_line)
        total += len(raw_line) + 1
    return _ensure_proper_ending("\n".join(kept) + "\n")


def _accept_script(script: str) -> str:
    """接受 LLM 稿前的最后一道整理：超长就截断，保证有结束语。"""
    if len(re.sub(r"\s+", "", script)) > _MAX_SALVAGE_CHARS:
        return _trim_script(script)
    return script


def _protect_chapter_markers(text: str) -> str:
    """把独立的 [CHAPTER] 标题行换成清洗安全的哨兵，避免标题被 clean_tts_text 删掉。"""
    protected: list[str] = []
    for raw_line in text.split("\n"):
        match = _CHAPTER_LINE_RE.match(raw_line.strip())
        if match:
            protected.append(f"{_CHAPTER_SENTINEL} {match.group(1).strip()}")
        else:
            protected.append(raw_line)
    return "\n".join(protected)


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


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# 纯模板兜底（LLM 不可用时）
# ---------------------------------------------------------------------------


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _pick_zh_summary(story: dict[str, Any]) -> str:
    """从上下文里挑中文摘要（最多两句）；没有中文就返回空（中文主持不要念英文原文）。"""
    summaries = (story.get("context") or {}).get("factual_summary") or []
    picked: list[str] = []
    for item in summaries:
        text = str(item).strip()
        if text and _has_cjk(text):
            picked.append(text)
        if len(picked) >= 2:
            break
    return " ".join(picked)


def _pick_zh_background(story: dict[str, Any]) -> str:
    """取中文背景补充（没有中文就返回空）。"""
    background = (story.get("context") or {}).get("historical_background") or ""
    text = str(background).strip()
    if not text or not _has_cjk(text):
        return ""
    return text[:160]


def _build_fallback(
    brief: dict[str, Any],
    episode_date: datetime,
    podcast_title: str,
) -> str:
    """无 LLM / 输出不合格时的双人对谈兜底：带章节、中文摘要优先、无编号腔、含雷达。

    2026-11-06 的教训：旧模板只有 ~500 字符/12 轮，TTS 出来 1 分钟，听众体验极差；
    这里扩到 5 条素材、每条 2 轮，保证兜底稿也有几分钟可听的正文。
    """
    stories = [
        story
        for story in (brief.get("stories") or [])
        if isinstance(story, dict) and story.get("role") != "skip"
    ]

    lines: list[str] = ["[CHAPTER] 开场"]
    lines.append(
        f"[Host A] 听众朋友们好，欢迎收听{podcast_title}，我是林夏。今天是{_cn_date(episode_date)}。"
    )
    lines.append("[Host B] 大家好，我是陈默。今天消息不少，我们从最值得展开的几条开始，挑重点聊。")

    if stories:
        lines.append("[CHAPTER] 今日头条")
        first = stories[0]
        first_title = str(first.get("representative_title") or "").strip()
        lines.append(f"[Host A] 先看今天的头条：{first_title}。")
        summary = _pick_zh_summary(first)
        if summary:
            lines.append(f"[Host B] 简单梳理一下：{summary}")
        else:
            lines.append("[Host B] 这条目前披露的信息还比较有限，我们等更多细节出来再看。")

        lines.append("[CHAPTER] 其他值得关注")
        for story in stories[1:5]:
            title = str(story.get("representative_title") or "").strip()
            if not title:
                continue
            summary = _pick_zh_summary(story)
            background = _pick_zh_background(story)
            lines.append(f"[Host A] 接下来这条是：{title}。")
            if summary:
                lines.append(f"[Host B] 我的理解是：{summary}")
            if background:
                lines.append(f"[Host A] 补充一点背景：{background}")
            if not summary and not background:
                lines.append("[Host B] 这条的细节还在陆续放出，先记下来，后面再跟。")

    radar = brief.get("radar") or {}
    pick_repo = str((radar.get("meta") or {}).get("pick_repo") or "").strip()
    if pick_repo:
        lines.append("[CHAPTER] 项目雷达")
        lines.append(f"[Host A] 项目雷达今天主推 {pick_repo}，感兴趣可以去 GitHub 看看。")
        lines.append("[Host B] 好，这个先记下，回头我们细聊。")

    lines.append("[CHAPTER] 收尾")
    lines.append("[Host B] 今天的分享就到这里，信息不少，建议收藏慢慢看。")
    lines.append("[Host A] 感谢收听，我们明天见。")
    lines.append("[Host B] 拜拜！")
    return "\n".join(lines) + "\n"


_HARD_CONSTRAINT = (
    "\n\n【输出格式硬约束】只输出逐行 [Host A]/[Host B] 对话，"
    "两位主持人交替发言、轮次大致均衡（各占一半）；总字数控制在 1500-4500 字符；"
    "不要输出任何思考、计划、分析、英文说明、格式要求复述或自我检查；"
    "不要出现 URL、Markdown 标记或编号清单。"
)


def _pick_writer_script(
    writer_prompt: str, llm_cfg: dict[str, Any]
) -> tuple[str | None, str, str | None]:
    """跑 writer 并挑出可用脚本：原文 → 抽出对白 → 硬约束重试（原文/抽对白）。

    返回 (script | None, mode, fallback_reason)。
    2026-11-06 实测：模型一次输出 19k 字符，把 `Need no ...`/`Let's count rough.`
    这类写稿计划混在 [Host A] 行之间。直接整篇丢弃会退化成 1 分钟模板稿，
    所以先抽出真正的 [Host A]/[Host B]/[CHAPTER] 行再判定。
    """
    raw_writer = _call_llm(writer_prompt, dict(llm_cfg, temperature=0.85))
    if not raw_writer:
        return None, "fallback", "Writer Agent 生成失败，已降级到 Fallback 模板"
    logger.info("Writer Agent 播客生成成功")

    candidate = _accept_script(raw_writer)
    if _script_looks_valid(candidate):
        return candidate, "Multi-Agent", None

    salvaged = _accept_script(_salvage_dialogue(raw_writer))
    if _script_looks_valid(salvaged):
        logger.warning("Writer 输出混有计划文本，已抽出对话行后采用")
        return salvaged, "Multi-Agent (salvaged)", None

    stats = _script_stats(raw_writer)
    logger.warning(
        "Writer 输出不合格 (chars=%s, turns=%s, ratio=%.2f, unmarked=%.2f, chapters=%s)",
        stats["chars"],
        stats["turns"],
        stats["host_char_ratio"],
        stats["unmarked_ratio"],
        stats["chapters"],
    )
    logger.warning("Writer 输出疑似思考过程或主持人口径失衡，带硬约束重试一次")
    raw_retry = _call_llm(writer_prompt + _HARD_CONSTRAINT, dict(llm_cfg, temperature=0.7))
    if raw_retry:
        for mode, retry_candidate in (
            ("Multi-Agent (retry)", _accept_script(raw_retry)),
            ("Multi-Agent (retry salvaged)", _accept_script(_salvage_dialogue(raw_retry))),
        ):
            if retry_candidate and _script_looks_valid(retry_candidate):
                logger.info("Writer Agent 重试成功")
                return retry_candidate, mode, None

    logger.warning("Writer 重试仍不合格，降级到 Fallback 模式")
    return None, "fallback", "LLM 输出疑似思考过程或主持人口径失衡，已降级到 Fallback 模板"


def generate_podcast(
    brief: dict[str, Any],
    *,
    episode_date: datetime,
    podcast_title: str = "今日AI看点",
    writer_cfg: dict[str, Any] | None = None,
    llm_cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """生成播客脚本 (双Agent双播客机制)：Editor 提纲 → Writer 对谈；失败降级模板。"""
    cfg = writer_cfg or {}
    style_cfg = cfg.get("style", {})
    banned_words = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    llm_cfg = llm_cfg or {}

    material = _build_material_text(brief, max_stories=5)
    radar_material = _build_radar_text(brief.get("radar"))
    logger.info("素材文本 %d 字符，准备启动 Multi-Agent 编剧流程", len(material))

    script: str | None = None
    mode_used = "fallback"
    fallback_reason: str | None = None

    if material.strip():
        # --- Node 1: Editor ---
        editor_prompt = build_editor_prompt(material, episode_date, radar_material=radar_material)
        raw_editor = _call_llm(editor_prompt, llm_cfg)
        if raw_editor:
            logger.info("Editor Agent 生成大纲成功，准备注入 Writer 节点")
            writer_prompt = build_writer_prompt(
                raw_editor,
                episode_date,
                podcast_title,
                style_cfg,
                has_radar="## 项目雷达" in raw_editor,
            )
            script, mode_used, fallback_reason = _pick_writer_script(writer_prompt, llm_cfg)
        else:
            logger.warning("Editor Agent 生成失败，降级到 Fallback 模式")
            fallback_reason = "Editor Agent 生成失败，已降级到 Fallback 模板"

    if not script:
        script = _build_fallback(brief, episode_date, podcast_title)

    script, warnings, _ = _post_process_script(
        script, brief, episode_date, podcast_title, banned_words, mode_used
    )
    if fallback_reason:
        warnings.append(fallback_reason)
    return script, warnings


def script_quality(script: str) -> dict:
    """脚本质量快照：normal / fallback / short / broken（供 publish 写入 episodes.json/digest）。"""
    stats = script_quality_stats(script)
    if stats.get("mode") != "fallback" and not _script_looks_valid(script):
        stats["mode"] = "broken"
    return stats


def _post_process_script(
    script: str,
    brief: dict[str, Any],
    episode_date: datetime,
    podcast_title: str,
    banned_words: list[str],
    mode_used: str,
) -> tuple[str, list[str], str]:
    """清洗、验证脚本，必要时回退到 fallback 模板。"""
    # clean_tts_text 会删掉 [CHAPTER] 的中括号，先换成哨兵保住章节标题
    script = clean_tts_text(_protect_chapter_markers(script))
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

    # 双主持口径失衡（一人几乎不发言）同样是坏稿，降级到 Fallback，别让垃圾稿进 TTS
    if not _host_balance_ok(script):
        logger.warning(
            "Host A/B 轮次严重失衡 (A:%d, B:%d)，强制降级到 Fallback 模板",
            host_a_count,
            host_b_count,
        )
        script = _build_fallback(brief, episode_date, podcast_title)
        script = clean_tts_text(_protect_chapter_markers(script))
        script = _replace_banned_words(script, banned_words)
        script = _normalize_host_tags(script)
        script = _ensure_proper_ending(script)
        mode_used = "fallback (forced: host imbalance)"
        warnings.append("LLM 输出主持人口径严重失衡，已强制降级到 Fallback 模板")
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
        # 章节行原样保留（不补 [Host A] 前缀，也不进 TTS 台词）；哨兵形式是
        # clean_tts_text 之前的占位，这里统一还原成标准 [CHAPTER] 标题行。
        chapter_match = _CHAPTER_LINE_RE.match(line) or _CHAPTER_SENTINEL_RE.match(line)
        if chapter_match:
            result.append(f"[CHAPTER] {chapter_match.group(1).strip()}")
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

    # 脚本被截断：先丢掉最后半句（模型 max_tokens 截断时常见 "[Host A] 游戏这边"），
    # 再追加标准结束语，避免出现「半句话 + 道别」的听感断裂。
    lines = stripped.splitlines()
    while lines:
        tail = lines[-1].strip()
        if not tail:
            lines.pop()
            continue
        if tail.startswith(("[Host A]", "[Host B]", "[CHAPTER]")) and not re.search(
            r"[。！？.!?…”」』]$", tail
        ):
            lines.pop()
            continue
        break
    logger.warning("脚本被截断，自动追加标准结束语")
    closing_lines = (
        "\n[Host A] 好了听众朋友们，以上就是今天的今日AI看点。感谢收听，我们明天见！"
        "\n[Host B] 拜拜！"
    )
    return "\n".join(lines) + closing_lines + "\n"
