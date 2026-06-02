"""Stage 3 — 脚本生产模块（LLM 多 Agent 融合写作版）

职责：episode_brief → 主编Agent选定头条/大纲 → 撰稿Agent转化为双人对白播客脚本（Host A, Host B）。
英文内容在 LLM 阶段直接翻译为中文，输出格式严格遵循对话剧本要求，供后续混音使用。
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


def _sanitize_for_tts(text: str) -> str:
    """清洗 LLM 输出中 TTS 不友好的残留内容。"""
    if not text:
        return ""
    # 1. 处理转义字符（如模型误输出的 \\n）
    text = text.replace("\\n", "\n")

    # 2. 清除特定标记和表情
    text = re.sub(r"\[(?:FACT|INFERENCE|OPINION)\]\s*", "", text)
    text = re.sub(r"[（(][^）)]{0,10}(?:doge|狗头|笑|手动|滑稽|哭|捂脸)[^）)]{0,5}[）)]", "", text)
    text = re.sub(r"[「」『』【】]", "", text)

    stripped_text = text.strip()
    is_ssml = (
        stripped_text.startswith("<speak") or "<speak" in stripped_text or "<voice" in stripped_text
    )
    if not is_ssml:
        text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"[（(]\s*[）)]", "", text)

    # 3. 压缩重复标点
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。.]{2,}", "。", text)

    # 4. 规范化空格和换行：不要把换行符全删了
    # 先把行首行尾空格去掉
    lines = [line.strip() for line in text.split("\n")]
    # 过滤掉过多连续空格，但保留非空行
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    # 压缩过多连续换行（最多保留两个，即一个空行间隔）
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# 新闻素材 → LLM prompt 构建
# ---------------------------------------------------------------------------



def _build_material_text(brief: dict[str, Any], max_stories: int = 8) -> str:
    """把 episode_brief 中评分最高的新闻素材整理为结构化文本。

    数据基础层（fetch → dedup → cluster → score）已保证每个「新闻簇」
    是经过质量控制的可用数据：每簇代表一个独立事件，内含多家媒体的
    报道汇总。此处直接按总分排序取前 max_stories 条，不做额外过滤。
    """
    stories = brief.get("stories", [])
    active: list[dict[str, Any]] = [
        s for s in stories if isinstance(s, dict) and s.get("role") != "skip"
    ]
    active.sort(key=lambda s: s.get("total_score", 0), reverse=True)
    top_active = active[:max_stories]

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


def _build_editor_prompt(
    material: str,
    episode_date: datetime,
) -> str:
    """第一阶段：主编 Agent，负责精简素材和定调。"""
    date_str = _cn_date(episode_date)
    return f"""你是顶级科技播客的「主编」(Editor)。今天是{date_str}。
你的任务是阅读下方的新闻素材池，筛选并组织出一份精简的「今日播报大纲」。

## 操作要求
1. **提炼金句 (Thesis)**：用一句话总结今天的整体科技趋势或最震撼的事件。
2. **挑选头条 (Headline)**：选出一个最有深度、最值得讨论的新闻作为今日头条。
3. **精选快讯 (Quick News)**：选出 3-5 条次要但不同主题、不同来源的有趣新闻作为快讯（涵盖不同的领域或主题，保证播客内容丰富多元）。
4. 忽略毫无价值或者完全重复的素材。

## 新闻素材
{material}

## 输出格式
请务必直接输出合法的 JSON 格式，不要包含 ```json 等标记，结构如下：
{{
  "thesis": "一句话整体总结",
  "headline": {{
    "title": "头条新闻标题",
    "summary": "深度的摘要，为什么这很重要，背后有什么深刻影响（控制在 150 字以内）"
  }},
  "quick_news": [
    {{
      "title": "快讯标题 1",
      "summary": "一句话介绍（50字以内）"
    }},
    ...
  ]
}}"""


def _build_writer_prompt(
    editor_plan_json: str,
    episode_date: datetime,
    podcast_title: str,
    style_cfg: dict[str, Any],
) -> str:
    """第二阶段：撰稿 Agent，将主编定下的大纲转化为双人对口相声/对谈剧本。"""
    banned = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    banned_str = "、".join(banned)
    date_str = _cn_date(episode_date)

    return f"""你是「{podcast_title}」的金牌撰稿人。今天是{date_str}。
你的任务是根据主编给出的【今日播报大纲】，写一份**双人对谈格式的播客录音剧本**。

## 主持人详细人设与角色分工（“主持人 + 嘉宾” 模式）
这两位主持人是常年合作的科技挚友，对话风格松弛、默契、充满温度，严禁任何机械化播报：
- **晓晓 [Host B] (主持人/Anchor) —— “体验派产品经理”**：
  - *音色与标签*：使用 `<voice name="zh-CN-XiaoxiaoNeural">`，内容中自称为“晓晓”。
  - *背景人设*：前互联网 AI 产品经理，热爱尝试各种新潮 AI 工具（ChatGPT、Midjourney、Cursor 等），极为关注“用户体验 (UX)”和“日常落地场景”。
  - *职责分工*：主导节目控场与流程推进，负责每一章新闻的开场与核心事实引入。当博文讲出晦涩术语时，她要及时打断追问（例如：“等等博文，你刚才提到的 MoE 架构，能给听众翻译翻译吗？”）；同时她是“神比喻担当”，负责把复杂技术转化为极度接地气的生活化类比。
  - *性格态度*：活泼机敏、好奇心强、心直口快，面对科技突破会发出真诚的感叹（如“哇”、“天哪真假的”），同时对隐私侵扰或技术泡沫持实用主义担忧。
- **博文 [Host A] (技术嘉宾/Expert) —— “硬核逻辑极客”**：
  - *音色与标签*：使用 `<voice name="zh-CN-YunxiNeural">`，内容中自称为“博文”。
  - *背景人设*：前大厂资深算法科学家与系统架构师，了解主流模型的演进史，深知哪些是换汤不换药的炒作、哪些是工程落地中的真正痛点（算力限制、显存占用、部署成本等）。
  - *职责分工*：负责在晓晓抛出新闻和问题后，进行深度的技术解构、商业战略剖析和行业格局研判。
  - *性格态度*：沉稳理性、儒雅风趣、审慎乐观。对重大技术进展充满热情，但对“PPT产品”持有极客特有的温和幽默吐槽（如“这就像是给旧轮子涂上了新油漆”），对晓晓充满信任，常用“晓晓，你说得没错，其实……”、“你这个比喻非常恰当”。

## 剧本编写核心要求
1. **互动默契（Chemical Reactions）**：
   - 绝非死板的一问一答，博文在说话时，晓晓会有短平快的捧哏或轻微回应（“嗯，对”、“没错”）；晓晓在播报时，博文也会及时搭腔（“这个确实很火”）。
   - 晓晓会吐槽博文“博文老师，你又开始讲公式了，快醒醒”，博文会笑着承认。博文也会夸奖晓晓的直觉“晓晓这个产品经理的雷达果然灵敏”。
   - 两人要在对话中自然、频繁地直呼对方名字（“博文”、“晓晓”），展现亲密老友的闲聊质感。
2. **播报与互动流程**：
   - 每一条新闻，都由主持人**晓晓**率先发起，播报新闻标题与基本动态，然后向**博文**发起提问（例如：“博文，这技术真的有这么神吗？”或“这个功能对我们普通人有什么用呢？”）。
   - **博文**作为嘉宾接话，进行深度技术原理解析和行业背景补充，把话题引向深处。
   - **晓晓**对博文的解答给予回馈、捧哏或做生动的总结性说明，然后顺滑地引导进入下一条新闻。
3. **口语化与互动感**：使用极其自然的口语化表达（如“确实”、“你想啊”、“哎我看到个很有意思的”）。晓晓经常会有情绪感叹词。
4. **格式极其严格**：输出必须符合标准的 SSML (Speech Synthesis Markup Language) XML 格式。
   - 使用 `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">` 作为根元素。
   - 每一段台词必须使用对应的 `<voice name="...">` 元素包裹。
5. **内容结构**：
   - **开场引言**：活泼互动，晓晓和博文互相打招呼，由晓晓引出今日主题 Thesis（金句总结）。
   - **深挖头条**：晓晓播报头条新闻，博文介绍核心事实与技术深度，晓晓负责好奇发问“这到底意味着什么？”或做生动比喻。
   - **快报环节**：由晓晓快速串起各个快讯，博文对每个快讯做1-2句核心点评，两人默契配合。
   - **结尾总结**：两人默契总结，简短说再见。
6. **英文处理**：英文专有名词尽可能转为流畅跟读的中文，如果不影响阅读也可保留简单单词。
7. **禁用词**：坚决避免使用以下词汇（会显得很假）：{banned_str}。不要用“今天的第一条新闻是”这种僵硬的罗列。

## 主编大纲 (JSON)
{editor_plan_json}

## 正确的输出格式示例（必须是合法的 XML/SSML，严格包含在 <speak> 中）：
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
  <voice name="zh-CN-XiaoxiaoNeural">
    听众朋友大家好，欢迎收听{podcast_title}，今天是{date_str}。我是主持人晓晓。
  </voice>
  <voice name="zh-CN-YunxiNeural">
    大家好，我是博文。
  </voice>
  <voice name="zh-CN-XiaoxiaoNeural">
    哎，博文，你注意到今天科技圈的头条了吗？谷歌今天发布了全新模型。
  </voice>
  <voice name="zh-CN-YunxiNeural">
    嗯，我关注了。这次更新在底层架构上做出了非常大的改变，特别是……
  </voice>
</speak>

请直接输出符合 SSML 标准的 XML 全文，**不要**包含任何 Markdown 标题、说明文字，也**绝对不要**包含 ```xml 这样的 Markdown 代码块标记。根元素必须是 <speak>。"""


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

        except Exception as e:
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
        if i >= 4:
            break
        title = story.get("representative_title", "")
        context = story.get("context", {})
        summaries = context.get("factual_summary", [])
        summary = summaries[0] if summaries else "细节不详，但极其重要。"

        if i == 0:
            lines.append(f"[Host A] 咱们先聊聊今天最大的头条。那就是，{title}。")
            lines.append("[Host B] 晓晓也一直在关注这个，博文，这个事影响真的很大吗？")
            lines.append(f"[Host A] 确实。简单来说，{summary}")
        else:
            lines.append(f"[Host B] 嗯，除了这个，我看到 {title} 也有新动态。")
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
        editor_prompt = _build_editor_prompt(material, episode_date)
        raw_editor = _call_llm(editor_prompt, llm_cfg)

        if raw_editor:
            logger.info("Editor Agent 生成大纲成功，准备注入 Writer 节点")
            # --- Node 2: Writer ---
            writer_prompt = _build_writer_prompt(raw_editor, episode_date, podcast_title, style_cfg)
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

    script = _sanitize_for_tts(script)
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
        host_a_count = script.count('name="zh-CN-YunxiNeural"')
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
        line = re.sub(r"\[mood:\w+\]\s*", "", line)

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
