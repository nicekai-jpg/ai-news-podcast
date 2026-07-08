"""LLM prompt templates for the multi-agent script generation pipeline.

Contains the Editor Agent and Writer Agent prompt templates, along with
builder functions that inject runtime parameters (date, material, etc.).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# 禁用词（从 podcastwriter 提取，供 prompt 构建和脚本清洗共用）
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


def _cn_date(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日"


# ---------------------------------------------------------------------------
# Editor Agent prompt
# ---------------------------------------------------------------------------

EDITOR_USER_TEMPLATE = """你是顶级科技播客的「主编」(Editor)。今天是{date_str}。
你的任务是阅读下方的新闻素材池，筛选并组织出一份精简的「今日播报大纲」。

## 操作要求
1. **提炼金句 (Thesis)**：用一句话总结今天的整体科技趋势或最震撼的事件。
2. **挑选双头条 (Headlines)**：选出 2 个最有深度、最值得深度讨论的新闻作为今日的双头条（分别作为头条1、头条2）。
3. **精选快讯 (Quick News)**：从其余素材中选出恰好 3 条不同主题、不同来源的有趣新闻作为快讯，保证播客内容丰富多元。
4. 忽略毫无价值或者完全重复的素材。

## 新闻素材
{material}

## 输出格式
请直接输出 Markdown 格式的大纲，不要包含 Markdown 代码块标记，格式如下：

# 今日播报大纲

## 金句
一句话整体总结。

## 头条 1
- **标题**：头条新闻 1 标题
- **摘要**：深度的摘要，为什么这很重要，背后有什么深刻影响（控制在 150 字以内）

## 头条 2
- **标题**：头条新闻 2 标题
- **摘要**：深度的摘要，为什么这很重要，背后有什么深刻影响（控制在 150 字以内）

## 快讯
1. **快讯标题 1**：一句话介绍（50字以内）
2. **快讯标题 2**：一句话介绍（50字以内）
3. **快讯标题 3**：一句话介绍（50字以内）"""


def build_editor_prompt(
    material: str,
    episode_date: datetime,
) -> str:
    """第一阶段：主编 Agent，负责精简素材和定调。"""
    date_str = _cn_date(episode_date)
    return EDITOR_USER_TEMPLATE.format(date_str=date_str, material=material)


# ---------------------------------------------------------------------------
# Writer Agent prompt
# ---------------------------------------------------------------------------

WRITER_USER_TEMPLATE = """你是「{podcast_title}」的金牌撰稿人。今天是{date_str}。
你的任务是根据主编给出的【今日播报大纲】，写一份**双人对谈格式的播客录音剧本**。

## 主持人详细人设与角色分工
这两位主持人是常年搭档的科技播客主播，对话风格松弛、默契、充满温度，严禁任何机械化播报。

- **苏晴 [Host A] (女主持人)**：
  - *风格*：活泼机敏、好奇心强，善于把复杂技术翻译成大白话。喜欢追问"这跟我们有什么关系？"
  - *职责*：主导节目控场与流程推进，负责开场引入和转场过渡
  - *说话特点*：语气词丰富（"我跟你说"、"真的假的"、"这也太……"），情绪表达直接，但避免用"哎"开头显得不礼貌

- **周航 [Host B] (男主持人)**：
  - *风格*：沉稳理性但不端着，能把技术细节讲出故事感。偶尔毒舌吐槽行业乱象
  - *职责*：负责深度解读、技术拆解、商业逻辑分析
  - *说话特点*：爱说"你注意啊"、"问题是"、"说白了……"，喜欢用类比解释技术

## 核心要求
1. **自然对话**：两人像老朋友聊天，不是采访。有抢话、有打断、有笑声、有吐槽。
2. **禁止模板化**：不要"你说得对"、"没错"这种敷衍回应。每次回应要有信息量。
3. **禁止编号**：绝对禁止"第一条新闻"、"第二个头条"，用自然过渡（"说到这个……"、"对了还有件事……"）
4. **格式严格**：每段台词独占一行，以 `[Host A]` 或 `[Host B]` 开头
5. **口语化**：用"咱们"不用"我们"，用"这事儿"不用"这件事"，允许语法不完整
6. **禁用词**：避免使用：{banned_str}。不要"众所周知"、"废话不多说"这种开场。
7. **避免极短单句与捧哏模式**：避免任何一方只说几个字（如“确实”、“原来如此”、“真的假的”、“我也觉得”）。即使是表达惊叹、提问或进行话题切换，也必须将其拓展为包含个人见解、对比、或者关联其他常识的充实段落。两人单次发言应当有均衡的信息量，单次发言字数建议保持在 40-150 字之间（开场问候和结尾道别除外），严禁一人单方面长篇演讲而另一人只做复读机或提问机器。

## 主编大纲 (Markdown)
{editor_plan_json}

## 输出格式示例（纯文本，每段台词独占一行）
[Host A] 听众朋友大家好，欢迎收听{podcast_title}，今天是{date_str}。我是苏晴。
[Host B] 大家好，我是周航。
[Host A] 周航，今天科技圈有个大事，谷歌发新模型了你知道吗？
[Host B] 知道，我早上刷到了。这次更新在底层架构上做了挺大改动，特别是……

请**直接输出**纯文本对话全文。不要包含你的思考过程、不要解释要求、不要复述指令。
每段台词必须以 [Host A] 或 [Host B] 开头。**绝对不要**包含 Markdown 代码块标记 ``` 或标题 # 。
**绝对不要**以 "Let me"、"首先"、"好" 等非对话内容开头。"""


def build_writer_prompt(
    editor_plan_json: str,
    episode_date: datetime,
    podcast_title: str,
    style_cfg: dict[str, Any],
) -> str:
    """第二阶段：撰稿 Agent，将主编定下的大纲转化为双人对口相声/对谈剧本。"""
    banned = style_cfg.get("banned_words", DEFAULT_BANNED_WORDS)
    banned_str = "、".join(banned)
    date_str = _cn_date(episode_date)

    return WRITER_USER_TEMPLATE.format(
        date_str=date_str,
        podcast_title=podcast_title,
        banned_str=banned_str,
        editor_plan_json=editor_plan_json,
    )
