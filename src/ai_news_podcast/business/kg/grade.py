"""判档：A 可信 / B 存疑 / C 丢弃（事实层 S6）。"""

from __future__ import annotations

import re

from ai_news_podcast.business.kg.types import FactCandidate

GENERIC_SUBJECTS = {
    "人工智能",
    "大模型",
    "模型",
    "行业",
    "市场",
    "用户",
    "客户",
    "企业",
    "公司",
    "团队",
    "研究",
    "技术",
    "平台",
    "系统",
    "ai",
}
PRONOUNS = {"它", "他", "她", "其", "该", "此", "这", "那", "其中", "后者", "前者"}
TRUNCATED_MARKERS = ("等", "等等", "之类", "等产品", "等服务")
GENERIC_OBJECTS = {
    "模型",
    "平台",
    "系统",
    "产品",
    "服务",
    "技术",
    "方案",
    "供应商",
    "合作伙伴",
    "合作",
    "业务",
    "能力",
    "芯片",
    "设备",
    "应用",
    "项目",
    "架构",
    "框架",
    "功能",
    "数据",
}
ROLE_IDENTITIES = {"联合创始人", "创始人", "联合创办人"}
TITLE_SEPARATORS = re.compile(r"[丨|!！?？;；]")
TITLE_MARKERS = ("实测第一", "榜单", "排行", "发布会", "直击")

# 证据句落在「虚构/假设/示例」引文里的信号。
# 例：「野生高考出题组连夜把 800 字材料作文题拟好了：「阅读下面的材料……近日，英伟达创始人黄仁勋辞去 CEO 职务」」
_FICTION_MARKERS = (
    "作文",
    "材料",
    "题目",
    "阅读下面",
    "假设",
    "假如",
    "示例",
    "例如",
    "比如",
    "考试",
    "拟好",
    "虚构",
    "段子",
    "大纲",
)
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (("「", "」"), ("“", "”"))
# 中英混排片段（"英伟达 NVIDIA 产品"）。纯英文专名（"Memory summary"）不算，
# 否则所有带空格的英文产品名都会被误杀。
_CJK = re.compile(r"[\u4e00-\u9fff]")
MIXED_FRAGMENT = re.compile(r"[A-Za-z]")
ENTITY_KINDS = {"company", "person", "org"}


def _inside_quoted_material(body: str, clause: str) -> bool:
    """证据句是否落在「假设/示例/作文」类引文中。

    只在引号内、且引号前 40 字内出现虚构标记时才算命中，
    避免把正常引用（「词元所」）误判。
    """
    needle = (clause or "").strip()[:20]
    if not body or not needle:
        return False
    position = body.find(needle)
    if position < 0:
        return False
    for opener, closer in _QUOTE_PAIRS:
        start = body.rfind(opener, 0, position)
        if start < 0:
            continue
        end = body.find(closer, start + 1)
        # 长引文常见「开了引号但一直没闭合」；这种情况仍算在引文内。
        if end >= 0 and end < position:
            continue
        context = body[max(0, start - 40) : start]
        if any(marker in context for marker in _FICTION_MARKERS):
            return True
    return False


def _object_text(candidate: FactCandidate) -> str:
    slot = candidate.slots.get("object")
    return str(slot.text or "").strip() if slot is not None else ""


def a_guard(candidate: FactCandidate) -> list[str]:  # noqa: C901, PLR0912
    """A 档守卫：只返回违规码，命中就降到 B（不丢卡、不影响召回）。"""
    violations: list[str] = []
    relation = candidate.relation
    object_slot = candidate.slots.get("object")
    object_text = _object_text(candidate)
    clause = str(candidate.clause or "")

    if relation == "personnel":
        role = candidate.slots.get("role")
        role_text = str(role.text or "").strip() if role is not None else ""
        if candidate.slots.get("person") is None:
            violations.append("a_guard_person_missing")
        if not role_text:
            violations.append("a_guard_role_missing")
        elif role_text in ROLE_IDENTITIES:
            violations.append("a_guard_role_identity")
        elif role_text not in clause:
            violations.append("a_guard_role_absent")
    elif object_slot is None or not object_text:
        violations.append("a_guard_object_missing")
    else:
        entity_id = str(object_slot.entity_id or "")
        if len(object_text) < 2:
            violations.append("a_guard_object_too_short")
        if len(object_text) > 30:
            violations.append("a_guard_object_too_long")
        if TITLE_SEPARATORS.search(object_text) or (
            not entity_id and any(marker in object_text for marker in TITLE_MARKERS)
        ):
            violations.append("a_guard_title_fragment")
        if object_text in GENERIC_OBJECTS:
            violations.append("a_guard_generic_object")
        if MIXED_FRAGMENT.search(object_text) and _CJK.search(object_text) and " " in object_text:
            violations.append("a_guard_mixed_fragment")
        if relation in ("funding", "pricing") and not (
            any(char.isdigit() for char in object_text) or entity_id
        ):
            violations.append("a_guard_amount_missing")
        if relation == "partnership":
            if object_slot.kind not in ENTITY_KINDS:
                violations.append("a_guard_partner_not_entity")
            if object_text == candidate.subject.text.strip():
                violations.append("a_guard_partner_self")
        if relation == "benchmark" and object_slot.kind not in ENTITY_KINDS | {"product"}:
            violations.append("a_guard_benchmark_no_target")
        if (
            relation == "research"
            and not entity_id
            and (
                object_text in GENERIC_OBJECTS
                or (object_text.endswith("模型") and len(object_text) <= 8)
            )
        ):
            violations.append("a_guard_generic_object")
    # 继承来的主语本来就不在当前子句里，这是定义而非缺陷；由 A2 档接管。
    if (
        not candidate.secondary
        and candidate.subject.text.strip()
        and candidate.subject.text.strip() not in clause
    ):
        violations.append("a_guard_subject_absent")
    # 主语继承不再在这里否决：它由 grade() 单独判成 A2。
    return violations


def _is_generic_subject(candidate: FactCandidate) -> bool:
    text = candidate.subject.text.strip().lower()
    return text in GENERIC_SUBJECTS or text in PRONOUNS or len(text) <= 1


def _object_truncated(candidate: FactCandidate) -> bool:
    object_value = candidate.slots.get("object")
    if object_value is None:
        return False
    text = object_value.text.strip()
    if len(text) > 30:
        return True
    return text.endswith(TRUNCATED_MARKERS)


def grade(  # noqa: C901, PLR0911, PLR0912
    candidate: FactCandidate,
    *,
    article_subject_text: str,
    object_max_chars: int = 30,
    evidence_scope: str = "body",
    source_text: str = "",
    title: str = "",
) -> tuple[str, list[str]]:
    """返回 (档位, 理由码)。C 档不落盘。

    ``source_text`` 传条目正文时，会额外校验证据句是否真在正文里：
    - 不在正文（通常来自标题）→ ``evidence_from_title``，降 B；
    - 落在「作文/假设/示例」类引文里 → ``evidence_quoted_material``，降 B。
    """
    if not candidate.subject.text.strip():
        return "C", ["empty_subject"]
    if candidate.relation == "other":
        return "C", ["relation_none"]
    if _is_generic_subject(candidate):
        return "C", ["subject_generic"]
    object_value = candidate.slots.get("object")
    if object_value is None and candidate.relation != "personnel":
        return "C", ["empty_object"]
    if object_value is not None and len(object_value.text.strip()) > object_max_chars:
        return "C", ["object_too_long"]

    reasons: list[str] = []
    if not candidate.subject.entity_id:
        reasons.append("subject_open")
    if (
        candidate.subject.text.strip() != article_subject_text.strip()
        and not candidate.subject.entity_id
    ):
        reasons.append("subject_open")
    if candidate.polarity == "negative":
        reasons.append("negated")
    if candidate.modality in {"rumor", "forecast"}:
        reasons.append("rumor")
    if candidate.secondary:
        reasons.append("subject_inherited")
    if evidence_scope in ("title", "summary"):
        reasons.append("title_only")
    clause = str(candidate.clause or "").strip()
    if source_text and clause:
        if title and clause == title.strip():
            reasons.append("evidence_from_title")
        elif clause not in source_text:
            # 正文里找不到：多半是标题或摘要派生的句子。
            reasons.append("evidence_from_title")
        elif _inside_quoted_material(source_text, clause):
            reasons.append("evidence_quoted_material")
    if "noun_trigger" in candidate.reasons:
        reasons.append("noun_trigger")
    if object_value is not None and not object_value.entity_id and object_value.kind == "other":
        reasons.append("object_weak")
    if _object_truncated(candidate):
        reasons.append("object_weak")

    unique = sorted(set(reasons))
    guard = a_guard(candidate)
    if guard:
        return "B", sorted(set(guard) | set(unique))
    if not unique:
        return "A", ["dep_confirmed"]
    if unique == ["subject_inherited"]:
        # 主语是从上一句继承来的：事实形态完整，但指代可能取错。
        # 单列 A2，让下游自己决定用不用，不污染 A 的可信度。
        return "A2", ["subject_inherited"]
    return "B", unique


__all__ = ["GENERIC_SUBJECTS", "a_guard", "grade"]
