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
MIXED_FRAGMENT = re.compile(r"[A-Za-z]")
ENTITY_KINDS = {"company", "person", "org"}


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
        if MIXED_FRAGMENT.search(object_text) and " " in object_text:
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
    if candidate.subject.text.strip() and candidate.subject.text.strip() not in clause:
        violations.append("a_guard_subject_absent")
    if candidate.secondary:
        violations.append("a_guard_inherited")
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
) -> tuple[str, list[str]]:
    """返回 (档位, 理由码)。C 档不落盘。"""
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
    if "noun_trigger" in candidate.reasons:
        reasons.append("noun_trigger")
    if object_value is not None and not object_value.entity_id and object_value.kind == "other":
        reasons.append("object_weak")
    if _object_truncated(candidate):
        reasons.append("object_weak")

    if reasons:
        return "B", sorted(set(reasons))
    guard = a_guard(candidate)
    if guard:
        return "B", sorted(set(guard))
    return "A", ["dep_confirmed"]


__all__ = ["GENERIC_SUBJECTS", "a_guard", "grade"]
