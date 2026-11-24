"""金标：抽样、校验、LLM 起草、冻结。

设计原则：
1. 抽样可复现（按 item_id 哈希排序，不依赖随机数）；
2. 金标必须带 evidence，且 evidence 能在原文中逐字找到；
3. 起草与冻结分离，`status=draft` 的金标不能用于最终验收。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .standard import (
    RELATIONS,
    Doc,
    GoldItem,
    Triple,
    normalize_relation,
    normalize_text,
)

VALUE_PATTERNS = (
    r"\d+(?:\.\d+)?\s*(?:亿|万|千)?\s*(?:元|美元|港币|欧元)",
    r"\d+(?:\.\d+)?\s*%",
    r"\d+(?:\.\d+)?\s*分",
    r"\d+(?:\.\d+)?\s*万(?:起)?",
    r"\d+(?:\.\d+)?\s*(?:million|billion|thousand|percent)",
    r"\d+(?:\.\d+)?",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b",
)


def _stable_key(item_id: str) -> str:
    return hashlib.sha1(item_id.encode("utf-8")).hexdigest()


def _is_zh(title: str, lang: str = "") -> bool:
    if lang:
        return lang.startswith("zh")
    return len(re.findall(r"[\u4e00-\u9fff]", title or "")) >= 5


def _length_bucket(text: str) -> str:
    size = len(text or "")
    if size < 900:
        return "short"
    if size < 2000:
        return "mid"
    return "long"


def load_docs_from_ocean(
    items_path: str | Path,
    *,
    min_chars: int = 400,
    require_ai: bool = True,
) -> list[Doc]:
    """读取 ocean items。默认只要 Stage 2 判定为 AI 相关的条目。

    三元组评测测的是「AI 新闻事实抽取」，如果混进纸飞机、手机评测这类
    非 AI 条目，金标会被无关事实稀释，方案差异也就测不出来。
    """
    rows = json.loads(Path(items_path).read_text(encoding="utf-8"))
    docs: list[Doc] = []
    for row in rows:
        item_id = str(row.get("item_id") or "")
        dup_of = str(row.get("dup_of") or "")
        if dup_of and dup_of != item_id:
            continue
        labels = row.get("labels") or {}
        if require_ai:
            relevance = str(labels.get("ai_relevance") or "")
            domain = str(labels.get("primary_domain") or "")
            if relevance not in ("core", "adjacent") and domain not in ("ai_core", "ai_infra"):
                continue
        text = str(row.get("full_text") or "").strip()
        if len(text) < min_chars:
            continue
        title = str(row.get("title") or "").strip()
        docs.append(
            Doc(
                item_id=str(row.get("item_id") or ""),
                title=title,
                text=text,
                lang="zh" if _is_zh(title, str(row.get("lang") or "")) else "en",
                published_at=str(row.get("published_at") or ""),
                event_id=str(row.get("event_id") or ""),
            )
        )
    return [d for d in docs if d.item_id]


def stratified_select(  # noqa: C901, PLR0912
    docs: list[Doc],
    *,
    total: int,
    quotas_zh: int,
    quotas_en: int,
    used_events: set[str] | None = None,
) -> list[Doc]:
    """按语言 × 长度分层，层内按哈希排序，保证可复现；同一事件只取一篇。"""
    used_events = set(used_events or ())
    buckets: dict[tuple[str, str], list[Doc]] = {}
    for doc in docs:
        buckets.setdefault((doc.lang, _length_bucket(doc.text)), []).append(doc)
    for rows in buckets.values():
        rows.sort(key=lambda d: _stable_key(d.item_id))

    def take(rows: list[Doc]) -> Doc | None:
        while rows:
            doc = rows.pop(0)
            if doc.event_id and doc.event_id in used_events:
                continue
            if doc.event_id:
                used_events.add(doc.event_id)
            return doc
        return None

    targets = {"zh": quotas_zh, "en": quotas_en}
    picked: list[Doc] = []
    # 先按层轮转，避免长文/短文失衡。
    order = [
        ("zh", "short"),
        ("zh", "mid"),
        ("en", "short"),
        ("zh", "long"),
        ("en", "mid"),
        ("en", "long"),
    ]
    while len(picked) < total:
        progressed = False
        for lang, bucket in order:
            if len(picked) >= total or targets.get(lang, 0) <= 0:
                continue
            doc = take(buckets.get((lang, bucket), []))
            if doc is None:
                continue
            picked.append(doc)
            targets[lang] -= 1
            progressed = True
        if not progressed:
            # 某语言配额没用完时，从其它语言补。
            for lang in ("zh", "en"):
                if len(picked) >= total:
                    break
                if targets.get(lang, 0) <= 0:
                    continue
                candidates = sorted(
                    (d for (blang, _b), rows in buckets.items() if blang == lang for d in rows),
                    key=lambda d: _stable_key(d.item_id),
                )
                for doc in candidates:
                    if len(picked) >= total or targets.get(lang, 0) <= 0:
                        break
                    if doc.event_id and doc.event_id in used_events:
                        continue
                    if doc.event_id:
                        used_events.add(doc.event_id)
                    picked.append(doc)
                    targets[lang] -= 1
            break
    return picked


def split_docs(
    docs: list[Doc],
    *,
    dev_size: int,
    existing: dict[str, str],
) -> dict[str, str]:
    """保留已有划分，新文档按语言平衡后分配 dev/test。"""
    out = {doc.item_id: existing[doc.item_id] for doc in docs if doc.item_id in existing}
    new = [doc for doc in docs if doc.item_id not in existing]
    new.sort(key=lambda d: _stable_key(d.item_id))
    remaining_dev = max(0, dev_size - sum(1 for v in out.values() if v == "dev"))
    by_lang: dict[str, list[Doc]] = {}
    for doc in new:
        by_lang.setdefault(doc.lang, []).append(doc)
    # 先按语言配额切 dev，避免 dev 全是中文或全是英文。
    for rows in by_lang.values():
        share = 0 if not new else round(remaining_dev * len(rows) / len(new))
        for doc in rows[:share]:
            out[doc.item_id] = "dev"
    for doc in new:
        out.setdefault(doc.item_id, "test")
    return out


def load_gold(path: str | Path) -> tuple[bool, list[GoldItem]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(payload.get("frozen")), [
        GoldItem.from_json(row) for row in payload.get("items", [])
    ]


def save_gold(path: str | Path, items: Iterable[GoldItem], *, frozen: bool) -> None:
    payload = {
        "version": 3,
        "frozen": frozen,
        "items": [item.to_json() for item in items],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(target)


@dataclass(slots=True)
class Issue:
    item_id: str
    triple: str
    reason: str


def _is_value(text: str) -> bool:
    return any(re.search(pattern, text or "") for pattern in VALUE_PATTERNS)


def validate_gold(  # noqa: PLR0912, C901
    items: Iterable[GoldItem], docs: dict[str, Doc]
) -> list[Issue]:
    issues: list[Issue] = []
    for item in items:
        doc = docs.get(item.item_id)
        if doc is None:
            issues.append(Issue(item.item_id, "", "missing_doc"))
            continue
        body = f"{doc.title}。{doc.text}"
        body_norm = normalize_text(body)
        seen: set[tuple[str, str, str]] = set()
        for triple in item.triples:
            label = f"{triple.subject} | {triple.relation} | {triple.object}"
            if triple.relation not in RELATIONS:
                issues.append(Issue(item.item_id, label, "bad_relation"))
            if not triple.subject or not triple.object:
                issues.append(Issue(item.item_id, label, "empty_endpoint"))
                continue
            if normalize_text(triple.subject) == normalize_text(triple.object):
                issues.append(Issue(item.item_id, label, "same_endpoint"))
            if len(triple.subject) > 40 or len(triple.object) > 40:
                issues.append(Issue(item.item_id, label, "endpoint_too_long"))
            if not triple.evidence:
                issues.append(Issue(item.item_id, label, "missing_evidence"))
            else:
                ev_norm = normalize_text(triple.evidence)
                if ev_norm not in body_norm:
                    issues.append(Issue(item.item_id, label, "evidence_not_in_text"))
                # 跨句事实（"该工作""团队"指代前文）无法同句共现，只校验两端在原文。
                cross = triple.meta.get("review") == "cross_sentence"
                if normalize_text(triple.subject) not in ev_norm and not cross:
                    issues.append(Issue(item.item_id, label, "subject_not_in_evidence"))
                if normalize_text(triple.object) not in ev_norm and not cross:
                    issues.append(Issue(item.item_id, label, "object_not_in_evidence"))
                if cross:
                    if normalize_text(triple.subject) not in body_norm:
                        issues.append(Issue(item.item_id, label, "subject_not_in_text"))
                    if normalize_text(triple.object) not in body_norm:
                        issues.append(Issue(item.item_id, label, "object_not_in_text"))
            key = (triple.relation, normalize_text(triple.subject), normalize_text(triple.object))
            if key in seen:
                issues.append(Issue(item.item_id, label, "duplicate"))
            seen.add(key)
            if triple.object_kind == "value" and not _is_value(triple.object):
                issues.append(Issue(item.item_id, label, "value_pattern_mismatch"))
    return issues


# ---------------------------------------------------------------------------
# LLM 起草
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """从下面这条 AI 新闻中抽取事实三元组（主语-关系-宾语）。

关系只能用这些 id：
release(发布) open_source(开源) update(更新) pricing(价格) supply(量产交付)
deploy(部署接入) support(支持) funding(融资上市) acquisition(收购) partnership(合作)
benchmark(评测对比) personnel(人事) layoff(裁员) regulation(监管) legal(诉讼)
incident(事故风险) research(研究) event(活动) info(信息披露) award(获奖认证) other(其他)

硬性规则：
1. subject 和 object 必须是原文中逐字出现的片段，不得改写、不得概括。
2. subject 与 object 各自不超过 20 字，必须是一个实体（公司/机构/人名/产品名/地点）或一个明确数值。
   - 不要以“部分”“某一批次”“能接触到……的人员”“玩家”这类泛指短语作端点。
   - 不要用带“的”的长定语，例如“欧洲及其他非美地区的部分实体版买家”应改写为“欧洲实体版买家”。
3. 只抽取原文明确陈述的事实，不推断、不补全。
4. evidence 必须是包含该三元组两端的最小原句，逐字复制。
5. 一条句子最多 1 条三元组；全文最多 8 条，只保留最重要的。
6. 关系必须与句中动词一致；仅表示对比、并列、转折的句子不构成三元组。
7. 没有符合条件的返回 []。

只输出 JSON 数组，不要解释。格式：
[{{"subject":"OpenAI","relation":"release","object":"GPT-5","object_kind":"entity","evidence":"原句"}}]

标题：{title}
正文：{text}
"""


def parse_llm_json(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    value = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = value.find("["), value.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [row for row in data if isinstance(row, dict)]


def draft_item(
    doc: Doc,
    *,
    call_llm,
    llm_cfg: dict[str, Any],
    split: str = "dev",
) -> GoldItem:
    raw = call_llm(DRAFT_PROMPT.format(title=doc.title, text=doc.text[:6000]), llm_cfg)
    triples: list[Triple] = []
    for row in parse_llm_json(raw):
        subject = str(row.get("subject") or "").strip()
        obj = str(row.get("object") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        relation = normalize_relation(str(row.get("relation") or ""))
        if not subject or not obj or not evidence:
            continue
        triples.append(
            Triple(
                subject=subject,
                relation=relation,
                object=obj,
                evidence=evidence,
                object_kind=str(
                    row.get("object_kind") or ("value" if _is_value(obj) else "entity")
                ),
            )
        )
    # 证据句必须真的在原文里；不在的直接丢弃。
    body_norm = normalize_text(f"{doc.title}。{doc.text}")
    kept = []
    for triple in triples:
        ev = normalize_text(triple.evidence)
        if ev and ev in body_norm:
            kept.append(triple)
    return GoldItem(
        item_id=doc.item_id,
        title=doc.title,
        lang=doc.lang,
        split=split,
        status="draft",
        triples=kept,
    )


def review_markdown(items: Iterable[GoldItem], docs: dict[str, Doc]) -> str:
    """人工审核清单：按文档分组，带编号和证据原句，便于逐条勾删。"""
    pending = [item for item in items if item.status == "draft"]
    lines = [
        f"# 待审核金标（{len(pending)} 篇 / {sum(len(i.triples) for i in pending)} 条）",
        "",
        "> deploy / benchmark / info / support / update 容易出现语义漂移，建议重点看。",
        "",
        "用法：默认全保留；要删的条目把「删」列改成 `x`，或直接说明「某篇只保留第 N 条」。",
        "",
    ]
    for item in sorted(pending, key=lambda row: (row.split, row.title)):
        lines.append(f"## [{item.split}] {item.title[:64]}")
        lines.append("")
        lines.append("| # | 删 | 主语 | 关系 | 宾语 | 证据原句 |")
        lines.append("|---|---|---|---|---|---|")
        for index, triple in enumerate(item.triples, 1):
            evidence = (triple.evidence or "").replace("|", "丨")[:70]
            lines.append(
                f"| {index} |  | {triple.subject} | {triple.relation} | "
                f"{triple.object} | {evidence} |"
            )
        if not item.triples:
            lines.append("| - | - | （本篇无三元组） |  |  |  |")
        lines.append("")
    return "\n".join(lines)
