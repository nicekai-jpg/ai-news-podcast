"""关系判定与槽位填充：谓词 + 词表 → 候选事实（事实层 S5）。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

from ai_news_podcast.business.kg.lexicon import KGlexicon
from ai_news_podcast.business.kg.relations import canonical_relation
from ai_news_podcast.business.kg.schema import EventSchema
from ai_news_podcast.business.kg.types import (
    FactCandidate,
    LexiconSpan,
    ParseResult,
    Predicate,
    SlotValue,
)

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = "config/nlp/negation.yaml"

ROLE_WORDS = (
    "联合创始人",
    "首席科学家",
    "首席执行官",
    "首席技术官",
    "董事长",
    "总裁",
    "副总裁",
    "总经理",
    "负责人",
    "首席财务官",
    "首席运营官",
    "董事",
    "主席",
    "高管",
    "CEO",
    "CTO",
    "CFO",
    "COO",
    "co-founder",
    "founder",
)
ROLE_RE = re.compile("|".join(re.escape(x) for x in sorted(ROLE_WORDS, key=len, reverse=True)))
EN_ROLE_RE = re.compile(r"\b(CEO|CTO|CFO|COO|founder|co-founder|president|chair)\b", re.I)
_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(亿|万|千|百)?\s*(元|美元|人民币|%|％|个百分点|倍|台|张|辆|人|名|卡|次|折)"
)
_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日|(\d{1,2})月(\d{1,2})日")
_ISO_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_RELATIVE = {
    "今天": 0,
    "今日": 0,
    "昨日": 1,
    "昨天": 1,
    "本周": 3,
    "上周": 10,
    "去年": 365,
    "明年": -365,
}
_PRODUCT_SUFFIX = (
    "模型",
    "芯片",
    "手机",
    "平台",
    "系统",
    "API",
    "接口",
    "框架",
    "算法",
    "数据集",
    "APP",
    "App",
)
_COMPANY_SUFFIX = (
    "公司",
    "集团",
    "科技",
    "通信",
    "电子",
    "软件",
    "网络",
    "医药",
    "药业",
    "生物",
    "能源",
    "控股",
    "实业",
    "传媒",
    "游戏",
    "数据",
    "信息",
    "电气",
    "家电",
    "汽车",
    "银行",
    "Inc.",
    "Ltd.",
    "Corp.",
    "Labs",
)
_ORG_SUFFIX = (
    "大学",
    "研究院",
    "实验室",
    "委员会",
    "政府",
    "University",
    "Institute",
    "Foundation",
)
_LATIN_NAME = re.compile(r"[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*)+")
_ORG_SUFFIX_EN = (
    "Capital",
    "Management",
    "PBC",
    "Inc",
    "Inc.",
    "Corp",
    "Corp.",
    "Ltd",
    "Ltd.",
    "Labs",
    "Group",
    "Partners",
    "Foundation",
    "University",
    "Bank",
    "Securities",
    "Technologies",
    "Holdings",
)
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "nsubj:xsubj"}
_OBJECT_DEPS = {"dobj", "obj", "attr", "pobj", "nmod", "compound:nn"}
_OBJECT_NOUNS = {"价格", "售价", "估值", "金额", "销量", "融资", "发行", "营收", "收入"}
_GENERIC_SUBJECT_WORDS = {
    "价格",
    "售价",
    "估值",
    "金额",
    "融资",
    "市场",
    "行业",
    "企业",
    "机构",
    "团队",
    "用户",
    "客户",
    "内容",
    "形式",
    "原因",
    "结果",
    "榜单",
    "漏洞",
    "报告",
    "数据",
    "消息",
    "公告",
    "技术",
    "产品",
}
_PRONOUNS = {"它", "他", "她", "其", "该", "此", "这", "那", "其中", "后者", "前者"}


@lru_cache(maxsize=4)
def _load_rules_cached(path: str) -> dict[str, tuple[str, ...]]:
    file = Path(path)
    data: dict[str, object] = {}
    if file.is_file():
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            logger.warning("极性词典读取失败：%s", path, exc_info=True)
    keys = ("negation", "rumor", "forecast", "actual", "reporting")
    return {
        key: tuple(str(x) for x in (data.get(key) or ())) if isinstance(data.get(key), list) else ()
        for key in keys
    }


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, tuple[str, ...]]:
    """加载否定/传闻/预测/已发生/引述动词列表。"""
    return _load_rules_cached(str(Path(path)))


def reporting_verbs(path: str | Path = DEFAULT_RULES_PATH) -> set[str]:
    return set(load_rules(path)["reporting"])


@dataclass
class FillContext:
    """填充上下文。"""

    lexicon: KGlexicon
    schemas: dict[str, EventSchema]
    item_id: str
    published_at: str = ""
    article_subject: SlotValue | None = None
    prev_subject: SlotValue | None = None
    object_max_chars: int = 30
    lang: str = "zh"
    sentence_modality: str = "actual"
    rules: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def rule(self, name: str) -> tuple[str, ...]:
        if not self.rules:
            self.rules = load_rules()
        return self.rules.get(name, ())


def _token(parse: ParseResult, index: int):
    if index < 0 or index >= len(parse.tokens):
        return None
    return parse.tokens[index]


def _span_for(parse: ParseResult, index: int) -> LexiconSpan | None:
    token = _token(parse, index)
    if token is None:
        return None
    for span in parse.spans:
        if span.start <= token.idx and token.idx + len(token.text) <= span.end:
            return span
    return None


def _subtree(parse: ParseResult, index: int) -> list[int]:
    """token 及其所有后代（按 head 关系）。"""
    children: dict[int, list[int]] = {}
    for token in parse.tokens:
        if token.head == token.i:
            continue
        children.setdefault(token.head, []).append(token.i)
    collected: list[int] = []
    visited: set[int] = set()
    stack = [index]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        collected.append(current)
        stack.extend(children.get(current, []))
    return collected


def _subtree_text(parse: ParseResult, index: int) -> str:
    indices = [i for i in _subtree(parse, index) if 0 <= i < len(parse.tokens)]
    if not indices:
        return ""
    start = min(parse.tokens[i].idx for i in indices)
    end = max(parse.tokens[i].idx + len(parse.tokens[i].text) for i in indices)
    phrase = parse.text[start:end]
    phrase = re.split(r"[丨|—·]", phrase)[0]
    return phrase.strip(" ，,。；;：:")


def _suffix_kind(text: str) -> str:
    if text.endswith("汽车") and text.startswith(("新能源", "智能", "电动", "燃油", "自动驾驶")):
        return "product"
    for suffix in _COMPANY_SUFFIX:
        if text.endswith(suffix):
            return "company"
    for suffix in _ORG_SUFFIX:
        if text.endswith(suffix):
            return "org"
    for suffix in _PRODUCT_SUFFIX:
        if text.endswith(suffix):
            return "product"
    return ""


def _ner_kind(ent_type: str) -> str:
    mapping = {
        "PERSON": "person",
        "ORG": "org",
        "GPE": "org",
        "NORP": "org",
        "PRODUCT": "product",
        "WORK_OF_ART": "product",
        "FAC": "org",
        "LAW": "other",
        "EVENT": "event",
    }
    return mapping.get((ent_type or "").upper(), "")


def resolve_mention(  # noqa: PLR0911, PLR0912
    parse: ParseResult, index: int, *, allow_open: bool = True
) -> SlotValue | None:
    """token → 实体值：词表 span 优先，其次 NER/后缀/拉丁名。"""
    token = _token(parse, index)
    if token is None or not token.text.strip():
        return None
    span = _span_for(parse, index)
    if span is not None:
        return SlotValue("mention", span.name, span.entity_id, span.kind)
    for ent_text, ent_start, ent_end, ent_label in parse.ents:
        if ent_start <= token.idx and token.idx + len(token.text) <= ent_end:
            kind = _ner_kind(ent_label) or "other"
            return SlotValue("mention", ent_text, "", kind)
    text = token.text.strip()
    if text in _PRONOUNS or len(text) <= 1:
        return None
    if not allow_open:
        return None
    kind = _ner_kind(token.ent_type) or _suffix_kind(text)
    if not kind and _LATIN_NAME.fullmatch(text):
        words = text.split()
        if text.endswith(_ORG_SUFFIX_EN) or any(
            word.isupper() and 2 <= len(word) <= 4 for word in words
        ):
            kind = "company"
        elif _looks_like_product_en(text):
            # "ChatGPT Wallet"、"Gemini Robotics 2.5"：拉丁专名不等于人名。
            kind = "product"
        elif any(word[:1].islower() for word in words[1:]):
            # "Memory summary" 第二个词小写，不像人名；宾语位按产品或未知处理。
            kind = "product" if token.dep in _OBJECT_DEPS else "other"
        else:
            kind = "person"
    if not kind and text.isascii() and len(text) >= 3 and token.dep in _SUBJECT_DEPS:
        kind = "company"
    if (
        not kind
        and token.dep in _SUBJECT_DEPS
        and token.pos in ("NOUN", "PROPN")
        and 2 <= len(text) <= 5
        and text not in _GENERIC_SUBJECT_WORDS
    ):
        kind = "company"
    if not kind:
        return None
    return SlotValue("mention", text, "", kind)


def _resolve_subject(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> tuple[SlotValue, bool] | None:
    """返回 (主体, 是否继承)。"""
    mentions: list[SlotValue] = []
    for index in pred.subjects:
        value = resolve_mention(parse, index)
        if value is not None:
            mentions.append(value)
    if not mentions:
        if pred.subjects:
            # 有主语 token 但都是泛词：只有价格/估值这类"对象化名词"才退回文章主体，
            # 企业/机构/内容/原因这类直接丢（避免制造错主体）
            texts = [_token(parse, index).text for index in pred.subjects if _token(parse, index)]
            if not any(text in _OBJECT_NOUNS for text in texts):
                return None
        for fallback in (ctx.prev_subject, ctx.article_subject):
            if fallback is not None and _type_ok(fallback.kind, schema.subject_types):
                return fallback, True
        return None
    article = ctx.article_subject
    if article is not None:
        for value in mentions:
            if value.text == article.text or (
                article.entity_id and value.entity_id == article.entity_id
            ):
                return value, False
    for value in mentions:
        if _type_ok(value.kind, schema.subject_types):
            return value, False
    return None


def _type_ok(kind: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    if kind in allowed:
        return True
    if kind == "company" and "org" in allowed:
        return True
    return kind == "org" and "company" in allowed


_EN_PRODUCT_SUFFIX = (
    "wallet",
    "model",
    "api",
    "sdk",
    "chip",
    "gpu",
    "cpu",
    "os",
    "app",
    "suite",
    "cloud",
    "platform",
    "engine",
    "kit",
    "agent",
    "assistant",
    "studio",
    "pro",
    "max",
    "lite",
    "beta",
    "preview",
    "coder",
    "search",
    "browser",
    "database",
    "server",
    "cli",
    "plugin",
    "extension",
    "framework",
    "library",
    "protocol",
    "token",
    "pipeline",
    "benchmark",
)


def _looks_like_product_en(text: str) -> bool:
    """拉丁专名是否更像产品而非人名：带版本号或以产品词结尾。"""
    if re.search(r"\d", text):
        return True
    lowered = text.lower()
    return any(lowered.endswith(suffix) for suffix in _EN_PRODUCT_SUFFIX)


def _first_object_token(parse: ParseResult, pred: Predicate) -> int:
    for index in pred.objects:
        token = _token(parse, index)
        if token is not None and token.text.strip():
            return index
    return -1


_OBJECT_SKIP_DEPS = {"nsubj", "nsubjpass", "nsubj:xsubj", "punct", "aux", "aux:asp", "case"}
_OBJECT_SKIP_POS = {"VERB", "AUX", "ADV", "PART", "ADP", "CCONJ", "SCONJ"}


def _object_phrase(parse: ParseResult, index: int) -> str:
    """宾语短语：子树去掉主语/动词/虚词，只留名词性成分。"""
    indices = [
        i
        for i in _subtree(parse, index)
        if 0 <= i < len(parse.tokens)
        and parse.tokens[i].dep not in _OBJECT_SKIP_DEPS
        and parse.tokens[i].pos not in _OBJECT_SKIP_POS
    ]
    if not indices:
        token = _token(parse, index)
        return token.text if token is not None else ""
    start = min(parse.tokens[i].idx for i in indices)
    end = max(parse.tokens[i].idx + len(parse.tokens[i].text) for i in indices)
    phrase = parse.text[start:end]
    phrase = re.split(r"[丨|—·]", phrase)[0]
    return phrase.strip(" ，,。；;：:")


def _expand_right(parse: ParseResult, index: int, base_text: str) -> str:
    """把「模型 Gemini X」这类同位语/并列名词并进宾语文本。"""
    if not base_text:
        return base_text
    subtree = [i for i in _subtree(parse, index) if 0 <= i < len(parse.tokens)]
    end = max((parse.tokens[i].idx + len(parse.tokens[i].text) for i in subtree), default=0)
    if not end:
        return base_text
    allowed = {"NOUN", "PROPN", "NUM", "X", "ADJ", "SYM"}
    extra = ""
    for token in sorted(parse.tokens, key=lambda row: row.idx):
        if token.idx < end:
            continue
        if token.idx - end > 2:
            break
        if token.pos not in allowed:
            if not (token.dep == "punct" and re.search(r"[A-Za-z0-9]", token.text)):
                break
        elif token.dep == "punct" and token.pos not in ("PROPN", "NOUN", "NUM", "X"):
            break
        chunk = parse.text[end : token.idx + len(token.text)]
        if any(char in chunk for char in "，,。；;：:、（）()【】[]"):
            break
        extra += chunk
        end = token.idx + len(token.text)
    return (base_text + extra).strip()


_QUOTE_PAIRS = {"「": "」", "“": "”", "（": "）"}
# 引号内的同位语常是真正专名（大模型交易平台「词元所」），但不能太长。
_APPOSITIVE_MAX = 20


def _quoted_appositive(parse: ParseResult, index: int) -> SlotValue | None:
    """宾语后紧跟引号同位语时，取引号内的专名。

    「百度推出大模型交易平台「词元所」」：spaCy 把「平台」和「所」都标成 dobj，
    取第一个会得到泛称「大模型交易平台」。真正的名字在引号里。
    """
    count = len(parse.tokens)
    for offset in range(index + 1, min(count, index + 9)):
        open_token = parse.tokens[offset]
        opener = open_token.text.strip()
        closer = _QUOTE_PAIRS.get(opener)
        if closer is None:
            continue
        for follow in range(offset + 1, min(count, offset + 14)):
            text = parse.tokens[follow].text
            if text.strip() == closer:
                # 用原文切片保留原空格（「Memory summary」不能拼成 Memorysummary）。
                first = parse.tokens[offset + 1]
                last = parse.tokens[follow - 1] if follow - 1 > offset else first
                name = parse.text[first.idx : last.idx + len(last.text)].strip()
                if 2 <= len(name) <= _APPOSITIVE_MAX and not name.isdigit():
                    # 类型按后缀推断，避免把机构名一律当产品。
                    return SlotValue("object", name, "", _suffix_kind(name) or "product")
                return None
        return None
    return None


_GENERIC_HEADS = (
    "模型",
    "芯片",
    "平台",
    "框架",
    "系统",
    "处理器",
    "服务",
    "功能",
    "产品",
    "技术",
    "架构",
    "协议",
)
_LATIN_APPOSITIVE_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-]*(?:\s+[A-Za-z0-9.\-]+){0,4}")


def _is_generic_object(text: str) -> bool:
    value = text.strip()
    return value in _GENERIC_HEADS or any(value.endswith(head) for head in _GENERIC_HEADS)


def _latin_appositive(  # noqa: PLR0911
    parse: ParseResult, index: int, object_text: str
) -> SlotValue | None:
    """无引号同位语：「高性能AI模型——Muse Ember」「机器人模型 Gemini Robotics 2.5」。

    只在宾语是泛称（模型/芯片/平台…）且后面紧跟拉丁专名时启用，
    避免把普通英文名词当产品名。
    """
    if not _is_generic_object(object_text):
        return None
    count = len(parse.tokens)
    start = index + 1
    # 跳过破折号/连字符/括号等连接符。
    while start < count and parse.tokens[start].text.strip() in (
        "——",
        "—",
        "-",
        "：",
        ":",
        "（",
        "）",
    ):
        start += 1
    if start >= count:
        return None
    tail = "".join(token.text for token in parse.tokens[start : min(count, start + 10)])
    tail = tail.strip(" ，,。；;：:（）()「」“”")
    tail = re.split(r"[，,。；;！？!?]", tail)[0].strip()
    match = _LATIN_APPOSITIVE_RE.match(tail)
    if match is None:
        return None
    name = match.group(0).strip()
    if not (2 <= len(name) <= _APPOSITIVE_MAX):
        return None
    if not any(char.isalpha() for char in name) or not any(char.isupper() for char in name):
        return None
    if name in _GENERIC_HEADS:
        return None
    return SlotValue("object", name, "", "product")


def _object_value(parse: ParseResult, index: int, ctx: FillContext) -> SlotValue:
    quoted = _quoted_appositive(parse, index)
    if quoted is not None:
        return quoted
    text = _object_phrase(parse, index)
    span = _span_for(parse, index)
    if span is not None:
        return SlotValue("object", span.name, span.entity_id, span.kind)
    mention = resolve_mention(parse, index)
    combined = _expand_right(parse, index, text or mention.text if mention is not None else text)
    latin = _latin_appositive(parse, index, combined)
    if latin is not None:
        return latin
    if mention is not None:
        return SlotValue("object", combined, mention.entity_id, mention.kind)
    return SlotValue("object", combined, "", "other")


def _prep_value(parse: ParseResult, pred: Predicate, names: tuple[str, ...]) -> SlotValue | None:
    for prep, index in pred.preps:
        if prep in names:
            value = resolve_mention(parse, index)
            if value is not None:
                return SlotValue("peer", value.text, value.entity_id, value.kind)
    return None


def _amount(clause: str) -> tuple[str, float | None, str]:
    match = _AMOUNT_RE.search(clause)
    if not match:
        return "", None, ""
    text = match.group(0).strip()
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        value = None
    unit = (match.group(2) or "") + (match.group(3) or "")
    return text, value, unit


def _event_time(clause: str, published_at: str) -> str:
    """抽事件时间；历史日期也要记，缺省空串。"""
    match = _ISO_RE.search(clause) or _DATE_RE.search(clause)
    base = _parse_time(published_at)
    if match:
        groups = match.groups()
        try:
            if groups[0]:
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            else:
                year, month, day = base.year, int(groups[3]), int(groups[4])
            formatted = f"{year:04d}-{month:02d}-{day:02d}"
        except (TypeError, ValueError):
            return ""
        return formatted
    for word, offset in _RELATIVE.items():
        if word in clause:
            return (base - timedelta(days=offset)).strftime("%Y-%m-%d")
    return ""


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return datetime.now(UTC)


def _polarity_and_modality(clause: str, ctx: FillContext) -> tuple[str, str]:
    lowered = clause.lower()
    polarity = (
        "negative" if any(word and word in lowered for word in ctx.rule("negation")) else "positive"
    )
    modality = "actual"
    if any(word and word in lowered for word in ctx.rule("rumor")):
        modality = "rumor"
    elif any(word and word in lowered for word in ctx.rule("forecast")):
        modality = "forecast"
    if modality == "actual" and ctx.sentence_modality in ("rumor", "forecast"):
        modality = ctx.sentence_modality
    return polarity, modality


def _role_value(clause: str, parse: ParseResult, pred: Predicate) -> SlotValue | None:
    """职务词：先看“担任/出任/接任”之后，再看动词之后，避免抓成身份或前置修饰。"""
    for marker in ("担任", "出任", "接任", "升任", "兼任"):
        marker_at = clause.find(marker)
        if marker_at < 0:
            continue
        for match in ROLE_RE.finditer(clause, marker_at + len(marker)):
            return SlotValue("role", match.group(0), "", "role")
    verb = _token(parse, pred.verb_index)
    position = verb.idx if verb is not None else 0
    for match in ROLE_RE.finditer(clause):
        if match.start() >= position:
            return SlotValue("role", match.group(0), "", "role")
    for index in pred.attrs:
        text = _token(parse, index)
        if text is not None:
            return SlotValue("role", text.text, "", "role")
    match = EN_ROLE_RE.search(clause) or ROLE_RE.search(clause)
    if match:
        return SlotValue("role", match.group(0), "", "role")
    return None


def _fill_generic(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    object_index = _first_object_token(parse, pred)
    if object_index < 0:
        return None
    value = _object_value(parse, object_index, ctx)
    if not _type_ok(value.kind, schema.object_types) and value.kind != "other":
        # 类型不符时退而求其次：仍然尝试下一个宾语
        for index in pred.objects:
            candidate = _object_value(parse, index, ctx)
            if _type_ok(candidate.kind, schema.object_types):
                value = candidate
                break
        else:
            return None
    slots: dict[str, SlotValue] = {"object": value}
    peer = _prep_value(parse, pred, ("于", "在", "with", "at"))
    if peer is not None and "peer" in schema.optional:
        slots["peer"] = peer
    return slots


def _fill_personnel(  # noqa: PLR0912
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    subject, _ = _resolve_subject(parse, pred, ctx, schema) or (None, False)
    person: SlotValue | None = None
    company: SlotValue | None = None
    if subject is not None and subject.kind == "person":
        person = subject
    elif subject is not None and subject.kind in ("company", "org"):
        company = subject
    if person is None:
        for index in pred.objects:
            value = resolve_mention(parse, index)
            if value is not None and value.kind == "person":
                person = value
                break
    if person is None:
        for index in pred.subjects:
            value = resolve_mention(parse, index)
            if value is not None and value.kind == "person":
                person = value
                break
    role = _role_value(parse.text, parse, pred)
    if person is None:
        return None
    if company is None:
        company = _prep_value(parse, pred, ("在", "加入", "at", "of"))
    if company is None:
        company = _owner_before(parse, pred, person)
    slots: dict[str, SlotValue] = {"person": person}
    if role is not None:
        slots["role"] = role
    if company is not None:
        slots["company"] = SlotValue("company", company.text, company.entity_id, company.kind)
    return slots


# 「公司联合创始人黄仁勋」里的「公司」是泛称，不是归属实体。
_GENERIC_OWNERS = {"公司", "企业", "该企业", "该公司", "厂商", "集团", "团队"}


def _owner_before(parse: ParseResult, pred: Predicate, person: SlotValue) -> SlotValue | None:
    """「英伟达创始人黄仁勋辞去 CEO」这类句式：公司是人名的定语，不是介词宾语。

    只在人名左侧、同一谓语范围内找最近的机构/公司，避免把远距离实体当归属。
    """
    person_index = None
    for index in pred.subjects:
        value = resolve_mention(parse, index)
        if value is not None and value.text == person.text:
            person_index = index
            break
    if person_index is None:
        return None
    for index in range(person_index - 1, max(-1, person_index - 8), -1):
        value = resolve_mention(parse, index)
        if value is None:
            continue
        if (
            value.kind in ("company", "org")
            and value.text != person.text
            and value.text.strip() not in _GENERIC_OWNERS
        ):
            return value
    return None


def _entity_peer(value: SlotValue | None) -> SlotValue | None:
    if value is None:
        return None
    if value.entity_id or value.kind in ("company", "person", "org"):
        return value
    return None


def _fill_partnership(  # noqa: C901, PLR0912
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    peer = _entity_peer(_prep_value(parse, pred, ("与", "和", "with")))
    if peer is None:
        for index in pred.objects:
            peer = _entity_peer(_object_value(parse, index, ctx))
            if peer is not None:
                break
    if peer is None:
        mentions: list[SlotValue] = []
        for index in pred.subjects:
            value = resolve_mention(parse, index)
            if value is not None:
                mentions.append(value)
        if len(mentions) >= 2:
            peer = mentions[1]
    subject_info = _resolve_subject(parse, pred, ctx, schema)
    if peer is not None and subject_info is not None and _same_mention(peer, subject_info[0]):
        object_index = _first_object_token(parse, pred)
        if object_index >= 0:
            peer = _object_value(parse, object_index, ctx)
        if peer is not None and _same_mention(peer, subject_info[0]):
            return None
    if peer is None:
        return None
    if not peer.entity_id and peer.kind not in ("company", "person", "org"):
        # 合作方必须是实体；产品/泛词当不了合作方，退回别的并列主语或丢弃
        alternatives: list[SlotValue] = []
        for index in pred.subjects:
            value = resolve_mention(parse, index)
            if value is not None:
                alternatives.append(value)
        peer = next(
            (
                value
                for value in alternatives
                if value.entity_id or value.kind in ("company", "person", "org")
            ),
            None,
        )
        if peer is None:
            return None
    slots: dict[str, SlotValue] = {"object": peer}
    project = _prep_value(parse, pred, ("在", "于", "at"))
    if project is not None and "project" in schema.optional:
        slots["project"] = SlotValue("project", project.text, project.entity_id, project.kind)
    return slots


def _fill_amount_relation(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    text, value, unit = _amount(parse.text)
    object_index = _first_object_token(parse, pred)
    object_value = _object_value(parse, object_index, ctx) if object_index >= 0 else None
    if object_value is None and not text:
        return None
    slots: dict[str, SlotValue] = {}
    if text:
        slots["object"] = SlotValue("object", text, "", "amount", value, unit)
    elif object_value is not None:
        slots["object"] = object_value
    investor = _prep_value(parse, pred, ("由", "来自", "by", "from"))
    if investor is not None and "investor" in schema.optional:
        slots["investor"] = SlotValue("investor", investor.text, investor.entity_id, investor.kind)
    return slots


def _fill_acquisition(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    for index in pred.objects:
        value = _object_value(parse, index, ctx)
        if _type_ok(value.kind, schema.object_types):
            slots: dict[str, SlotValue] = {"object": value}
            text, amount, unit = _amount(parse.text)
            if text:
                slots["amount"] = SlotValue("amount", text, "", "amount", amount, unit)
            return slots
    return None


def _fill_pricing(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    text, value, unit = _amount(parse.text)
    object_index = _first_object_token(parse, pred)
    if object_index >= 0:
        slots: dict[str, SlotValue] = {"object": _object_value(parse, object_index, ctx)}
    elif text:
        slots = {"object": SlotValue("object", text, "", "amount", value, unit)}
    else:
        return None
    if text and "object" in slots and slots["object"].text != text:
        slots["amount"] = SlotValue("amount", text, "", "amount", value, unit)
    return slots


def _fill_supply(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    slots = _fill_generic(parse, pred, ctx, schema)
    if slots is None:
        return None
    text, value, unit = _amount(parse.text)
    if text:
        slots["amount"] = SlotValue("amount", text, "", "amount", value, unit)
    return slots


def _fill_layoff(
    parse: ParseResult, pred: Predicate, ctx: FillContext, schema: EventSchema
) -> dict[str, SlotValue] | None:
    text, value, unit = _amount(parse.text)
    if not text:
        return None
    return {"object": SlotValue("object", text, "", "amount", value, unit)}


FILLERS = {
    "generic": _fill_generic,
    "personnel": _fill_personnel,
    "partnership": _fill_partnership,
    "funding": _fill_amount_relation,
    "pricing": _fill_pricing,
    "supply": _fill_supply,
    "layoff": _fill_layoff,
    "acquisition": _fill_acquisition,
}


# 只给事实层用的兜底映射：这些词不放进 actions.yaml，避免影响 Stage 2 标注
KG_TRIGGER_RELATIONS: dict[str, str] = {
    "并入": "acquisition",
    "表示": "info",
    "公告": "info",
    "财报": "info",
    "业绩": "info",
    "声明": "info",
    "售价": "pricing",
    "价格": "pricing",
    "定价": "pricing",
    "发行": "funding",
}


def classify_relation(  # noqa: PLR0912
    parse: ParseResult, pred: Predicate, lexicon: KGlexicon
) -> tuple[str, str, str] | None:
    """返回 (关系, 动作 id, 原始词)；匹配不上返回 None。"""
    verb = _token(parse, pred.verb_index)
    if verb is None:
        return None
    action_id = lexicon.action_id(verb.text)
    if not action_id:
        for child in parse.tokens:
            if child.dep in ("compound:vc", "aux", "xcomp") and child.text != verb.text:
                action_id = lexicon.action_id(child.text)
                if action_id:
                    break
    if action_id:
        relation = canonical_relation(action_id)
        if relation != "other":
            return relation, action_id, verb.text
    for token in parse.tokens:
        if token.pos not in ("NOUN", "PROPN"):
            continue
        action_id = lexicon.action_id(token.text)
        if not action_id:
            continue
        relation = canonical_relation(action_id)
        if relation != "other":
            return relation, action_id, token.text
    for token in parse.tokens:
        if token.pos not in ("NOUN", "PROPN", "VERB", "ADJ"):
            continue
        relation = KG_TRIGGER_RELATIONS.get(token.text)
        if relation is not None:
            return relation, "kg_trigger", token.text
    return None


_INFO_MARKERS = ("公告", "声明", "通知", "消息", "财报", "业绩", "数据", "预测")
_RESEARCH_MARKERS = ("论文", "研究", "预印本", "专利", "报告")


def disambiguate(relation: str, raw: str, object_text: str, clause: str = "") -> str:
    """动词 + 对象消歧：发布公告算披露，发布论文算研究，合作开发算合作。"""
    text = str(object_text or "")
    if relation in ("release", "update") and any(
        marker in str(clause or "")
        for marker in ("合作开发", "联合开发", "共同开发", "携手", "联手")
    ):
        return "partnership"

    if relation == "release" and any(marker in text for marker in _INFO_MARKERS):
        return "info"
    if relation in ("release", "info") and any(marker in text for marker in _RESEARCH_MARKERS):
        return "research"
    return relation


def _same_mention(left: SlotValue, right: SlotValue) -> bool:
    a = str(left.text or "").strip()
    b = str(right.text or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    return bool(left.entity_id and left.entity_id == right.entity_id)


NOMINAL_TRIGGERS: dict[str, str] = {
    "售价": "pricing",
    "价格": "pricing",
    "定价": "pricing",
    "起售价": "pricing",
    "估值": "funding",
    "融资": "funding",
    "募资": "funding",
    "发行": "funding",
    "上市": "funding",
    "IPO": "funding",
    "漏洞": "incident",
    "事故": "incident",
    "故障": "incident",
    "宕机": "incident",
    "获奖": "award",
    "奖项": "award",
    "公告": "info",
    "财报": "info",
    "业绩": "info",
}


def build_nominal_candidate(  # noqa: PLR0911
    parse: ParseResult, ctx: FillContext
) -> FactCandidate | None:
    """名词性事实通道：售价/估值/IPO/漏洞这类无动词子句，只出 B 档。"""
    subject = ctx.article_subject
    if subject is None or subject.kind not in ("company", "org", "person"):
        return None
    text = parse.text
    hit: tuple[str, str] | None = None
    for trigger, relation in NOMINAL_TRIGGERS.items():
        if trigger in text:
            hit = (trigger, relation)
            break
    if hit is None:
        return None
    trigger, relation = hit
    schema = ctx.schemas.get(relation)
    if (
        schema is None
        or schema.filler == "none"
        or not _type_ok(subject.kind, schema.subject_types)
    ):
        return None
    trigger_dep = ""
    for token in parse.tokens:
        if token.text == trigger:
            trigger_dep = token.dep
            break
    amount_text, amount_value, unit = _amount(text)
    if relation in ("pricing", "funding"):
        if not amount_text or trigger_dep not in ("ROOT", "nsubj", "nsubjpass", "attr", "dobj"):
            return None
    elif trigger_dep not in ("ROOT", "nsubj", "nsubjpass", "attr", "dobj", "nmod"):
        return None
    if len(text) > 80:
        return None
    object_text = amount_text or trigger
    object_kind = "amount" if amount_text else "other"
    slots: dict[str, SlotValue] = {
        "object": SlotValue("object", object_text, "", object_kind, amount_value, unit),
        "event_time": SlotValue("event_time", "", "", "date"),
    }
    event_date = _event_time(text, ctx.published_at)
    if event_date:
        slots["event_time"] = SlotValue("event_time", event_date, "", "date")
    polarity, modality = _polarity_and_modality(text, ctx)
    return FactCandidate(
        clause=text,
        subject=SlotValue("subject", subject.text, subject.entity_id, subject.kind),
        relation=relation,
        relation_raw=trigger,
        slots=slots,
        polarity=polarity,
        modality=modality,
        reasons=["noun_trigger", "subject_inherited"],
        published_at=ctx.published_at,
        secondary=True,
        event_time=event_date,
    )


def build_candidate(  # noqa: PLR0911
    parse: ParseResult,
    pred: Predicate,
    match: tuple[str, str, str],
    ctx: FillContext,
) -> FactCandidate | None:
    """关系 + 谓词 + 词表 → 候选事实。"""
    relation, _action_id, raw = match
    schema = ctx.schemas.get(relation)
    if schema is None or schema.filler == "none":
        return None
    subject_info = _resolve_subject(parse, pred, ctx, schema)
    if subject_info is None:
        return None
    subject, inherited = subject_info
    fill = FILLERS.get(schema.filler, _fill_generic)
    slots = fill(parse, pred, ctx, schema)
    if slots is None:
        return None
    object_slot = slots.get("object")
    if object_slot is not None and _same_mention(object_slot, subject):
        return None
    object_slot = slots.get("object")
    if object_slot is not None:
        relation = disambiguate(relation, raw, object_slot.text, parse.text)
        schema = ctx.schemas.get(relation)
        if schema is None or schema.filler == "none":
            return None
    for required in schema.required:
        if required == "person":
            if "person" not in slots:
                return None
        elif required not in slots:
            return None
    polarity, modality = _polarity_and_modality(parse.text, ctx)
    article = ctx.article_subject
    secondary = inherited or (
        article is not None
        and subject.text != article.text
        and not (article.entity_id and subject.entity_id == article.entity_id)
    )
    event_time = _event_time(parse.text, ctx.published_at)
    if event_time:
        slots["event_time"] = SlotValue("event_time", event_time, "", "date")
    candidate = FactCandidate(
        clause=parse.text,
        subject=subject,
        relation=relation,
        relation_raw=raw,
        slots=slots,
        polarity=polarity,
        modality=modality,
        reasons=["dep_confirmed"],
        published_at=ctx.published_at,
        secondary=secondary,
        event_time=event_time,
    )
    if raw and lex_action_is_noun(parse, raw):
        candidate.reasons.append("noun_trigger")
    return candidate


def lex_action_is_noun(parse: ParseResult, raw: str) -> bool:
    return any(token.text == raw and token.pos in ("NOUN", "PROPN") for token in parse.tokens)


__all__ = [
    "DEFAULT_RULES_PATH",
    "FillContext",
    "build_candidate",
    "classify_relation",
    "load_rules",
    "reporting_verbs",
    "resolve_mention",
]


def _fallback_object(parse: ParseResult, pred: Predicate) -> int:
    """没有句法宾语时（"创维推出显示器"根是名词），取动词后最近的名词当宾语。"""
    verb = _token(parse, pred.verb_index)
    if verb is None:
        return -1
    later = [
        token
        for token in parse.tokens
        if token.idx > verb.idx and token.pos in ("NOUN", "PROPN", "NUM", "X")
    ]
    if not later:
        return -1
    heads = [token for token in later if token.dep in ("ROOT", "dobj", "obj", "attr", "nsubj")]
    chosen = heads[-1] if heads else later[-1]
    return chosen.i


def build_candidates(
    parse: ParseResult,
    pred: Predicate,
    match: tuple[str, str, str],
    ctx: FillContext,
) -> list[FactCandidate]:
    """一个谓词 → 候选事实列表（通用关系按不同宾语各出一条，符合设计「不同对象各留一条」）。"""
    relation = match[0]
    schema = ctx.schemas.get(relation)
    if not pred.objects:
        fallback_index = _fallback_object(parse, pred)
        if fallback_index >= 0:
            pred = Predicate(
                verb_index=pred.verb_index,
                verb_text=pred.verb_text,
                subjects=pred.subjects,
                objects=(fallback_index,),
                attrs=pred.attrs,
                preps=pred.preps,
                reported=pred.reported,
            )
    if schema is not None and schema.filler == "generic" and len(pred.objects) > 1:
        rows: list[FactCandidate] = []
        seen: set[str] = set()
        for index in pred.objects:
            slot = _object_value(parse, index, ctx) if ctx is not None else None
            if (
                slot is not None
                and not slot.entity_id
                and slot.kind
                not in (
                    "company",
                    "person",
                    "org",
                    "product",
                )
            ):
                continue
            single = Predicate(
                verb_index=pred.verb_index,
                verb_text=pred.verb_text,
                subjects=pred.subjects,
                objects=(index,),
                attrs=pred.attrs,
                preps=pred.preps,
                reported=pred.reported,
            )
            candidate = build_candidate(parse, single, match, ctx)
            if candidate is None:
                continue
            slot = candidate.slots.get("object")
            key = str(slot.text) if slot is not None else ""
            if key in seen:
                continue
            seen.add(key)
            rows.append(candidate)
        if rows:
            return rows
    candidate = build_candidate(parse, pred, match, ctx)
    return [candidate] if candidate is not None else []


__all__ += ["NOMINAL_TRIGGERS", "build_candidates", "build_nominal_candidate"]
