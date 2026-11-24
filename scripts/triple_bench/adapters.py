"""参评方案适配器：每个方案只负责「文档 → 三元组」，其余交给统一评测。

历史上调研过的方案都在这里登记；跑不动的会抛 `AdapterUnavailable`，
在报告里单列为 N/A，而不是静默跳过。
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .standard import Doc, Triple, normalize_relation, sentence_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class AdapterUnavailable(RuntimeError):  # noqa: N818
    """依赖或模型不可用；报告里必须显式说明原因。"""


_ERRORS: dict[str, list[str]] = {}


def _record_error(adapter: str, exc: Exception) -> None:
    """记录单句级异常。静默吞异常会让整轮结果为 0 却看不出原因。"""
    bucket = _ERRORS.setdefault(adapter, [])
    if len(bucket) < 20:
        bucket.append(f"{type(exc).__name__}: {exc}")


def adapter_errors() -> dict[str, list[str]]:
    return {name: list(items) for name, items in _ERRORS.items()}


PRONOUNS = {
    "她",
    "他",
    "它",
    "他们",
    "我们",
    "其",
    "这",
    "那",
    "该公司",
    "企业",
    "平台",
    "模型",
    "开发者",
    "官方",
    "人们",
    "公司",
    "厂商",
    "用户",
    "团队",
    "双方",
    "后者",
    "前者",
    "此举",
    "该技术",
    "该系统",
    "该功能",
    "该机构",
    "该模型",
}

_EXTRA_ACTIONS = {
    "提出": "research",
    "发表": "research",
    "研制": "research",
    "入选": "award",
    "收录": "award",
    "荣获": "award",
    "上市": "release",
    "搭载": "deploy",
    "内置": "deploy",
    "集成": "deploy",
    "联手": "partnership",
    "联合": "partnership",
    "签署": "partnership",
    "投资": "funding",
    "注资": "funding",
    "领投": "funding",
    "递交": "funding",
    "定价": "pricing",
    "量产": "supply",
    "任命": "personnel",
    "辞任": "personnel",
    "处罚": "regulation",
    "罚款": "regulation",
}

_REL_MAP = {
    "release": "release",
    "open_source": "open_source",
    "demo": "release",
    "product_update": "update",
    "pricing": "pricing",
    "supply": "supply",
    "deployment": "deploy",
    "support": "support",
    "provide": "support",
    "funding": "funding",
    "ipo": "funding",
    "acquisition": "acquisition",
    "partnership": "partnership",
    "sponsor": "partnership",
    "cooperate_std": "partnership",
    "benchmark": "benchmark",
    "outperform": "benchmark",
    "ranking": "benchmark",
    "personnel": "personnel",
    "layoff": "layoff",
    "regulation": "regulation",
    "investigation": "regulation",
    "ban": "regulation",
    "lawsuit": "legal",
    "settle": "legal",
    "accident": "incident",
    "recall": "incident",
    "research": "research",
    "training": "research",
    "event": "event",
    "found": "event",
    "award": "award",
    "disclosed": "info",
    "rumor": "info",
    "deny": "info",
    "predict": "info",
}


def _load_action_aliases() -> dict[str, str]:
    import yaml

    path = ROOT / "config" / "nlp" / "actions.yaml"
    aliases: dict[str, str] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for action in data.get("actions", []):
            for name in [action.get("name"), *(action.get("aliases") or [])]:
                if name:
                    aliases[str(name).strip().lower()] = str(action.get("id") or "")
    for name, action in _EXTRA_ACTIONS.items():
        aliases.setdefault(name, action)
    return aliases


def _canon(action_id: str) -> str:
    return _REL_MAP.get(str(action_id or "").strip().lower(), "other")


def _trim(text: str, max_len: int = 22) -> str | None:
    value = (text or "").strip()
    for sep in ("——", "—", "：", ":"):
        if sep in value:
            value = value.split(sep)[-1]
    if len(value) > max_len and "的" in value:
        value = value.split("的")[-1]
    value = value.strip(" ，。；、")
    if not value or len(value) > max_len:
        return None
    if re.search(r"[，。；！？]", value):
        return None
    if value in PRONOUNS:
        return None
    return value


def _dedupe(triples: list[Triple]) -> list[Triple]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Triple] = []
    for triple in triples:
        key = (
            normalize_relation(triple.relation),
            re.sub(r"\s", "", triple.subject),
            re.sub(r"\s", "", triple.object),
        )
        if key in seen or not triple.subject or not triple.object:
            continue
        seen.add(key)
        out.append(triple)
    return out


# ---------------------------------------------------------------------------
# 1. 仓库现有事实层（V2，spaCy 依存 + 词表 + 槽位）
# ---------------------------------------------------------------------------


def repo_rules(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    from ai_news_podcast.business.kg.service import build_facts
    from ai_news_podcast.business.pipeline.kg_stage_service import load_facts_config

    cfg = load_facts_config(ROOT / "config")
    item = dict(raw)
    item.setdefault("full_text", doc.text)
    item.setdefault("title", doc.title)
    result = build_facts(
        [item],
        [],
        date_str=str(doc.published_at or "")[:10] or "2026-11-16",
        config=cfg,
    )
    triples: list[Triple] = []
    for card in result.cards:
        obj = (card.slots.get("object") or {}).get("text", "")
        if not obj:
            continue
        triples.append(
            Triple(
                subject=card.subject.get("name", ""),
                relation=normalize_relation(card.relation),
                object=obj,
                confidence=0.9 if card.grade == "A" else 0.6,
                evidence=card.evidence.get("sentence", ""),
                meta={"grade": card.grade},
            )
        )
    return _dedupe(triples)


# ---------------------------------------------------------------------------
# 2/3. HanLP 2.x：SRL 帧 / SDP 语义依存
# ---------------------------------------------------------------------------

_HANLP_TOK = None


def _hanlp_tokenizer():
    global _HANLP_TOK
    if _HANLP_TOK is not None:
        return _HANLP_TOK
    try:
        import hanlp
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise AdapterUnavailable(f"hanlp 未安装: {exc}") from exc
    _HANLP_TOK = hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
        verbose=False,
    )
    return _HANLP_TOK


def hanlp_srl(doc: Doc, raw: dict[str, Any]) -> list[Triple]:  # noqa: PLR0912
    if doc.lang != "zh":
        # HanLP 这里是中文 MTL 模型，英文只能靠词表碰运气，且长文极慢。
        return []
    aliases = _load_action_aliases()
    tok = _hanlp_tokenizer()
    sentences = sentence_split(f"{doc.title}。{doc.text}")
    parsed = tok(sentences)
    triples: list[Triple] = []
    for index, sentence in enumerate(sentences):
        for frame in parsed["srl"][index]:
            predicate = None
            arg0: list[str] = []
            arg1 = None
            for text, role, _start, _end in frame:
                if role == "PRED":
                    predicate = text
                elif role == "ARG0":
                    arg0.append(text)
                elif role == "ARG1" and arg1 is None:
                    arg1 = text
            if not predicate or not arg1:
                continue
            action = aliases.get(predicate.strip().lower())
            if not action:
                continue
            relation = _canon(action)
            if len(arg0) >= 2 and relation == "partnership":
                left, right = _trim(arg0[0]), _trim(arg0[1])
                if left and right and left != right:
                    triples.append(Triple(left, relation, right, confidence=0.6, evidence=sentence))
                continue
            if not arg0:
                continue
            subject, obj = _trim(arg0[0]), _trim(arg1)
            if not subject or not obj or subject == obj:
                continue
            triples.append(Triple(subject, relation, obj, confidence=0.6, evidence=sentence))
    return _dedupe(triples)


def hanlp_sdp(doc: Doc, raw: dict[str, Any]) -> list[Triple]:  # noqa: C901
    if doc.lang != "zh":
        return []
    aliases = _load_action_aliases()
    tok = _hanlp_tokenizer()
    sentences = sentence_split(f"{doc.title}。{doc.text}")
    parsed = tok(sentences)

    def subtree(tokens: list[str], heads: list[int], root: int) -> str:
        keep = [root]
        changed = True
        while changed:
            changed = False
            for index, head in enumerate(heads):
                if index != root and head in [item + 1 for item in keep] and index not in keep:
                    keep.append(index)
                    changed = True
        return "".join(tokens[index] for index in sorted(keep)).strip("，。；、 ")

    triples: list[Triple] = []
    for index, sentence in enumerate(sentences):
        tokens = parsed["tok/fine"][index]
        dep = parsed["dep"][index]
        sdp = parsed["sdp"][index]
        heads = [head for head, _ in dep]
        agents: dict[int, list[int]] = {}
        objects: dict[int, list[int]] = {}
        for position, edges in enumerate(sdp):
            for head, label in edges:
                if label == "Agt":
                    agents.setdefault(head - 1, []).append(position)
                elif label in ("Pat", "Cont"):
                    objects.setdefault(head - 1, []).append(position)
        for predicate in set(agents) | set(objects):
            if not (0 <= predicate < len(tokens)):
                continue
            action = aliases.get(tokens[predicate].strip().lower())
            if not action:
                continue
            relation = _canon(action)
            for agent_index in agents.get(predicate, []):
                for object_index in objects.get(predicate, []):
                    subject = _trim(subtree(tokens, heads, agent_index))
                    obj = _trim(subtree(tokens, heads, object_index))
                    if subject and obj and subject != obj:
                        triples.append(
                            Triple(subject, relation, obj, confidence=0.5, evidence=sentence)
                        )
    return _dedupe(triples)


# ---------------------------------------------------------------------------
# 4/5. GLiNER2：JointIE 联合抽取 / AutoExtractor 关系抽取
# ---------------------------------------------------------------------------

_GLINER2_MODEL = os.environ.get(
    "GLINER2_MODEL", str(Path.home() / "nas/tmp/ner-eval/models/gliner2.5-multi-v1")
)

_ZH_ENTS = {
    "公司": "公司、企业、厂商",
    "人物": "人物姓名",
    "组织机构": "大学、研究机构、政府部门",
    "产品": "产品、模型、芯片",
    "其他": "金额、业务、事件等对象",
}
_EN_ENTS = {
    "company": "companies and vendors",
    "person": "person names",
    "organization": "universities, labs, government bodies",
    "product": "products, models, chips",
    "other": "amounts, businesses, events",
}
_ZH_REL = [
    ("发布", "公司", "产品"),
    ("开源", "公司", "产品"),
    ("更新", "公司", "产品"),
    ("量产交付", "公司", "产品"),
    ("收购", "公司", "公司"),
    ("合作", "公司", "公司"),
    ("融资上市", "公司", "其他"),
    ("研究", "组织机构", "产品"),
    ("人事", "人物", "公司"),
    ("监管", "组织机构", "公司"),
]
_EN_REL = [
    ("release", "company", "product"),
    ("open source", "company", "product"),
    ("update", "company", "product"),
    ("supply", "company", "product"),
    ("acquisition", "company", "company"),
    ("partnership", "company", "company"),
    ("funding", "company", "other"),
    ("research", "organization", "product"),
    ("personnel", "person", "company"),
    ("regulation", "organization", "company"),
]


_GLINER2_CACHE: dict[str, Any] = {}


def _gliner2_jointie():
    if "jointie" in _GLINER2_CACHE:
        return _GLINER2_CACHE["jointie"]
    try:
        from gliner2.joint_ie import JointIE, JointIEConfig
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise AdapterUnavailable(f"gliner2 未安装: {exc}") from exc
    model = Path(_GLINER2_MODEL)
    if not model.exists():
        raise AdapterUnavailable(f"GLiNER2 模型不存在: {model}")
    zh = JointIE.from_pretrained(str(model), word_splitter="char")
    en = JointIE.from_pretrained(str(model), word_splitter="whitespace")
    schema_zh = zh.create_schema().entities(_ZH_ENTS)
    for relation, head, tail in _ZH_REL:
        schema_zh = schema_zh.relation(relation, head, tail)
    schema_en = en.create_schema().entities(_EN_ENTS)
    for relation, head, tail in _EN_REL:
        schema_en = schema_en.relation(relation, head, tail)
    bundle = ((zh, schema_zh), (en, schema_en), JointIEConfig(optimizer="beam", beam_size=16))
    _GLINER2_CACHE["jointie"] = bundle
    return bundle


def gliner2_jointie(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    (zh, schema_zh), (en, schema_en), cfg = _gliner2_jointie()
    is_zh = doc.lang == "zh"
    model, schema = (zh, schema_zh) if is_zh else (en, schema_en)
    triples: list[Triple] = []
    for sentence in sentence_split(f"{doc.title}。{doc.text}"):
        try:
            result = model.extract(sentence, schema, config=cfg)
        except Exception as exc:
            _record_error("gliner2_jointie", exc)
            continue
        heads = {entity.id: entity for entity in result.entities}
        # 同一实体对常同时命中 release/open_source/supply 等互斥关系；
        # schema 本身没有互斥约束，这里按置信度只保留最强的一条。
        best: dict[tuple[str, str], Any] = {}
        for relation in result.relations:
            key = (relation.head, relation.tail)
            if key not in best or relation.confidence > best[key].confidence:
                best[key] = relation
        for (head, tail), relation in best.items():
            if relation.confidence < 0.5:
                continue
            if head not in heads or tail not in heads:
                continue
            subject, obj = heads[head].text, heads[tail].text
            # 端点必须比关系词更像实体：长度过短或与关系名同形时丢弃。
            if len(subject) < 2 or len(obj) < 2:
                continue
            if obj in ("产品", "新材料", "技术", "模型", "平台", "服务"):
                continue
            triples.append(
                Triple(
                    subject=subject,
                    relation=normalize_relation(relation.type),
                    object=obj,
                    confidence=float(relation.confidence),
                    evidence=sentence,
                )
            )
    return _dedupe(triples)


def gliner2_rel(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    try:
        from gliner2 import AutoExtractor
    except ImportError as exc:  # pragma: no cover
        raise AdapterUnavailable(f"gliner2 未安装: {exc}") from exc
    model_path = Path(_GLINER2_MODEL)
    if not model_path.exists():
        raise AdapterUnavailable(f"GLiNER2 模型不存在: {model_path}")
    if "autoext" not in _GLINER2_CACHE:
        _GLINER2_CACHE["autoext"] = AutoExtractor.from_pretrained(str(model_path))
    extractor = _GLINER2_CACHE["autoext"]
    relations_zh = [
        "发布",
        "开源",
        "更新",
        "价格",
        "量产交付",
        "部署接入",
        "融资上市",
        "收购",
        "合作",
        "人事",
        "裁员",
        "监管",
        "诉讼",
        "事故风险",
        "研究",
        "活动",
        "信息披露",
        "获奖认证",
    ]
    relations_en = [
        "release",
        "open source",
        "update",
        "pricing",
        "supply",
        "deployment",
        "funding",
        "acquisition",
        "partnership",
        "personnel",
        "layoff",
        "regulation",
        "lawsuit",
        "incident",
        "research",
        "event",
        "information disclosure",
        "award",
    ]
    relations = relations_zh if doc.lang == "zh" else relations_en
    triples: list[Triple] = []
    for sentence in sentence_split(f"{doc.title}。{doc.text}"):
        try:
            result = extractor.batch_extract_relations(
                [sentence], relations, threshold=0.4, include_confidence=True
            )[0]
        except Exception as exc:
            _record_error("gliner2_rel", exc)
            continue
        # 同一端点对会同时命中多个标签（release/open_source/update/supply…），
        # 只保留置信度最高的一条，否则精度被互斥标签拉垮。
        best: dict[tuple[str, str], tuple[str, float]] = {}
        for label, values in (result.get("relation_extraction") or {}).items():
            for value in values:
                confidence = float(value.get("confidence") or 0.0)
                key = (value["head"]["text"], value["tail"]["text"])
                if key not in best or confidence > best[key][1]:
                    best[key] = (label, confidence)
        for (subject, obj), (label, confidence) in best.items():
            triples.append(
                Triple(
                    subject=subject,
                    relation=normalize_relation(label),
                    object=obj,
                    confidence=confidence,
                    evidence=sentence,
                )
            )
    return _dedupe(triples)


# ---------------------------------------------------------------------------
# 6. LTP 4.x SRL
# ---------------------------------------------------------------------------

_LTP = None


def _ltp_model():
    global _LTP
    if _LTP is not None:
        return _LTP
    try:
        from ltp import LTP
    except ImportError as exc:  # pragma: no cover
        raise AdapterUnavailable(f"ltp 未安装: {exc}") from exc
    _LTP = LTP("LTP/small")
    return _LTP


def ltp_srl(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    if doc.lang != "zh":
        return []
    aliases = _load_action_aliases()
    model = _ltp_model()
    sentences = sentence_split(f"{doc.title}。{doc.text}")
    parsed = model.pipeline(sentences, tasks=["cws", "pos", "srl"])
    triples: list[Triple] = []
    for index, sentence in enumerate(sentences):
        for frame in parsed.srl[index] or []:
            predicate = str(frame.get("predicate") or "").strip()
            action = aliases.get(predicate.lower())
            if not action:
                continue
            args = {role: (start, end) for role, _text, start, end in frame["arguments"]}
            if "A0" not in args or "A1" not in args:
                continue
            words = parsed.cws[index]
            start, end = args["A0"]
            subject = _trim("".join(words[start : end + 1]))
            start, end = args["A1"]
            obj = _trim("".join(words[start : end + 1]))
            if not subject or not obj or subject == obj:
                continue
            triples.append(Triple(subject, _canon(action), obj, confidence=0.5, evidence=sentence))
    return _dedupe(triples)


# ---------------------------------------------------------------------------
# 7. LLM zero-shot（MiniMax）
# ---------------------------------------------------------------------------

LLM_PROMPT = """从下面这条 AI 新闻中抽取事实三元组。

关系只能用这些 id：
release open_source update pricing supply deploy support funding acquisition
partnership benchmark personnel layoff regulation legal incident research event info award other

规则：
1. subject/object 必须逐字出现在原文；必须是实体、产品名或明确数值，不能是代词或整句。
2. 只抽明确陈述的事实；evidence 是包含两端的最小原句。
3. 每条句子最多 1 条，全文最多 10 条。
4. 没有则返回 []。

只输出 JSON 数组：
[{{"subject":"OpenAI","relation":"release","object":"GPT-5","object_kind":"entity","evidence":"原句"}}]

标题：{title}
正文：{text}
"""


def _llm_cfg() -> dict[str, Any]:
    return {
        "provider": "openai_compatible",
        "api_key_env": "MINIMAX_API_KEY",
        "model": os.environ.get("TRIPLE_BENCH_LLM_MODEL", "MiniMax-M3.1-Flash"),
        "base_url": "https://api.minimaxi.com/v1",
        "temperature": 0.0,
        "max_output_tokens": 24576,
        "timeout": 300,
    }


def llm_zero_shot(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    from ai_news_podcast.business.pipeline.llm_client_service import call_llm

    from .gold import parse_llm_json
    from .standard import normalize_text

    raw_text = call_llm(
        LLM_PROMPT.format(title=doc.title, text=doc.text[:6000]),
        _llm_cfg(),
    )
    body = normalize_text(f"{doc.title}。{doc.text}")
    triples: list[Triple] = []
    for row in parse_llm_json(raw_text):
        subject = str(row.get("subject") or "").strip()
        obj = str(row.get("object") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        if not subject or not obj:
            continue
        if evidence and normalize_text(evidence) not in body:
            continue
        triples.append(
            Triple(
                subject=subject,
                relation=normalize_relation(str(row.get("relation") or "")),
                object=obj,
                confidence=0.8,
                evidence=evidence,
                object_kind=str(row.get("object_kind") or "entity"),
            )
        )
    return _dedupe(triples)


# ---------------------------------------------------------------------------
# 8/9. 历史上调研但跑不了的方案：显式登记
# ---------------------------------------------------------------------------


def uie(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    raise AdapterUnavailable(
        "UIE 需要 Docker + paddlenlp==2.8.1；历史实测 106–308 秒/篇，未纳入主表"
    )


def opennre(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    raise AdapterUnavailable(
        "OpenNRE 只有 wiki80/TACRED 英文关系分类权重，且需先给实体对，无法映射到 21 类"
    )


def deepke(doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    raise AdapterUnavailable(
        "DeepKE 2.2.7 安装失败（sdist 缺 requirements.txt），源码版依赖 python3.8/torch1.5"
    )


# 只登记非 LLM 方案：本评测的目标是「不用大模型能到什么程度」。
ADAPTERS: dict[str, Callable[[Doc, dict[str, Any]], list[Triple]]] = {
    "repo_rules": repo_rules,
    "hanlp_srl": hanlp_srl,
    "hanlp_sdp": hanlp_sdp,
    "gliner2_jointie": gliner2_jointie,
    "gliner2_rel": gliner2_rel,
    "ltp_srl": ltp_srl,
    "uie": uie,
    "opennre": opennre,
    "deepke": deepke,
}

# LLM 只作为离线参照，不参与排名；需要时用 --adapter 显式指定。
REFERENCE_ADAPTERS: dict[str, Callable[[Doc, dict[str, Any]], list[Triple]]] = {
    "llm_zero_shot": llm_zero_shot,
}

# 解析器/分词类方案（jahba、THULAC 等）只做句法，不单独产出三元组；
# 它们的结果体现在 docs/kg_extraction.md 的 UD 横评表里。
PARSER_ONLY = ("spacy", "custom_tokenizer", "jieba", "thulac", "stanza", "lac", "supar")


# 线程安全的方案可以并行处理文档；句法分析类共用解析器实例，保持串行。
CONCURRENCY: dict[str, int] = {
    "llm_zero_shot": int(os.environ.get("TRIPLE_BENCH_LLM_WORKERS", "6")),
    # GLiNER 的推理对象不是线程安全的：并发调用会静默返回空结果。
    "gliner2_jointie": 1,
    "gliner2_rel": 1,
}


def concurrency(name: str) -> int:
    return max(1, CONCURRENCY.get(name, 1))


def available() -> list[str]:
    return sorted(ADAPTERS)


def run_adapter(name: str, doc: Doc, raw: dict[str, Any]) -> list[Triple]:
    if name in ADAPTERS:
        return ADAPTERS[name](doc, raw)
    if name in REFERENCE_ADAPTERS:
        return REFERENCE_ADAPTERS[name](doc, raw)
    raise KeyError(f"未知方案: {name}")
