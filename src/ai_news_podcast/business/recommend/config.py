"""Stage 3 推荐配置：`config/recommend_v3.yaml` 的类型化加载与默认值。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_news_podcast.business.recommend.base import DIMENSIONS, JUDGE_DIMENSIONS

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/recommend_v3.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_RECALL_AI_TOPICS: tuple[str, ...] = (
    "ai",
    "人工智能",
    "大模型",
    "模型",
    "技术",
    "产品",
    "研究",
    "商业",
    "资本",
    "融资",
    "政策",
    "监管",
    "安全",
    "开源",
    "芯片",
    "算力",
    "agent",
    "多模态",
    "机器人",
    "推理",
    "训练",
    "数据",
)

DEFAULT_RECALL_AI_TECH: tuple[str, ...] = (
    "llm",
    "大模型",
    "agent",
    "智能体",
    "多模态",
    "芯片",
    "gpu",
    "机器人",
    "具身",
    "推理",
    "训练",
    "微调",
    "数据",
    "语音",
    "视觉",
    "强化学习",
    "rag",
    "mcp",
    "扩散",
)

DEFAULT_JUNK_KEYWORDS: tuple[str, ...] = (
    "招聘",
    "诚聘",
    "招人",
    "内推",
    "求职",
    "训练营",
    "课程",
    "报名",
    "招生",
    "付费",
    "广告",
    "推广",
    "优惠",
    "榜单",
    "排行榜",
    "带货",
    "hiring",
    "recruit",
    "job",
    "webinar",
    "course",
    "sponsored",
    "advertisement",
)

DEFAULT_AI_TEXT_KEYWORDS: tuple[str, ...] = (
    "ai",
    "ai model",
    "ai models",
    "ai agent",
    "ai agents",
    "language model",
    "language models",
    "large language model",
    "llm",
    "llms",
    "chatbot",
    "chatbots",
    "agentic",
    "moe",
    "mixture of experts",
    "mixture-of-experts",
    "foundation model",
    "world model",
    "generative ai",
    "machine learning",
    "deep learning",
    "neural network",
    "diffusion model",
    "open weights",
    "open-weight",
    "coding agent",
    "copilot",
    "world model",
    "world models",
    "gaussian splatting",
    "case-based reasoning",
    "state space model",
    "state space models",
    "spiking neural",
    "neurips",
    "icml",
    "iclr",
    "cvpr",
    "corl",
    "acl",
    "emnlp",
    "aaai",
    "ijcai",
    "seedream",
    "vla",
    "text-to-image",
    "text-to-video",
    "image model",
    "lithography",
    "neural",
    "segmentation",
    "tactile",
    "dexterous",
    "dexterity",
    "bimanual",
    "slam",
    "foundation-model",
    "qoder",
    "bayesian optimization",
    "bayesian optimisation",
    "artificial intelligence",
    "人工智能",
    "大语言模型",
    "大模型",
    "语言模型",
    "llm",
    "agent",
    "智能体",
    "gpt",
    "chatgpt",
    "claude",
    "gemini",
    "deepseek",
    "openai",
    "anthropic",
    "qwen",
    "通义",
    "文心",
    "豆包",
    "kimi",
    "minimax",
    "智谱",
    "glm",
    "商汤",
    "多模态",
    "生成式",
    "aigc",
    "机器学习",
    "深度学习",
    "神经网络",
    "transformer",
    "扩散模型",
    "rag",
    "mcp",
    "微调",
    "算力",
    "gpu",
    "英伟达",
    "nvidia",
    "芯片",
    "机器人",
    "具身",
    "自动驾驶",
    "智能驾驶",
    "计算机视觉",
    "语音合成",
    "文生图",
    "文生视频",
    "视频生成",
    "图像生成",
    "embedding",
    "开源模型",
    "图像编辑",
    "半导体",
    "数据集",
    "langchain",
    "llama",
    "mistral",
    "huggingface",
    "mamba",
    "language model",
    "multimodal",
    "reinforcement",
    "pretraining",
    "fine-tuning",
    "finetuning",
    "robot",
    "autonomous",
    "self-driving",
)

DEFAULT_DEMOTE_KEYWORDS: tuple[str, ...] = (
    "评测",
    "测评",
    "软文",
    "推广",
    "上手体验",
    "促销",
    "review",
    "hands-on",
)

DEFAULT_PRIORITY_ACTIONS: tuple[str, ...] = (
    "发布",
    "开源",
    "上线",
    "推出",
    "融资",
    "收购",
    "合作",
    "政策",
    "监管",
    "事故",
    "诉讼",
    "突破",
    "release",
    "launch",
    "open-source",
    "funding",
    "acquisition",
    "policy",
    "lawsuit",
)

DEFAULT_CLICKBAIT_KEYWORDS: tuple[str, ...] = (
    "震惊",
    "炸裂",
    "史上最强",
    "吊打",
    "碾压",
    "颠覆",
    "重磅",
    "太强了",
    "必看",
    "shocking",
    "game changer",
    "game-changer",
)

# 泛技术标签不参与「同技术栈 ≤2」截断，否则几乎每条 AI 新闻都会命中同一标签。
DEFAULT_GENERIC_TECH_TAGS: tuple[str, ...] = (
    "ai",
    "人工智能",
    "llm",
    "大模型",
    "模型",
    "技术",
    "机器学习",
    "深度学习",
    "生成式ai",
    "数据",
)

DEFAULT_DECAY_POINTS: tuple[tuple[float, float], ...] = (
    (24.0, 1.0),
    (48.0, 0.95),
    (168.0, 0.85),
    (720.0, 0.7),
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "impact": 0.20,
    "novelty": 0.18,
    "momentum": 0.14,
    "audience_fit": 0.14,
    "narratability": 0.14,
    "evidence": 0.12,
    "risk": 0.08,
}


@dataclass(frozen=True)
class RecallSettings:
    """多路召回阈值（只做资格判断，不打分）。"""

    new_hours: float = 48.0
    evolving_hours: float = 168.0
    long_tail_hours: float = 720.0
    min_subject_weight: float = 0.6
    min_independent_sources: int = 2
    min_evidence_score: float = 2.5
    min_source_count: int = 3
    unknown_age_as_fresh: bool = True
    ai_topics: tuple[str, ...] = DEFAULT_RECALL_AI_TOPICS
    ai_tech_stack: tuple[str, ...] = DEFAULT_RECALL_AI_TECH
    require_ai_relevance: bool = True
    ai_text_keywords: tuple[str, ...] = DEFAULT_AI_TEXT_KEYWORDS


@dataclass(frozen=True)
class CoarseSettings:
    """规则粗排：淘汰门槛 + 排序参数。"""

    max_candidates: int = 20
    subject_cap: int = 2
    fact_consistency_min: float = 0.5
    rumor_tiers: tuple[str, ...] = ("rumor",)
    junk_keywords: tuple[str, ...] = DEFAULT_JUNK_KEYWORDS
    demote_keywords: tuple[str, ...] = DEFAULT_DEMOTE_KEYWORDS
    priority_actions: tuple[str, ...] = DEFAULT_PRIORITY_ACTIONS
    demote_penalty: float = 0.6
    single_source_penalty: float = 0.4
    require_ai_relevance: bool = True
    ai_text_keywords: tuple[str, ...] = DEFAULT_AI_TEXT_KEYWORDS


@dataclass(frozen=True)
class FineSettings:
    """精排：七个规则特征权重 + LLM 融合参数。"""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    llm_top_k: int = 20
    blend_rule: float = 0.6
    blend_llm: float = 0.4
    llm_samples: int = 1  # >1 时多次采样取中位数（成本×N，默认关）
    clickbait_keywords: tuple[str, ...] = DEFAULT_CLICKBAIT_KEYWORDS


@dataclass(frozen=True)
class RerankSettings:
    """重排：MMR/多样性/去重/时效衰减/top N。"""

    shortlist_size: int = 8
    mmr_lambda: float = 0.72
    max_per_subject: int = 2
    max_per_tech_stack: int = 2
    max_per_topic: int = 0  # 0=关闭；评测后再决定是否开启
    max_per_category: int = 0
    title_dedupe_ratio: float = 0.92
    subject_similarity_weight: float = 0.45
    tech_similarity_weight: float = 0.35
    topic_similarity_weight: float = 0.20
    generic_tech_tags: tuple[str, ...] = DEFAULT_GENERIC_TECH_TAGS
    decay_points: tuple[tuple[float, float], ...] = DEFAULT_DECAY_POINTS
    stale_factor: float = 0.5


@dataclass(frozen=True)
class RecommendConfig:
    """推荐全链路配置。"""

    recall: RecallSettings = field(default_factory=RecallSettings)
    coarse: CoarseSettings = field(default_factory=CoarseSettings)
    fine: FineSettings = field(default_factory=FineSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecommendConfig:
        """从 YAML dict 构造配置；缺省字段用默认值，未知字段忽略。"""
        recall_raw = _mapping(raw.get("recall"))
        coarse_raw = _mapping(raw.get("coarse"))
        fine_raw = _mapping(raw.get("fine"))
        rerank_raw = _mapping(raw.get("rerank"))

        weights = dict(DEFAULT_WEIGHTS)
        raw_weights = _mapping(fine_raw.get("weights"))
        for dim in DIMENSIONS:
            if dim in raw_weights:
                weights[dim] = to_float_config(raw_weights[dim], weights[dim])
        unknown = sorted(set(raw_weights) - set(DIMENSIONS))
        if unknown:
            logger.warning("recommend 配置含未知权重字段，已忽略: %s", unknown)

        recall = RecallSettings(
            new_hours=to_float_config(recall_raw.get("new_hours"), 48.0),
            evolving_hours=to_float_config(recall_raw.get("evolving_hours"), 168.0),
            long_tail_hours=to_float_config(recall_raw.get("long_tail_hours"), 720.0),
            min_subject_weight=to_float_config(recall_raw.get("min_subject_weight"), 0.6),
            min_independent_sources=int(
                to_float_config(recall_raw.get("min_independent_sources"), 2)
            ),
            min_evidence_score=to_float_config(recall_raw.get("min_evidence_score"), 2.5),
            min_source_count=int(to_float_config(recall_raw.get("min_source_count"), 3)),
            unknown_age_as_fresh=bool(recall_raw.get("unknown_age_as_fresh", True)),
            ai_topics=_tuple_of_str(recall_raw.get("ai_topics"), DEFAULT_RECALL_AI_TOPICS),
            ai_tech_stack=_tuple_of_str(recall_raw.get("ai_tech_stack"), DEFAULT_RECALL_AI_TECH),
            require_ai_relevance=bool(recall_raw.get("require_ai_relevance", True)),
            ai_text_keywords=_tuple_of_str(
                recall_raw.get("ai_text_keywords"), DEFAULT_AI_TEXT_KEYWORDS
            ),
        )
        coarse = CoarseSettings(
            max_candidates=int(to_float_config(coarse_raw.get("max_candidates"), 20)),
            subject_cap=int(to_float_config(coarse_raw.get("subject_cap"), 2)),
            fact_consistency_min=to_float_config(coarse_raw.get("fact_consistency_min"), 0.5),
            rumor_tiers=_tuple_of_str(coarse_raw.get("rumor_tiers"), ("rumor",)),
            junk_keywords=_tuple_of_str(coarse_raw.get("junk_keywords"), DEFAULT_JUNK_KEYWORDS),
            demote_keywords=_tuple_of_str(
                coarse_raw.get("demote_keywords"), DEFAULT_DEMOTE_KEYWORDS
            ),
            priority_actions=_tuple_of_str(
                coarse_raw.get("priority_actions"), DEFAULT_PRIORITY_ACTIONS
            ),
            demote_penalty=to_float_config(coarse_raw.get("demote_penalty"), 0.6),
            single_source_penalty=to_float_config(coarse_raw.get("single_source_penalty"), 0.4),
            require_ai_relevance=bool(coarse_raw.get("require_ai_relevance", True)),
            ai_text_keywords=_tuple_of_str(
                coarse_raw.get("ai_text_keywords"), DEFAULT_AI_TEXT_KEYWORDS
            ),
        )
        fine = FineSettings(
            weights=weights,
            llm_top_k=int(to_float_config(fine_raw.get("llm_top_k"), 20)),
            blend_rule=to_float_config(fine_raw.get("blend_rule"), 0.6),
            blend_llm=to_float_config(fine_raw.get("blend_llm"), 0.4),
            llm_samples=max(1, int(to_float_config(fine_raw.get("llm_samples"), 1))),
            clickbait_keywords=_tuple_of_str(
                fine_raw.get("clickbait_keywords"), DEFAULT_CLICKBAIT_KEYWORDS
            ),
        )
        rerank = RerankSettings(
            shortlist_size=int(to_float_config(rerank_raw.get("shortlist_size"), 8)),
            mmr_lambda=to_float_config(rerank_raw.get("mmr_lambda"), 0.72),
            max_per_subject=int(to_float_config(rerank_raw.get("max_per_subject"), 2)),
            max_per_tech_stack=int(to_float_config(rerank_raw.get("max_per_tech_stack"), 2)),
            max_per_topic=int(to_float_config(rerank_raw.get("max_per_topic"), 0)),
            max_per_category=int(to_float_config(rerank_raw.get("max_per_category"), 0)),
            title_dedupe_ratio=to_float_config(rerank_raw.get("title_dedupe_ratio"), 0.92),
            subject_similarity_weight=to_float_config(
                rerank_raw.get("subject_similarity_weight"), 0.45
            ),
            tech_similarity_weight=to_float_config(rerank_raw.get("tech_similarity_weight"), 0.35),
            topic_similarity_weight=to_float_config(
                rerank_raw.get("topic_similarity_weight"), 0.20
            ),
            generic_tech_tags=_tuple_of_str(
                rerank_raw.get("generic_tech_tags"), DEFAULT_GENERIC_TECH_TAGS
            ),
            decay_points=_decay_points(rerank_raw.get("decay_points"), DEFAULT_DECAY_POINTS),
            stale_factor=to_float_config(rerank_raw.get("stale_factor"), 0.5),
        )
        recommend = cls(recall=recall, coarse=coarse, fine=fine, rerank=rerank)
        recommend.validate()
        return recommend

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> RecommendConfig:
        """从 YAML 加载配置。"""
        return cls.from_dict(load_yaml(resolve_config_path(path)))

    def validate(self) -> None:
        """校验阈值范围，配置错误尽早失败。"""
        self._validate_fine()
        self._validate_coarse()
        self._validate_rerank()

    def _validate_fine(self) -> None:
        weights = self.fine.weights
        missing = [dim for dim in DIMENSIONS if dim not in weights]
        if missing:
            raise ValueError(f"fine.weights 缺少维度: {missing}")
        negative = [dim for dim, weight in weights.items() if weight < 0]
        if negative:
            raise ValueError(f"fine.weights 不能为负: {negative}")
        if sum(weights.values()) <= 0:
            raise ValueError("fine.weights 之和必须大于 0")
        if self.fine.blend_rule + self.fine.blend_llm <= 0:
            raise ValueError("fine 融合权重不能同时为 0")
        for name in ("blend_rule", "blend_llm"):
            value = getattr(self.fine, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"fine.{name} 必须在 [0, 1] 内: {value}")
        if self.fine.llm_top_k < 1:
            raise ValueError("fine.llm_top_k 必须 >= 1")

    def _validate_coarse(self) -> None:
        if self.coarse.max_candidates < 1:
            raise ValueError("coarse.max_candidates 必须 >= 1")
        if self.coarse.subject_cap < 1:
            raise ValueError("coarse.subject_cap 必须 >= 1")

    def _validate_rerank(self) -> None:
        if not 0.0 <= self.rerank.mmr_lambda <= 1.0:
            raise ValueError(f"rerank.mmr_lambda 必须在 [0, 1] 内: {self.rerank.mmr_lambda}")
        if self.rerank.shortlist_size < 1:
            raise ValueError("rerank.shortlist_size 必须 >= 1")
        if not 0.0 < self.rerank.title_dedupe_ratio <= 1.0:
            raise ValueError("rerank.title_dedupe_ratio 必须在 (0, 1] 内")
        for max_hours, factor in self.rerank.decay_points:
            if max_hours <= 0 or not 0.0 < factor <= 1.0:
                raise ValueError(f"rerank.decay_points 非法: ({max_hours}, {factor})")


def to_float_config(value: Any, default: float) -> float:
    """配置数值读取：非法/缺失回退默认值。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("recommend 配置值 %r 不是数字，使用默认值 %s", value, default)
        return default


def resolve_config_path(path: str | Path) -> Path:
    """解析配置路径：相对路径先按 CWD，再按仓库根目录。"""
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return _PROJECT_ROOT / config_path


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置；文件缺失或顶层不是 mapping 时抛错。"""
    if not path.exists():
        raise FileNotFoundError(f"recommend 配置文件不存在: {path}")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"recommend 配置顶层必须是 mapping: {path}")
    return raw


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tuple_of_str(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        return items or default
    return default


def _decay_points(
    value: Any, default: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        return default
    points: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            logger.warning("recommend 配置 decay_points 条目非法，已忽略: %r", row)
            continue
        points.append((to_float_config(row[0], 0.0), to_float_config(row[1], 0.0)))
    return tuple(sorted(points)) if points else default


def judge_dimension_weights(settings: FineSettings) -> dict[str, float]:
    """把七个规则特征权重映射到 LLM 六维（evidence+risk 合并为 evidence_risk）。"""
    weights = settings.weights
    mapping = dict.fromkeys(JUDGE_DIMENSIONS, 0.0)
    mapping["impact"] = weights.get("impact", 0.0)
    mapping["novelty"] = weights.get("novelty", 0.0)
    mapping["momentum"] = weights.get("momentum", 0.0)
    mapping["audience_fit"] = weights.get("audience_fit", 0.0)
    mapping["narratability"] = weights.get("narratability", 0.0)
    mapping["evidence_risk"] = weights.get("evidence", 0.0) + weights.get("risk", 0.0)
    return mapping
