"""Typed configuration models for ai-news-podcast.

Replaces the dict[str, Any] anti-pattern with Pydantic dataclasses
for compile-time safety and IDE autocompletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PodcastConfig:
    """Podcast metadata configuration."""

    title: str = "AI 每日先锋"
    description: str = ""
    language: str = "zh-cn"
    author: str = ""
    explicit: bool = False
    category: str = "Technology"
    keep_last: int = 30
    max_stories: int = 5


@dataclass(frozen=True)
class CosyVoiceRefAudio:
    """Reference audio paths for a single voice style."""

    professional: str = ""
    professional_text: str = ""
    lively: str = ""
    lively_text: str = ""


@dataclass(frozen=True)
class CosyVoiceConfig:
    """CosyVoice-specific TTS configuration."""

    model_dir: str = ""
    ref_audio: dict[str, CosyVoiceRefAudio] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioConfig:
    """Audio post-processing configuration."""

    bgm_volume_db: int = -12
    bgm_fade_in_ms: int = 2000
    bgm_fade_out_ms: int = 3000
    vocal_pad_ms: int = 1000
    loudnorm: str = "I=-16:LRA=11:TP=-1.5"
    sample_rate: int = 24000
    chunk_silence_base: int = 300
    chunk_silence_min: int = 400
    chunk_silence_max: int = 800
    chunk_silence_jitter: int = 100


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech configuration."""

    backend: str = "cosyvoice2"
    bgm_path: str = ""
    cosyvoice: CosyVoiceConfig = field(default_factory=CosyVoiceConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


@dataclass(frozen=True)
class FetchConfig:
    """News fetching configuration."""

    timeout_seconds: int = 10
    user_agent: str = "ai-news-podcast/0.1"
    max_items_per_feed: int = 30
    junk_summary_patterns: list[str] = field(default_factory=list)
    category_keywords: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionConfig:
    """Story selection configuration."""

    prefer_recent_hours: int = 36
    fallback_recent_hours: int = 96
    per_feed_cap: int = 6
    include_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DedupConfig:
    """Deduplication configuration."""

    rapidfuzz_threshold: int = 92
    jieba_keyword_overlap: float = 0.35
    title_sim_threshold: int = 85
    dedup_window_hours: int = 48
    semantic_sim_threshold: float = 0.20
    recent_episodes_limit: int = 14
    semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class ClusteringConfig:
    """DBSCAN clustering configuration."""

    ngram_range: list[int] = field(default_factory=lambda: [2, 4])
    eps: float = 0.35
    min_samples: int = 2


@dataclass(frozen=True)
class ScoringConfig:
    """Story scoring configuration."""

    dimensions: list[str] = field(default_factory=list)
    role_thresholds: dict[str, list[int]] = field(default_factory=dict)
    skip_below: int = 5


@dataclass(frozen=True)
class SourceAuthorityConfig:
    """Source authority ranking configuration."""

    official: int = 1
    research: int = 2
    news: int = 3
    product: int = 3
    analysis: int = 3
    tools: int = 4
    events: int = 4


@dataclass(frozen=True)
class ProcessingConfig:
    """Data processing configuration."""

    thesis_templates: list[str] = field(default_factory=list)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    source_authority: SourceAuthorityConfig = field(default_factory=SourceAuthorityConfig)
    story_memory_days: int = 90
    max_pages_per_episode: int = 80


@dataclass(frozen=True)
class ScriptModeConfig:
    """Script generation mode configuration."""

    name: str = ""
    hook_chars: list[int] = field(default_factory=list)
    thesis_chars: list[int] = field(default_factory=list)
    main_chars: list[int] = field(default_factory=list)
    supporting_chars: list[int] = field(default_factory=list)
    quick_chars: list[int] = field(default_factory=list)
    closing_chars: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ScriptStyleConfig:
    """Script writing style configuration."""

    sentence_length: list[int] = field(default_factory=list)
    banned_words: list[str] = field(default_factory=list)
    total_chars: list[int] = field(default_factory=list)
    target_duration_minutes: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ScriptConfig:
    """Script generation configuration."""

    mode_a: ScriptModeConfig = field(default_factory=ScriptModeConfig)
    mode_b: ScriptModeConfig = field(default_factory=ScriptModeConfig)
    style: ScriptStyleConfig = field(default_factory=ScriptStyleConfig)


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai_compatible"
    api_key_env: str = "MINIMAX_API_KEY"
    model: str = "MiniMax-M3"
    base_url: str = "https://api.minimaxi.com/v1"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    timeout: int = 120


@dataclass(frozen=True)
class BuildConfig:
    """Build output configuration."""

    site_dir: str = "site"
    episodes_dir: str = "site/episodes"
    episodes_index: str = "data/episodes.json"


@dataclass(frozen=True)
class EntitiesConfig:
    """Named entity lists for content analysis."""

    companies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    """Root configuration container for the entire application."""

    podcast: PodcastConfig = field(default_factory=PodcastConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    script: ScriptConfig = field(default_factory=ScriptConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    entities: EntitiesConfig = field(default_factory=EntitiesConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Build AppConfig from a nested dict (e.g. loaded YAML)."""
        return cls(
            podcast=_build_podcast(data),
            tts=_build_tts(data),
            fetch=_build_fetch(data),
            selection=_build_selection(data),
            processing=_build_processing(data),
            script=_build_script(data),
            llm=_build_llm(data),
            build=_build_build(data),
            entities=_build_entities(data),
        )


def _build_podcast(data: dict[str, Any]) -> PodcastConfig:
    p = data.get("podcast", {})
    return PodcastConfig(**{k: v for k, v in p.items() if k in PodcastConfig.__dataclass_fields__})


def _build_tts(data: dict[str, Any]) -> TTSConfig:
    t = data.get("tts", {})
    cosyvoice_data = t.get("cosyvoice", {})
    ref_audio_raw = cosyvoice_data.get("ref_audio", {})
    ref_audio = {k: CosyVoiceRefAudio(**v) for k, v in ref_audio_raw.items() if isinstance(v, dict)}
    cosyvoice = CosyVoiceConfig(
        model_dir=cosyvoice_data.get("model_dir", ""),
        ref_audio=ref_audio,
    )
    audio_data = t.get("audio", {})
    audio = AudioConfig(
        **{k: v for k, v in audio_data.items() if k in AudioConfig.__dataclass_fields__}
    )
    return TTSConfig(
        backend=t.get("backend", "cosyvoice2"),
        bgm_path=t.get("bgm_path", ""),
        cosyvoice=cosyvoice,
        audio=audio,
    )


def _build_fetch(data: dict[str, Any]) -> FetchConfig:
    f = data.get("fetch", {})
    return FetchConfig(**{k: v for k, v in f.items() if k in FetchConfig.__dataclass_fields__})


def _build_selection(data: dict[str, Any]) -> SelectionConfig:
    s = data.get("selection", {})
    return SelectionConfig(
        **{k: v for k, v in s.items() if k in SelectionConfig.__dataclass_fields__}
    )


def _build_processing(data: dict[str, Any]) -> ProcessingConfig:
    p = data.get("processing", {})
    dedup = DedupConfig(
        **{k: v for k, v in p.get("dedup", {}).items() if k in DedupConfig.__dataclass_fields__}
    )
    clustering = ClusteringConfig(
        **{
            k: v
            for k, v in p.get("clustering", {}).items()
            if k in ClusteringConfig.__dataclass_fields__
        }
    )
    scoring = ScoringConfig(
        **{k: v for k, v in p.get("scoring", {}).items() if k in ScoringConfig.__dataclass_fields__}
    )
    source_authority = SourceAuthorityConfig(
        **{
            k: v
            for k, v in p.get("source_authority", {}).items()
            if k in SourceAuthorityConfig.__dataclass_fields__
        }
    )
    return ProcessingConfig(
        thesis_templates=p.get("thesis_templates", []),
        dedup=dedup,
        clustering=clustering,
        scoring=scoring,
        source_authority=source_authority,
        story_memory_days=p.get("story_memory_days", 90),
        max_pages_per_episode=p.get("max_pages_per_episode", 80),
    )


def _build_script(data: dict[str, Any]) -> ScriptConfig:
    s = data.get("script", {})
    mode_a_data = s.get("mode_a", {})
    mode_b_data = s.get("mode_b", {})
    style_data = s.get("style", {})

    mode_a = ScriptModeConfig(
        **{k: v for k, v in mode_a_data.items() if k in ScriptModeConfig.__dataclass_fields__}
    )
    mode_b = ScriptModeConfig(
        **{k: v for k, v in mode_b_data.items() if k in ScriptModeConfig.__dataclass_fields__}
    )
    style = ScriptStyleConfig(
        **{k: v for k, v in style_data.items() if k in ScriptStyleConfig.__dataclass_fields__}
    )
    return ScriptConfig(mode_a=mode_a, mode_b=mode_b, style=style)


def _build_llm(data: dict[str, Any]) -> LLMConfig:
    llm_data = data.get("llm", {})
    return LLMConfig(**{k: v for k, v in llm_data.items() if k in LLMConfig.__dataclass_fields__})


def _build_build(data: dict[str, Any]) -> BuildConfig:
    b = data.get("build", {})
    return BuildConfig(**{k: v for k, v in b.items() if k in BuildConfig.__dataclass_fields__})


def _build_entities(data: dict[str, Any]) -> EntitiesConfig:
    e = data.get("entities", {})
    return EntitiesConfig(
        **{k: v for k, v in e.items() if k in EntitiesConfig.__dataclass_fields__}
    )
