"""Stage 4 编辑点评 + Stage 5 播客文稿。

对外接口（全部不发网络请求，LLM 调用由调用方注入）：

- ``build_commentaries(shortlist, events, llm_call, *, item_evidence=None, config_path=...)``
  → ``list[Commentary]``，可 ``to_dict()`` 后落盘 ``commentary.json``；
- ``build_script(shortlist, events, commentaries, llm_call, *, item_evidence=None,
  config_path=...)`` → 标准 ``[Host A]/[Host B]/[CHAPTER]`` 纯文本文稿；
- ``quality(script)`` → 字数/章节/主持占比/禁用词/未标记行质量报告。

配置见 ``config/editorial_v3.yaml``（report_top_n=5、15 分钟、3000–3600 字、3–4 章）。
"""

from __future__ import annotations

from ai_news_podcast.business.editorial.commentary import (
    EditorialValidationError,
    build_commentaries,
    extract_json_object,
    parse_commentary,
)
from ai_news_podcast.business.editorial.config import (
    DEFAULT_CONFIG_PATH,
    EditorialConfig,
    LLMCall,
    config_from_mapping,
    load_config,
)
from ai_news_podcast.business.editorial.quality import (
    check_script,
    commentary_quality,
    count_chapters,
    count_spoken_chars,
    find_banned_words,
    host_char_ratio,
)
from ai_news_podcast.business.editorial.script import (
    build_script,
    normalize_script,
    quality,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EditorialConfig",
    "EditorialValidationError",
    "LLMCall",
    "build_commentaries",
    "build_script",
    "check_script",
    "commentary_quality",
    "config_from_mapping",
    "count_chapters",
    "count_spoken_chars",
    "extract_json_object",
    "find_banned_words",
    "host_char_ratio",
    "load_config",
    "normalize_script",
    "parse_commentary",
    "quality",
]
