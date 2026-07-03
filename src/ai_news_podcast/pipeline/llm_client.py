"""通用 LLM 客户端（OpenAI 兼容协议）

封装 OpenAI-compatible API 调用，提供统一的 prompt→response 接口，
供 podcast 生成和 daily report 生成共用。
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from ai_news_podcast.pipeline.llm_backends import LLMBackendFactory

logger = logging.getLogger(__name__)


def call_llm(prompt: str, llm_cfg: dict[str, Any] | Any) -> str | None:
    """使用通用 OpenAI 兼容协议调用大语言模型。

    Parameters
    ----------
    prompt: 发送给 LLM 的完整 prompt 文本。
    llm_cfg: 配置字典或 dataclass，包含以下可选键：
        - provider (str): 后端提供商，默认 "openai_compatible"
        - api_key_env (str): 环境变量名，默认 "OPENAI_API_KEY"
        - model (str): 模型名称，默认 "deepseek-chat"
        - base_url (str | None): API 基础 URL
        - timeout (int): 请求超时，默认 60 秒
        - temperature (float): 采样温度，默认 0.7
        - max_output_tokens (int): 最大输出 token 数，默认 2048

    Returns
    -------
    str | None: LLM 返回的文本内容；失败时返回 None。
    """
    if not isinstance(llm_cfg, dict):
        llm_cfg = dataclasses.asdict(llm_cfg)  # type: ignore[arg-type]
    provider = llm_cfg.get("provider", "openai_compatible")
    backend = LLMBackendFactory.create(provider, llm_cfg)
    return backend.call(prompt)
