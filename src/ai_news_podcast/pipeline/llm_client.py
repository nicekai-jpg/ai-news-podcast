"""通用 LLM 客户端（OpenAI 兼容协议）

封装 OpenAI-compatible API 调用，提供统一的 prompt→response 接口，
供 podcast script 生成和 daily report 生成共用。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def call_llm(prompt: str, llm_cfg: dict[str, Any]) -> str | None:
    """使用通用 OpenAI 兼容协议调用大语言模型。

    Parameters
    ----------
    prompt: 发送给 LLM 的完整 prompt 文本。
    llm_cfg: 配置字典，包含以下可选键：
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
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        logger.exception("缺少 openai 库，请运行 uv pip install openai")
        return None

    env_key_name = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(env_key_name, "").strip()

    if not api_key:
        logger.error("未找到对应的环境变量 %s 用于 API 鉴权。请在 .env 中设置。", env_key_name)
        return None

    model_name = llm_cfg.get("model", "deepseek-chat")
    base_url = llm_cfg.get("base_url")
    timeout = llm_cfg.get("timeout", 60)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(
                "发起 LLM 调用 (模型: %s, 节点: %s)",
                model_name,
                base_url or "官方 OpenAI",
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_output_tokens", 2048),
            )

            text = response.choices[0].message.content
            if text:
                # 过滤 MiniMax M3 等推理模型返回的 思考链内容
                text = re.sub(r"\s*<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
                text = re.sub(r"\s*[^\w\s]*thinking[^\w\s]*\s*", "", text, flags=re.DOTALL)
                logger.info("LLM 调用成功，返回 %d 字符", len(text))
                return text.strip()

        except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as e:
            logger.warning("LLM 调用失败 (第 %d 次重试): %s", attempt + 1, e)
            time.sleep((attempt + 1) * 2)

    return None
