"""OpenAI-compatible LLM backend adapter.

Supports OpenAI, MiniMax, and any other OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

from ai_news_podcast.pipeline.llm_backends.base import LLMBackend

logger = logging.getLogger(__name__)


def _validate_llm_text(t: str) -> None:
    if not t or len(t) < 10:
        raise ValueError(f"LLM 返回过滤后内容为空或过短 ({len(t)} 字符)")


class OpenAILLMBackend(LLMBackend):
    """OpenAI-compatible LLM backend."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def call(self, prompt: str) -> str | None:
        """Call the LLM via OpenAI-compatible API."""
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover
            logger.exception("缺少 openai 库，请运行 uv pip install openai")
            return None

        env_key_name = self.config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(env_key_name, "").strip()

        if not api_key:
            logger.error("未找到对应的环境变量 %s 用于 API 鉴权。请在 .env 中设置。", env_key_name)
            return None

        model_name = self.config.get("model", "deepseek-chat")
        base_url = self.config.get("base_url")
        timeout = self.config.get("timeout", 60)

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
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_output_tokens", 2048),
                )

                raw_text = response.choices[0].message.content
                if raw_text:
                    # 过滤 MiniMax M3/DeepSeek 等推理模型返回的思考链内容
                    text = re.sub(r"\s*<thinking>.*?</thinking>\s*", "", raw_text, flags=re.DOTALL)
                    text = re.sub(r"\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                    text = re.sub(r"\s*[^\w\s]*thinking[^\w\s]*\s*", "", text, flags=re.DOTALL)
                    text = re.sub(r"\s*[^\w\s]*think[^\w\s]*\s*", "", text, flags=re.DOTALL)
                    text = text.strip()

                    # 如果去除 <think> 标签后内容为空或被思考链吞没，尝试直接从原始返回中恢复 [Host A]/[Host B] 对白
                    if not text or len(text) < 10:
                        host_match = re.search(
                            r"(\[Host\s*[AB]\].*)", raw_text, re.DOTALL | re.IGNORECASE
                        )
                        if host_match:
                            text = host_match.group(1).strip()
                            logger.info("从思考链或未闭合标签中自动找回对白: %d 字符", len(text))

                    _validate_llm_text(text)
                    logger.info("LLM 调用成功，返回 %d 字符", len(text))
                    return text

            except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as e:
                logger.warning("LLM 调用失败 (第 %d 次重试): %s", attempt + 1, e)
                time.sleep((attempt + 1) * 2)

        return None
