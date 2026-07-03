"""LLM backend abstraction layer.

Provides the LLMBackend protocol and factory for pluggable LLM backends.
"""

from ai_news_podcast.pipeline.llm_backends.base import LLMBackend
from ai_news_podcast.pipeline.llm_backends.registry import LLMBackendFactory

__all__ = ["LLMBackend", "LLMBackendFactory"]
