"""LLM backend abstraction layer.

Provides the LLMBackend protocol and factory for pluggable LLM backends.
"""

from ai_news_podcast.business.pipeline.llm_backends.base_service import LLMBackend
from ai_news_podcast.business.pipeline.llm_backends.registry_service import LLMBackendFactory

__all__ = ["LLMBackend", "LLMBackendFactory"]
