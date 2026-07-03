"""LLM backend registry — Factory Pattern.

Usage:
    from ai_news_podcast.pipeline.llm_backends import LLMBackendFactory

    backend = LLMBackendFactory.create("openai_compatible", config)
    response = backend.call(prompt)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ai_news_podcast.pipeline.llm_backends.base import LLMBackend
from ai_news_podcast.pipeline.llm_backends.openai_adapter import OpenAILLMBackend


class LLMBackendFactory:
    """Factory for creating LLM backend instances."""

    _registry: ClassVar[dict[str, type[LLMBackend]]] = {
        "openai_compatible": OpenAILLMBackend,
    }

    @classmethod
    def register(cls, name: str, backend_class: type[LLMBackend]) -> None:
        """Register a new LLM backend."""
        cls._registry[name] = backend_class

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> LLMBackend:
        """Create an LLM backend instance by name.

        Args:
            name: Backend name (e.g. "openai_compatible").
            config: Configuration dict for the backend.

        Returns:
            LLMBackend instance.

        Raises:
            ValueError: If the backend is not registered.
        """
        name = name.strip().lower()
        if name not in cls._registry:
            raise ValueError(
                f"Unsupported LLM backend: {name}. Registered: {list(cls._registry.keys())}"
            )

        return cls._registry[name](config)
