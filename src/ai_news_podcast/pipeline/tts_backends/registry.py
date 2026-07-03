"""TTS backend registry — Abstract Factory pattern.

Usage:
    from ai_news_podcast.pipeline.tts_backends.registry import TTSBackendFactory

    # Register a new backend
    TTSBackendFactory.register("cosyvoice2", CosyVoice2Backend)

    # Create a backend instance
    backend = TTSBackendFactory.create("cosyvoice2")
    await backend.synthesize(text, output_path=path)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ai_news_podcast.pipeline.tts_backends.base import TTSBackend


class TTSBackendFactory:
    """Factory for creating TTS backend instances."""

    _registry: ClassVar[dict[str, type[TTSBackend]]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[TTSBackend]) -> None:
        """Register a TTS backend class."""
        cls._registry[name] = backend_class

    @classmethod
    def create(cls, name: str, **config: Any) -> TTSBackend:
        """Create a TTS backend instance by name.

        Args:
            name: Backend name (e.g. "cosyvoice2").
            **config: Configuration passed to the backend constructor.

        Returns:
            TTSBackend instance.

        Raises:
            ValueError: If the backend is not registered.
        """
        name = name.strip().lower()
        if name in ("edge-tts", "edge"):
            raise ValueError("Edge-TTS backend has been archived and is no longer available.")

        # Map legacy names
        if name == "hybrid":
            name = "cosyvoice2"

        if name not in cls._registry:
            raise ValueError(
                f"Unsupported TTS backend: {name}. Registered: {list(cls._registry.keys())}"
            )

        return cls._registry[name](**config)


# Register built-in backends
def _register_builtin_backends() -> None:
    """Register all built-in TTS backends."""
    from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import CosyVoice2Backend

    TTSBackendFactory.register("cosyvoice2", CosyVoice2Backend)
    TTSBackendFactory.register("cosyvoice", CosyVoice2Backend)


_register_builtin_backends()
