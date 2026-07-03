"""TTS backend abstraction layer.

Provides the TTSBackend protocol and factory for pluggable TTS backends.
"""

from ai_news_podcast.pipeline.tts_backends.base import TTSBackend
from ai_news_podcast.pipeline.tts_backends.registry import TTSBackendFactory

__all__ = ["TTSBackend", "TTSBackendFactory"]
