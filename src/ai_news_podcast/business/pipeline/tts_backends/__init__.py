"""TTS backend abstraction layer.

Provides the TTSBackend protocol and factory for pluggable TTS backends.
"""

from ai_news_podcast.business.pipeline.tts_backends.base_service import TTSBackend
from ai_news_podcast.business.pipeline.tts_backends.registry_service import TTSBackendFactory

__all__ = ["TTSBackend", "TTSBackendFactory"]
