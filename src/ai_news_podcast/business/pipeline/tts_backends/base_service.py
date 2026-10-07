"""TTS Backend base protocol and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TTSBackend(ABC):
    """Abstract base for TTS backends.

    Implementations must override :meth:`synthesize` to produce audio
    from a dialogue script.
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        output_path: Path,
        bgm_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Convert dialogue text into audio.

        Parameters
        ----------
        text:
            Dialogue script with ``[Host A]`` / ``[Host B]`` markers.
        output_path:
            Target MP3 file path.
        bgm_path:
            Optional background music audio file.
        **kwargs:
            Backend-specific parameters (``cfg``, ``project_root``, etc.).
        """
        ...
