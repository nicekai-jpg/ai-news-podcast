"""Configuration loader utilities.

Replaces the scattered `read_yaml(root / args.config)` calls with a typed
configuration loading mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_news_podcast.config.models import AppConfig


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Typed AppConfig instance.
    """
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid YAML object at {path}")
    return AppConfig.from_dict(raw)
