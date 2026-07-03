"""Base CLI command class using Template Method pattern.

Eliminates duplicated argparse + logging + config loading boilerplate
across all CLI entry points.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ai_news_podcast.config.loader import load_config
from ai_news_podcast.config.models import AppConfig

log = logging.getLogger("cli")


def _setup_logging() -> None:
    """Configure root logging for CLI commands."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _load_dotenv() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


class BaseCommand(ABC):
    """Abstract base class for CLI commands.

    Subclasses must implement:
      - `add_arguments(parser)` – add command-specific CLI flags
      - `execute(args, cfg, root)` – run the command logic (or `execute_async` for async)
    """

    # Override in subclass
    description: str = ""
    config_default: str = "config/config.yaml"

    def __init__(self) -> None:
        self.parser: argparse.ArgumentParser | None = None

    # ------------------------------------------------------------------ #
    # Template Method
    # ------------------------------------------------------------------ #
    def run(self, argv: list[str] | None = None) -> int:
        """Parse arguments, load config, and execute the command."""
        _load_dotenv()
        _setup_logging()

        self.parser = argparse.ArgumentParser(description=self.description)
        self.parser.add_argument(
            "--config",
            default=self.config_default,
            help="配置文件路径",
        )
        self.add_arguments(self.parser)
        args = self.parser.parse_args(argv)

        root = Path(__file__).resolve().parents[3]
        config_path = root / args.config
        cfg = load_config(config_path)

        return self.execute(args, cfg, root)

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        ...

    def execute(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        """Execute the command. Sync subclasses must override this.
        Async subclasses should override ``execute_async`` instead.
        """
        raise NotImplementedError("Subclasses must override execute() or execute_async()")


class AsyncCommand(BaseCommand):
    """Base class for async CLI commands.

    Subclasses must implement ``execute_async()`` (not ``execute()``).
    """

    def run(self, argv: list[str] | None = None) -> int:
        """Parse arguments, load config, and execute the async command."""
        _load_dotenv()
        _setup_logging()

        self.parser = argparse.ArgumentParser(description=self.description)
        self.parser.add_argument(
            "--config",
            default=self.config_default,
            help="配置文件路径",
        )
        self.add_arguments(self.parser)
        args = self.parser.parse_args(argv)

        root = Path(__file__).resolve().parents[3]
        config_path = root / args.config
        cfg = load_config(config_path)

        return asyncio.run(self.execute_async(args, cfg, root))

    @abstractmethod
    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        """Async command execution hook."""
        ...

    # execute() is deliberately NOT abstract here — subclasses implement
    # execute_async() and BaseCommand.execute() is never called for async
    # commands because run() is overridden above.


def entrypoint_for(cmd_class: type[BaseCommand]) -> int:
    """Convenience factory: create a command instance and run it.

    Usage in pyproject.toml console_scripts:
        podcast-pipeline = "ai_news_podcast.cli.podcast_pipeline:entrypoint"
    where entrypoint is:
        entrypoint = lambda: entrypoint_for(PipelineCommand)
    """
    return cmd_class().run()
