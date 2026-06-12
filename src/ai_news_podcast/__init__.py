"""AI Daily Pioneer — Daily AI news podcast generator."""

try:
    from importlib.metadata import version

    __version__ = version("ai-news-podcast")
except Exception:
    __version__ = "0.0.0"
