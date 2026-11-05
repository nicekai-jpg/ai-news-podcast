"""writer CLI 端到端回归：保存脚本时不能把 [CHAPTER] 章节行洗掉。"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from ai_news_podcast.data.config.loader_dao import load_config
from ai_news_podcast.presentation.cli import podcast_writer_controller

SCRIPT = (
    "[CHAPTER] 开场\n"
    "[Host A] 听众朋友们好，欢迎收听今天的节目。\n"
    "[Host B] 大家好，我们来看看今天发生了什么。\n"
    "[CHAPTER] 今日头条\n"
    "[Host A] 第一条消息来自模型发布。\n"
    "[Host B] 这条值得关注。\n"
)


@pytest.mark.asyncio
async def test_writer_cli_preserves_chapter_markers(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    brief_path = root / "data" / "briefs" / "brief_2026-11-05.json"
    brief_path.parent.mkdir(parents=True)
    brief_path.write_text(
        json.dumps({"stories": [{"role": "main", "representative_title": "T", "context": {}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        podcast_writer_controller,
        "generate_podcast",
        lambda *args, **kwargs: (SCRIPT, []),
    )

    cfg = load_config(Path("config/config.yaml"))
    command = podcast_writer_controller.WriterCommand()
    result = await command.execute_async(Namespace(date="2026-11-05"), cfg, root)

    assert result == 0
    saved = (root / cfg.build.episodes_dir / "2026-11-05.txt").read_text(encoding="utf-8")
    assert "[CHAPTER] 开场" in saved
    assert "[CHAPTER] 今日头条" in saved
