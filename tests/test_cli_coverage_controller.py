"""Coverage tests for CLI commands."""

import argparse
from unittest.mock import AsyncMock, patch

import pytest

from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.presentation.cli.podcast_pipeline_controller import PipelineCommand
from ai_news_podcast.presentation.cli.podcast_publish_controller import PublishCommand
from ai_news_podcast.presentation.cli.podcast_report_controller import ReportCommand
from ai_news_podcast.presentation.cli.podcast_tts_controller import TTSCommand
from ai_news_podcast.presentation.cli.podcast_writer_controller import WriterCommand


@pytest.fixture
def dummy_cfg():
    return AppConfig()


@pytest.mark.asyncio
async def test_pipeline_command(dummy_cfg, tmp_path):
    cmd = PipelineCommand()
    args = argparse.Namespace(sources="config/sources.yaml", date="2026-07-22", force_refresh=False)

    with (
        patch(
            "ai_news_podcast.presentation.cli.podcast_pipeline_controller.load_sources",
            return_value=[],
        ),
        patch(
            "ai_news_podcast.presentation.cli.podcast_pipeline_controller.run_pipeline",
            new_callable=AsyncMock,
        ) as mock_run,
    ):
        mock_run.return_value = {"stories": [{"role": "main"}, {"role": "supporting"}]}
        res = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res == 0

        # No stories branch
        mock_run.return_value = {"stories": []}
        res_empty = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res_empty == 1


@pytest.mark.asyncio
async def test_writer_command(dummy_cfg, tmp_path):
    cmd = WriterCommand()
    args = argparse.Namespace(date="2026-07-22")

    # Missing brief
    res_no_brief = await cmd.execute_async(args, dummy_cfg, tmp_path)
    assert res_no_brief == 1

    # Empty stories in brief
    brief_dir = tmp_path / "data" / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    brief_path = brief_dir / "brief_2026-07-22.json"
    brief_path.write_text('{"stories": []}', encoding="utf-8")

    res_empty_stories = await cmd.execute_async(args, dummy_cfg, tmp_path)
    assert res_empty_stories == 1

    # Valid brief
    brief_path.write_text('{"stories": [{"title": "Test"}]}', encoding="utf-8")
    with patch(
        "ai_news_podcast.presentation.cli.podcast_writer_controller.generate_podcast",
        return_value=("👩‍💼 苏晴: 大家好\n👨‍💼 周航: 你好", []),
    ):
        res_success = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res_success == 0
        output_txt = tmp_path / "site" / "episodes" / "2026-07-22.txt"
        assert output_txt.exists()


@pytest.mark.asyncio
async def test_tts_command(dummy_cfg, tmp_path):
    cmd = TTSCommand()
    args = argparse.Namespace(date="2026-07-22", script=None, output=None)

    # Script missing
    res_no_script = await cmd.execute_async(args, dummy_cfg, tmp_path)
    assert res_no_script == 1

    # Script exists
    episodes_dir = tmp_path / "site" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    script_path = episodes_dir / "2026-07-22.txt"
    script_path.write_text("👩‍💼 苏晴: 测试脚本", encoding="utf-8")

    with patch(
        "ai_news_podcast.presentation.cli.podcast_tts_controller.synthesize", new_callable=AsyncMock
    ) as mock_synth:
        res_success = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res_success == 0
        mock_synth.assert_called_once()


@pytest.mark.asyncio
async def test_publish_command(dummy_cfg, tmp_path):
    cmd = PublishCommand()
    args = argparse.Namespace(date="2026-07-22", base_url=None, sources="config/sources.yaml")

    # Audio missing
    res_no_audio = await cmd.execute_async(args, dummy_cfg, tmp_path)
    assert res_no_audio == 1

    episodes_dir = tmp_path / "site" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    (episodes_dir / "2026-07-22.mp3").write_bytes(b"dummy mp3")

    brief_dir = tmp_path / "data" / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "brief_2026-07-22.json").write_text(
        '{"stories": [{"title": "Test"}]}', encoding="utf-8"
    )

    with (
        patch(
            "ai_news_podcast.presentation.cli.podcast_publish_controller.generate_show_notes_html",
            return_value="<p>Notes</p>",
        ),
        patch("ai_news_podcast.presentation.cli.podcast_publish_controller.build_index_html"),
        patch(
            "ai_news_podcast.presentation.cli.podcast_publish_controller.build_feed_xml",
            return_value="<rss></rss>",
        ),
    ):
        res_success = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res_success == 0


@pytest.mark.asyncio
async def test_report_command_cleaning(dummy_cfg, tmp_path):
    cmd = ReportCommand()
    args = argparse.Namespace(date="2026-07-22", clean_think=True, outdir="data/reports")

    brief_dir = tmp_path / "data" / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    (brief_dir / "brief_2026-07-22.json").write_text(
        '{"stories": [{"title": "Test"}], "radar": {"projects": [{"repo": "owner/hot", '
        '"url": "https://github.com/owner/hot", "stars": 1500, "delta_stars": 1000, '
        '"language": "Python", "license": "MIT", "description": "d"}], '
        '"meta": {"pick_repo": "owner/hot"}}}',
        encoding="utf-8",
    )

    with patch(
        "ai_news_podcast.presentation.cli.podcast_report_controller.call_llm",
        return_value="<think>internal reasoning</think>\n# ✨ 科技新闻日报 | 2026年7月22日\n**导语**：今天有重磅新闻。\n## 🚀 重磅解读\n- 新闻1",
    ):
        res = await cmd.execute_async(args, dummy_cfg, tmp_path)
        assert res == 0
        report_file = tmp_path / "data" / "reports" / "daily_report_2026-07-22.md"
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "<think>" not in content
        assert "internal reasoning" not in content
        assert "✨ 科技新闻日报" in content
        assert "项目雷达" in content


def test_write_digest_outputs_creates_files(tmp_path) -> None:
    """write_digest_outputs 必须在 site/data/ 下写出 digest_{date}.json 与 calendar.json。"""
    import json

    root = tmp_path
    (root / "data/briefs").mkdir(parents=True)
    (root / "data/reports").mkdir(parents=True)
    (root / "data/gh_radar").mkdir(parents=True)
    (root / "site/episodes/2026-01-02").mkdir(parents=True)
    (root / "data/briefs/brief_2026-01-02.json").write_text(
        json.dumps({"thesis": "t", "stories": [], "metadata": {"total_deduped": 0}}),
        encoding="utf-8",
    )
    (root / "site/episodes/2026-01-02/playlist.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "id": 1,
                        "host": "A",
                        "text": "x",
                        "start": 1.0,
                        "duration": 2.0,
                        "audios": {"lively": "c.mp3"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "site/episodes/2026-01-02.mp3").write_bytes(b"x" * 200)
    (root / "data/episodes.json").write_text("[]", encoding="utf-8")

    from ai_news_podcast.presentation.site_builder.digest_view import write_digest_outputs

    episodes = [
        {
            "id": "2026-01-02",
            "title": "t",
            "enclosure_url": "https://example.com/episodes/2026-01-02.mp3",
        }
    ]
    written = write_digest_outputs(
        root,
        episodes,
        {"podcast": {}},
        "https://example.com",
        site_dir=root / "site",
        episodes_dir=root / "site/episodes",
    )
    assert (root / "site/data/digest_2026-01-02.json").exists()
    assert (root / "site/data/calendar.json").exists()
    assert written and written[0].name == "digest_2026-01-02.json"
    digest = json.loads((root / "site/data/digest_2026-01-02.json").read_text(encoding="utf-8"))
    assert digest["date"] == "2026-01-02" and digest["schema_version"] == 1
