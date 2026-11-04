"""sources_digest_view 单测：brief.stories → sources 块。"""

from ai_news_podcast.presentation.site_builder.sources_digest_view import (
    build_sources_block,
    classify_group,
)


def _story(title: str, source: str, link: str, role: str = "main", category: str = "official"):
    return {
        "representative_title": title,
        "role": role,
        "role_emoji": "🔴",
        "items": [{"source_name": source, "link": link, "source_category": category}],
    }


class TestClassifyGroup:
    def test_arxiv_by_host(self) -> None:
        assert classify_group("arXiv cs.AI", "https://arxiv.org/abs/2610.1") == "arXiv 论文"

    def test_openai_by_source_name(self) -> None:
        assert classify_group("OpenAI Blog", "https://openai.com/index/x") == "OpenAI"

    def test_google_by_host(self) -> None:
        assert classify_group("DeepMind Blog", "https://blog.google/x") == "Google"

    def test_nvidia_by_source_name(self) -> None:
        assert classify_group("NVIDIA 官方博客", "https://blogs.nvidia.com/x") == "NVIDIA"

    def test_hn(self) -> None:
        assert classify_group("Hacker News", "https://news.ycombinator.com/x") == "Hacker News"

    def test_media_fallback(self) -> None:
        assert classify_group("量子位", "https://qbitai.com/x") == "科技媒体"


class TestBuildSourcesBlock:
    def test_counts_and_order(self) -> None:
        stories = [
            _story("A", "OpenAI Blog", "https://openai.com/a"),
            _story("B", "量子位", "https://qbitai.com/b", role="supporting"),
            {"representative_title": "skip", "role": "skip", "items": []},
        ]
        block = build_sources_block(stories)
        assert block["total_all"] == 2
        assert block["total_main"] == 1
        assert block["items"][0]["role_label"] == "核心信源"
        assert block["items"][1]["role_label"] == "前沿参考"
        assert {g["group"] for g in block["groups"]} == {"OpenAI", "科技媒体"}

    def test_story_without_items_is_dropped(self) -> None:
        block = build_sources_block([{"representative_title": "x", "role": "main", "items": []}])
        assert block["items"] == []

    def test_hard_cap_150(self) -> None:
        stories = [_story(f"t{i}", "量子位", f"https://qbitai.com/{i}") for i in range(200)]
        block = build_sources_block(stories)
        assert len(block["items"]) == 150
        assert "sources_truncated" in block["warnings"]

    def test_role_filter_all_includes_everything(self) -> None:
        stories = [
            _story("A", "量子位", "https://x/a"),
            _story("B", "量子位", "https://x/b", role="quick"),
        ]
        assert build_sources_block(stories)["shown"] == 2
