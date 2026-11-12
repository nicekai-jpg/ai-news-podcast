"""Full-text extraction fallback tests."""

from __future__ import annotations

from ai_news_podcast.business.collect.base import extract_article_text

BODY = "OpenAI 发布新模型。" * 30


def test_json_ld_article_body_fallback() -> None:
    html = f"""
    <html><head>
    <script type="application/ld+json">
    {{"@type": "NewsArticle", "articleBody": "{BODY}"}}
    </script>
    </head><body><div>短</div></body></html>
    """
    text = extract_article_text(html, min_chars=100)
    assert "OpenAI" in text
    assert len(text) >= 100


def test_plain_long_text_still_extracted() -> None:
    text = extract_article_text(f"<html><body><p>{BODY}</p></body></html>", min_chars=100)
    assert "OpenAI" in text


def test_short_page_returns_empty() -> None:
    assert extract_article_text("<html><body>短</body></html>", min_chars=100) == ""
