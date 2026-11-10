"""真实模块跨阶段集成测试（Stage 2→3→4→5）。

只用合成 Stage 1 artifact + 假 LLM，不联网；验证 ocean/recommend/editorial
与编排层 v3_runner_service 的真实接口能够串起来。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_news_podcast.business.contracts import Item, make_item_id
from ai_news_podcast.business.contracts.artifact import write_artifact
from ai_news_podcast.business.editorial.commentary import build_commentaries
from ai_news_podcast.business.editorial.script import build_script
from ai_news_podcast.business.ocean.build import build_ocean
from ai_news_podcast.business.pipeline.v3_runner_service import run_v3_pipeline
from ai_news_podcast.business.recommend.service import recommend

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_items() -> list[Item]:
    rows = [
        (
            "https://a.com/1",
            "DeepSeek 发布 V4.1 模型，参数 100B",
            "DeepSeek 发布了 V4.1 模型，参数 100B。",
            "DeepSeek 发布了 V4.1 模型，参数 100B，支持 256K 上下文。",
            "src_a",
        ),
        (
            "https://b.com/2",
            "DeepSeek 正式发布 V4.1，参数 100B",
            "DeepSeek 正式发布 V4.1 模型，参数 100B。",
            "DeepSeek 正式发布 V4.1 模型，参数 100B。",
            "src_b",
        ),
        (
            "https://c.com/3",
            "DeepSeek V4.1 发布：100B 参数",
            "DeepSeek V4.1 发布，100B 参数。",
            "DeepSeek V4.1 发布，100B 参数。",
            "src_c",
        ),
        (
            "https://d.com/4",
            "OpenAI 发布新 Agent 工具",
            "OpenAI 发布新 Agent 工具。",
            "OpenAI 发布新 Agent 工具，支持多步骤任务。",
            "src_d",
        ),
        (
            "https://e.com/5",
            "OpenAI 推出 Agent 工具，支持多步骤任务",
            "OpenAI 推出 Agent 工具。",
            "OpenAI 推出 Agent 工具，支持多步骤任务。",
            "src_e",
        ),
    ]
    return [
        Item(
            item_id=make_item_id(link),
            channel="c1_rss",
            source=source,
            source_label=source,
            title=title,
            link=link,
            normalized_link=link,
            published_at="2026-11-09T02:00:00+00:00",
            fetched_at="2026-11-09T16:00:00+00:00",
            summary=summary,
            full_text=full_text,
            full_text_status="ok",
            lang="zh",
        )
        for link, title, summary, full_text, source in rows
    ]


def _make_script() -> str:
    line_a = (
        "[Host A] 今天我们聊一条重要消息，DeepSeek 发布了新的模型版本，"
        "参数规模达到一百亿级别，行业内外都在关注。"
    )
    line_b = (
        "[Host B] 这条消息的关键在参数与上下文窗口，我们来看看它和上一代相比"
        "有哪些变化，以及开发者能怎么用。"
    )
    chapters = ["开场", "主推一", "主推二", "收尾"]
    lines: list[str] = []
    for index, chapter in enumerate(chapters):
        lines.append(f"[CHAPTER] {chapter}")
        lines.extend(line_a if (round_no + index) % 2 == 0 else line_b for round_no in range(16))
    return "\n".join(lines)


def _make_commentary(prompt: str) -> str:
    """根据 prompt 里的事件主体返回一条合法点评 JSON。"""
    evidence_ids = []
    text = "该模型发布后引发行业关注。"
    if "DeepSeek" in prompt:
        evidence_ids = [make_item_id("https://a.com/1")]
        text = "DeepSeek 发布 V4.1 模型，参数 100B。"
    elif "OpenAI" in prompt:
        evidence_ids = [make_item_id("https://d.com/4")]
        text = "OpenAI 发布新的 Agent 工具，支持多步骤任务。"
    payload = {
        "headline": "模型与工具更新",
        "one_liner": "一次值得关注的发布。",
        "facts": [{"text": text, "evidence_ids": evidence_ids}],
        "why_matters": "相关能力有所提升，值得开发者关注。",
        "background": "这是近期模型迭代的一部分。",
        "impact": "对开发者的使用方式可能产生一定影响。",
        "counterpoint": "",
        "uncertainty": "",
        "sources": [{"label": "src", "link": "https://a.com/1"}],
    }
    return json.dumps(payload, ensure_ascii=False)


def _make_report_review() -> str:
    return json.dumps(
        {
            "overall": "日报整体准确，注意单一来源表述。",
            "items": [],
            "risks": ["单一来源"],
        },
        ensure_ascii=False,
    )


def _make_podcast_review() -> str:
    return json.dumps(
        {
            "overall": "本期结构清晰。",
            "show_notes": "本期聊聊模型与工具更新。",
            "chapters": [
                {"title": "开场", "intro": "今天从两件事说起。", "highlights": ["模型发布"]}
            ],
            "items": [],
            "fact_check_notes": [],
        },
        ensure_ascii=False,
    )


def _fake_llm(prompt: str) -> str:
    for marker, builder in (
        ("值班编辑", _make_report_review),
        ("播客审稿编辑", _make_podcast_review),
        ("事实底稿", _make_script),
    ):
        if marker in prompt:
            return builder()
    if "evidence_ids" in prompt or '"facts"' in prompt:
        return _make_commentary(prompt)
        evidence_ids = []
        text = "该模型发布后引发行业关注。"
        if "DeepSeek" in prompt:
            evidence_ids = [make_item_id("https://a.com/1")]
            text = "DeepSeek 发布 V4.1 模型，参数 100B。"
        elif "OpenAI" in prompt:
            evidence_ids = [make_item_id("https://d.com/4")]
            text = "OpenAI 发布新的 Agent 工具，支持多步骤任务。"
        payload = {
            "headline": "模型与工具更新",
            "one_liner": "一次值得关注的发布。",
            "facts": [{"text": text, "evidence_ids": evidence_ids}],
            "why_matters": "相关能力有所提升，值得开发者关注。",
            "background": "这是近期模型迭代的一部分。",
            "impact": "对开发者的使用方式可能产生一定影响。",
            "counterpoint": "",
            "uncertainty": "",
            "sources": [{"label": "src", "link": "https://a.com/1"}],
        }
        return json.dumps(payload, ensure_ascii=False)
    if "CHAPTER" in prompt or "播客" in prompt:
        return _make_script()
    # Stage 3 rubric judge 等：返回空对象，降级为纯规则分
    return "{}"


def _build_ocean(tmp_path: Path) -> Any:
    items = _make_items()
    raw = write_artifact(
        tmp_path / "data" / "raw" / "raw-2026-11-09.json.gz", items, date_str="2026-11-09"
    )
    return build_ocean(
        raw,
        date_str="2026-11-09",
        history_dir=tmp_path / "data" / "ocean",
        ledger_path=tmp_path / "data" / "coverage" / "ledger.json",
        config_path="config/ocean_v3.yaml",
    )


def test_ocean_recommend_editorial_interop(tmp_path: Path) -> None:
    snapshot = _build_ocean(tmp_path)
    events = list(snapshot.events)
    assert len(events) >= 2

    shortlist = recommend(events, config_path="config/recommend_v3.yaml")
    assert shortlist, "recommend 不应返回空 shortlist"
    assert [entry.rank for entry in shortlist] == list(range(1, len(shortlist) + 1))

    ocean_dir = tmp_path / "data" / "ocean" / "2026-11-09"
    item_evidence = {
        str(row["item_id"]): str(row.get("full_text") or "")
        for row in json.loads((ocean_dir / "items.json").read_text(encoding="utf-8"))
    }
    commentaries = build_commentaries(
        shortlist,
        events,
        _fake_llm,
        item_evidence=item_evidence,
        config_path="config/editorial_v3.yaml",
    )
    assert commentaries, "至少应有一条点评通过客观性校验"
    assert all(commentary.facts for commentary in commentaries)

    script = build_script(
        shortlist,
        events,
        item_evidence,
        _fake_llm,
        config_path="config/editorial_v3.yaml",
    )
    assert "[CHAPTER]" in script
    assert "[Host A]" in script and "[Host B]" in script


def test_run_v3_pipeline_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    items = _make_items()

    def fake_collect(config_path: Any, date_str: str, output_path: Any, **kwargs: Any) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_artifact(path, items, date_str=date_str)

    monkeypatch.setattr("ai_news_podcast.business.collect.runner.collect_and_write", fake_collect)

    result = run_v3_pipeline(
        "2026-11-09",
        data_dir=tmp_path / "data",
        config_dir=REPO_ROOT / "config",
        llm_call_factory=lambda stage: _fake_llm,
    )

    assert result.raw_path.exists()
    assert result.events_path.exists()
    assert result.shortlist_path.exists()
    assert result.commentary_path.exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.report_review_path is not None and result.report_review_path.exists()
    assert result.podcast_review_path is not None and result.podcast_review_path.exists()
    assert result.script_path is not None and result.script_path.exists()
    assert result.brief_path is not None and result.brief_path.exists()
    brief = json.loads(result.brief_path.read_text(encoding="utf-8"))
    assert brief["stories"], "兼容 brief 应包含 stories（供旧 publish 消费）"
    assert brief["thesis"]
    assert brief["stories"][0]["items"][0]["link"]

    # 旧 publish/show-notes 消费方兼容性：不抛异常且渲染出标题
    from ai_news_podcast.presentation.site_builder.show_notes_view import (
        generate_show_notes_html,
    )

    html = generate_show_notes_html(
        brief,
        episode_title="AI 每日先锋",
        episode_date=datetime(2026, 11, 9),
    )
    assert "AI 每日先锋" in html
    assert brief["stories"][0]["representative_title"] in html

    commentary_rows = json.loads(result.commentary_path.read_text(encoding="utf-8"))
    assert commentary_rows, "编排层应产出点评"
    ledger = json.loads(
        (tmp_path / "data" / "coverage" / "ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["entries"], "选中事件应写入覆盖账本"
