"""digest_view 单测：组装规则与确定性。"""

import json
from pathlib import Path

from ai_news_podcast.presentation.site_builder.digest_view import (
    build_calendar,
    build_digest,
    detect_chapters,
    duration_stats,
    resolve_issue,
    write_digest_outputs,
)

CHUNKS = [
    {
        "id": 1,
        "host": "A",
        "text": "听众朋友大家好。",
        "start": 1.0,
        "duration": 8.0,
        "audios": {"lively": "chunk_001_lively.mp3"},
    },
    {
        "id": 21,
        "host": "A",
        "text": "说到这个，NVIDIA今天还有个动作。",
        "start": 228.346,
        "duration": 4.18,
        "audios": {"lively": "chunk_021_lively.mp3"},
    },
    {
        "id": 31,
        "host": "A",
        "text": "还有几条快讯也挺有意思。",
        "start": 347.985,
        "duration": 7.88,
        "audios": {"lively": "chunk_031_lively.mp3"},
    },
    {
        "id": 47,
        "host": "A",
        "text": "最后一条，Google DevFest。",
        "start": 519.828,
        "duration": 5.18,
        "audios": {"lively": "chunk_047_lively.mp3"},
    },
]

BRIEF = {
    "thesis": "当 DevFest 遇上落地",
    "metadata": {"total_raw": 125, "total_deduped": 123, "total_clusters": 123},
    "stories": [
        {
            "representative_title": "A",
            "role": "main",
            "items": [{"source_name": "OpenAI Blog", "link": "https://openai.com/a"}],
        },
        {
            "representative_title": "B",
            "role": "supporting",
            "items": [{"source_name": "量子位", "link": "https://qbitai.com/b"}],
        },
    ],
}

CFG = {
    "podcast": {"title": "AI 每日先锋"},
    "gh_radar": {
        "min_stars": 500,
        "created_window_days": 30,
        "recent_push_days": 21,
        "repeat_window_days": 30,
        "readme_probe_limit": 12,
        "pick_count": 1,
        "runner_up_count": 2,
    },
    "tts": {
        "voice_names": {"host_a": {"lively": "青春女声"}, "host_b": {"professional": "专业男声"}},
        "cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}},
    },
}

RADAR_EMPTY = {
    "date": "2026-10-15",
    "projects": [],
    "meta": {
        "pick_repo": "",
        "runner_up_repos": [],
        "degraded": False,
        "reason": "",
        "candidates": 12,
        "excluded_recent": 12,
    },
}


class TestIssue:
    def test_missing_issue_is_unavailable(self) -> None:
        issue = resolve_issue(None)
        assert issue["number"] is None and issue["label"] is None
        assert issue["source"] == "unavailable"

    def test_persisted_issue_is_passed_through(self) -> None:
        issue = resolve_issue(42)
        assert issue["number"] == 42 and issue["label"] == "第 42 期"
        assert issue["source"] == "persisted"


class TestDuration:
    def test_timeline_and_speech(self) -> None:
        timeline, speech = duration_stats(CHUNKS)
        assert round(timeline, 3) == 525.008
        assert round(speech, 3) == 25.24

    def test_empty(self) -> None:
        assert duration_stats([]) == (0.0, 0.0)


class TestChapters:
    def test_cue_heuristic_matches(self) -> None:
        chapters = detect_chapters(CHUNKS)
        starts = [c["start"] for c in chapters]
        assert starts[0] == 1.0
        assert 228.346 in starts and 347.985 in starts and 519.828 in starts
        assert all(c["source"] == "cue_heuristic" for c in chapters)

    def test_fallback_when_no_cues(self) -> None:
        quiet = [
            {
                "id": i,
                "host": "A",
                "text": f"普通段落{i}",
                "start": i * 10.0,
                "duration": 9.0,
                "audios": {},
            }
            for i in range(1, 6)
        ]
        chapters = detect_chapters(quiet)
        assert 2 <= len(chapters) <= 3 and all(c["source"] == "fallback" for c in chapters)

    def test_script_chapters_take_priority(self) -> None:
        chunks = [
            {"id": 1, "host": "A", "text": "开场白。", "start": 0.0, "chapter": "今日头条"},
            {"id": 2, "host": "B", "text": "说到这个……", "start": 120.0, "chapter": "快讯速览"},
            {"id": 3, "host": "A", "text": "最后一条。", "start": 240.0, "chapter": "收尾闲聊"},
        ]
        chapters = detect_chapters(chunks)
        assert [(c["title"], c["start"], c["label"], c["source"]) for c in chapters] == [
            ("今日头条", 0.0, "00:00", "script"),
            ("快讯速览", 120.0, "02:00", "script"),
            ("收尾闲聊", 240.0, "04:00", "script"),
        ]

    def test_single_script_chapter_falls_back_to_heuristic(self) -> None:
        chunks = [
            dict(chunk, chapter="今日头条") if i == 0 else dict(chunk)
            for i, chunk in enumerate(CHUNKS)
        ]
        chapters = detect_chapters(chunks)
        assert all(c["source"] == "cue_heuristic" for c in chapters)

    def test_duplicate_script_titles_count_once_and_fall_back(self) -> None:
        chunks = [dict(chunk, chapter="今日头条") for chunk in CHUNKS]
        chapters = detect_chapters(chunks)
        assert all(c["source"] == "cue_heuristic" for c in chapters)

    def test_legacy_playlist_without_chapter_key_keeps_heuristic(self) -> None:
        """历史 30 期 playlist 没有 chapter 字段 → 必须继续走转场词启发式。"""
        assert all("chapter" not in chunk for chunk in CHUNKS)
        chapters = detect_chapters(CHUNKS)
        assert all(c["source"] == "cue_heuristic" for c in chapters)


class TestHostsCatalog:
    def test_all_variants_present_with_availability(self) -> None:
        from ai_news_podcast.presentation.site_builder.digest_view import _hosts

        hosts = _hosts(CFG)
        assert [host["key"] for host in hosts] == ["A", "B"]
        host_a = hosts[0]["variants"]
        assert {variant["id"] for variant in host_a} == {"professional", "lively"}
        assert [variant["id"] for variant in host_a if variant["available"]] == ["lively"]
        assert [variant["id"] for variant in host_a if variant["default"]] == ["lively"]

    def test_config_without_variants_marks_all_available(self) -> None:
        from ai_news_podcast.presentation.site_builder.digest_view import _hosts

        hosts = _hosts({})
        assert all(variant["available"] for host in hosts for variant in host["variants"])


class TestBuildDigest:
    def _build(self, **overrides):
        kwargs = {
            "date_str": "2026-10-15",
            "brief": BRIEF,
            "report_md": "",
            "radar": RADAR_EMPTY,
            "playlist": {"chunks": CHUNKS},
            "episode": {"title": "AI 新闻快报 | 2026-10-15"},
            "cfg": CFG,
            "base_url": "https://example.com",
        }
        kwargs.update(overrides)
        return build_digest(**kwargs)

    def test_schema_core(self) -> None:
        digest = self._build()
        assert digest["schema_version"] == 1
        assert digest["date"] == "2026-10-15"
        assert digest["issue"]["number"] is None
        assert digest["issue"]["source"] == "unavailable"
        assert digest["availability"]["podcast"] is True
        assert digest["availability"]["report"] is False
        assert digest["report"] is None

    def test_metrics_are_real_numbers(self) -> None:
        metrics = {m["key"]: m["value"] for m in self._build()["metrics"]}
        assert metrics["deduped_items"] == 123
        assert metrics["main_stories"] == 1
        assert metrics["podcast_segments"] == 4
        assert metrics["media_outlets"] == 2

    def test_radar_empty_state(self) -> None:
        radar = self._build()["radar"]
        assert radar["status"] == "empty" and radar["pick"] is None
        assert radar["candidates"]["total"] == 12
        assert radar["rules"]["min_stars"] == 500

    def test_radar_pick(self) -> None:
        radar = {
            "date": "2026-10-15",
            "projects": [
                {
                    "repo": "a/b",
                    "url": "https://github.com/a/b",
                    "description": "d",
                    "language": "Python",
                    "stars": 5000,
                    "delta_stars": 800,
                    "license": "MIT",
                    "readme_intro": "intro",
                    "readme_excerpt": "x",
                    "score": 1.0,
                    "score_parts": {},
                    "affinity": 0.5,
                    "topics": [],
                    "is_new": True,
                    "created_at": "",
                    "pushed_at": "",
                    "has_install_docs": True,
                    "mentions": 0,
                }
            ],
            "meta": {
                "pick_repo": "a/b",
                "runner_up_repos": [],
                "candidates": 30,
                "excluded_recent": 3,
                "fresh_candidates": 27,
                "pool_by_source": {},
            },
        }
        out = self._build(radar=radar)["radar"]
        assert out["status"] == "pick" and out["pick"]["repo"] == "a/b"
        assert out["candidates"]["semantics"] == "v4"

    def test_candidate_audit_preview(self) -> None:
        audit = [
            {
                "repo": f"a/r{i}",
                "stars": 1000 - i,
                "delta_stars": 10,
                "language": "Python",
                "license": "MIT",
                "exclusion_reason": "score_rank",
            }
            for i in range(13)
        ]
        radar = {
            **RADAR_EMPTY,
            "meta": {
                **RADAR_EMPTY["meta"],
                "candidate_audit": audit,
                "audit_semantics": "v5",
            },
        }
        out = self._build(radar=radar)["radar"]
        candidates = out["candidates"]
        assert candidates["detail_available"] is True
        assert candidates["semantics"] == "v5"
        assert len(candidates["preview"]) == 12
        assert set(candidates["preview"][0]) == {
            "repo",
            "stars",
            "delta_stars",
            "language",
            "license",
            "exclusion_reason",
        }
        assert candidates["preview"][0]["repo"] == "a/r0"

    def test_candidate_audit_missing_keeps_legacy(self) -> None:
        out = self._build()["radar"]
        assert out["candidates"]["detail_available"] is False
        assert out["candidates"]["preview"] is None
        blank = {
            **RADAR_EMPTY,
            "meta": {**RADAR_EMPTY["meta"], "candidate_audit": [{"repo": ""}]},
        }
        assert self._build(radar=blank)["radar"]["candidates"]["detail_available"] is False

    def test_deterministic_no_now(self) -> None:
        assert json.dumps(self._build(), sort_keys=True) == json.dumps(
            self._build(), sort_keys=True
        )
        assert "generated_at" not in self._build()["meta"]

    def test_calendar(self) -> None:
        cal = build_calendar(
            episodes=[
                {"id": "2026-10-15", "title": "t", "issue_no": None},
                {"id": "2026-10-14", "title": "t2", "issue_no": None},
            ]
        )
        assert cal["default_date"] == "2026-10-15"
        # D4：v1 不显示期号，字段保留但为 null
        assert [d["issue_number"] for d in cal["dates"]] == [None, None]


class TestWriteOutputs:
    def _seed(self, root: Path, date_str: str) -> tuple[Path, Path]:
        site_dir = root / "site"
        episodes_dir = site_dir / "episodes"
        (root / "data" / "briefs").mkdir(parents=True, exist_ok=True)
        (root / "data" / "briefs" / f"brief_{date_str}.json").write_text(
            json.dumps(BRIEF, ensure_ascii=False), encoding="utf-8"
        )
        (episodes_dir / date_str).mkdir(parents=True, exist_ok=True)
        (episodes_dir / date_str / "playlist.json").write_text(
            json.dumps({"chunks": CHUNKS}, ensure_ascii=False), encoding="utf-8"
        )
        return site_dir, episodes_dir

    def test_writes_digest_and_calendar(self, tmp_path: Path) -> None:
        site_dir, episodes_dir = self._seed(tmp_path, "2026-10-15")
        written = write_digest_outputs(
            tmp_path,
            [{"id": "2026-10-15", "title": "t"}],
            CFG,
            "https://example.com",
            site_dir=site_dir,
            episodes_dir=episodes_dir,
        )
        assert [p.name for p in written] == ["digest_2026-10-15.json"]
        digest = json.loads(
            (site_dir / "data" / "digest_2026-10-15.json").read_text(encoding="utf-8")
        )
        assert digest["date"] == "2026-10-15" and digest["podcast"]["segment_count"] == 4
        calendar = json.loads((site_dir / "data" / "calendar.json").read_text(encoding="utf-8"))
        assert calendar["default_date"] == "2026-10-15"

    def test_fail_soft_per_episode(self, tmp_path: Path) -> None:
        site_dir, episodes_dir = self._seed(tmp_path, "2026-10-15")
        broken = episodes_dir / "2026-10-14"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "playlist.json").write_text("{not json", encoding="utf-8")
        written = write_digest_outputs(
            tmp_path,
            [{"id": "2026-10-14"}, {"id": "2026-10-15"}],
            CFG,
            "https://example.com",
            site_dir=site_dir,
            episodes_dir=episodes_dir,
        )
        assert [p.name for p in written] == ["digest_2026-10-15.json"]
        calendar = json.loads((site_dir / "data" / "calendar.json").read_text(encoding="utf-8"))
        assert [d["date"] for d in calendar["dates"]] == ["2026-10-14", "2026-10-15"]

    def test_stale_digests_removed(self, tmp_path: Path) -> None:
        site_dir, episodes_dir = self._seed(tmp_path, "2026-10-15")
        data_dir = site_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "digest_2026-01-01.json").write_text("{}", encoding="utf-8")
        write_digest_outputs(
            tmp_path,
            [{"id": "2026-10-15", "title": "t"}],
            CFG,
            "https://example.com",
            site_dir=site_dir,
            episodes_dir=episodes_dir,
        )
        assert not (data_dir / "digest_2026-01-01.json").exists()
        assert (data_dir / "digest_2026-10-15.json").exists()


class TestCalendarAvailability:
    def test_availability_comes_from_generated_digests(self) -> None:
        cal = build_calendar(
            episodes=[{"id": "2026-10-15", "title": "t"}, {"id": "2026-10-14", "title": "t2"}],
            availability_by_date={"2026-10-15": {"podcast": True, "report": False}},
        )
        first, second = cal["dates"]
        assert first["availability"] == {"podcast": True, "report": False}
        assert second["availability"]["podcast"] is False
