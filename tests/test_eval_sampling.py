"""Tests for business/eval/sampling.py."""

from __future__ import annotations

from ai_news_podcast.business.eval.sampling import classify_suggestion, stratified_sample


class TestClassifySuggestion:
    def test_core_for_model_news(self) -> None:
        assert classify_suggestion("OpenAI 发布新模型，支持 Agent") == "core"

    def test_adjacent_for_chip_news(self) -> None:
        assert classify_suggestion("英伟达 GPU 算力供应紧张") == "adjacent"

    def test_none_for_car_news(self) -> None:
        assert classify_suggestion("某车企发布新车，明年上市") == "none"

    def test_unknown_for_neutral_news(self) -> None:
        assert classify_suggestion("公司发布季度财报") == "unknown"


class TestStratifiedSample:
    def _rows(self) -> list[dict]:
        rows = []
        for group, count in (("core", 5), ("adjacent", 3), ("none", 4), ("unknown", 2)):
            rows.extend(
                {"group": group, "title": f"{group}-{i}", "date": "2026-11-11"}
                for i in range(count)
            )
        return rows

    def test_respects_quotas(self) -> None:
        picked = stratified_sample(
            self._rows(),
            {"core": 2, "adjacent": 1, "none": 2, "unknown": 1},
        )
        counts = {}
        for row in picked:
            counts[row["group"]] = counts.get(row["group"], 0) + 1
        assert counts == {"core": 2, "adjacent": 1, "none": 2, "unknown": 1}

    def test_fills_when_group_short(self) -> None:
        picked = stratified_sample(self._rows(), {"core": 10, "adjacent": 1}, total_limit=6)
        assert len(picked) == 6
        assert sum(1 for row in picked if row["group"] == "core") == 5

    def test_deterministic(self) -> None:
        first = stratified_sample(self._rows(), {"core": 2, "adjacent": 2})
        second = stratified_sample(list(reversed(self._rows())), {"core": 2, "adjacent": 2})
        assert [row["title"] for row in first] == [row["title"] for row in second]


def test_split_by_day_groups_and_preserves_order() -> None:
    from ai_news_podcast.business.eval.sampling import split_by_day

    rows = [
        {"date": "2026-11-10", "title": "b"},
        {"date": "2026-11-09", "title": "a"},
        {"date": "2026-11-10", "title": "c"},
    ]
    groups = split_by_day(rows)
    assert list(groups) == ["2026-11-09", "2026-11-10"]
    assert [row["title"] for row in groups["2026-11-10"]] == ["b", "c"]
