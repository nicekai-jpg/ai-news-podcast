"""三元组统一评测的口径测试：只测标准本身，不依赖模型。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triple_bench.metrics import (  # noqa: E402
    bootstrap_f1,
    endpoint_match,
    score_all,
    score_document,
)
from scripts.triple_bench.standard import (  # noqa: E402
    Doc,
    GoldItem,
    Triple,
    normalize_relation,
    normalize_text,
    sentence_split,
)


def _gold(item_id: str, triples: list[tuple[str, str, str]], *, split: str = "test") -> GoldItem:
    return GoldItem(
        item_id=item_id,
        title=item_id,
        lang="zh",
        split=split,
        status="reviewed",
        triples=[Triple(s, normalize_relation(r), o) for s, r, o in triples],
    )


class TestStandard:
    def test_normalize_relation_maps_chinese_alias(self):
        assert normalize_relation("发布") == "release"
        assert normalize_relation("Open Source") == "open_source"
        assert normalize_relation("收购") == "acquisition"

    def test_unknown_relation_falls_back_to_other(self):
        assert normalize_relation("随便什么") == "other"

    def test_normalize_text_strips_space_and_width(self):
        assert normalize_text("Ｏｐｅｎ ＡＩ") == normalize_text("open ai")
        assert normalize_text("  《Nature》 ") == "nature"

    def test_sentence_split_uses_same_rules_for_all(self):
        text = "第一条消息发布了新产品。第二条消息很短。第三条这是一句足够长的话"
        out = sentence_split(text)
        assert out[0] == "第一条消息发布了新产品。"
        assert all(len(s) >= 8 for s in out)


class TestMetrics:
    def test_perfect_prediction_scores_one(self):
        items = [_gold("a", [("OpenAI", "release", "GPT-5")])]
        preds = {"a": [Triple("OpenAI", "release", "GPT-5")]}
        score = score_all(items, preds)
        assert score.f1 == 1.0
        assert score.tp == 1 and score.fp == 0 and score.fn == 0

    def test_wrong_relation_counts_as_endpoint_only(self):
        items = [_gold("a", [("OpenAI", "release", "GPT-5")])]
        preds = {"a": [Triple("OpenAI", "update", "GPT-5")]}
        row = score_document(items[0], preds["a"])
        assert row.tp == 0
        assert row.attribution["endpoint_only"] == 1

    def test_swapped_endpoints_are_not_a_match(self):
        items = [_gold("a", [("OpenAI", "release", "GPT-5")])]
        preds = {"a": [Triple("GPT-5", "release", "OpenAI")]}
        row = score_document(items[0], preds["a"])
        assert row.tp == 0
        assert row.attribution["no_candidate"] == 1

    def test_missing_document_scores_zero_not_skipped(self):
        items = [_gold("a", [("OpenAI", "release", "GPT-5")])]
        score = score_all(items, {})
        assert score.gold == 1
        assert score.recall == 0.0
        assert score.docs[0].item_id == "a"

    def test_endpoint_match_exact_rejects_containment(self):
        assert not endpoint_match("浙江大学", "浙江大学与联想团队", "exact")
        assert endpoint_match("浙江大学", "浙江大学与联想团队", "contain")

    def test_bipartite_matching_beats_greedy(self):
        # 两条金标共享一个宾语，贪心会把第二条挤掉。
        items = [
            _gold(
                "a",
                [("浙大", "partnership", "联想"), ("浙大", "partnership", "清华")],
            )
        ]
        preds = {
            "a": [
                Triple("浙大", "partnership", "清华"),
                Triple("浙大", "partnership", "联想"),
            ]
        }
        score = score_all(items, preds)
        assert score.tp == 2
        assert score.fp == 0

    def test_bootstrap_interval_brackets_point_estimate(self):
        gold = [_gold(f"d{i}", [("A", "release", f"P{i}")]) for i in range(10)]
        preds = {
            f"d{i}": [Triple("A", "release", f"P{i}")] if i % 2 == 0 else [] for i in range(10)
        }
        score = score_all(gold, preds)
        mean, low, high = bootstrap_f1(score, rounds=200, seed=7)
        assert low <= score.f1 <= high
        assert low <= mean <= high

    def test_relaxed_mode_recovers_containment(self):
        items = [_gold("a", [("浙大与联想团队", "research", "EM2Mem")])]
        preds = {"a": [Triple("浙大", "research", "EM2Mem")]}
        strict = score_all(items, preds, mode="exact")
        relaxed = score_all(items, preds, mode="contain")
        assert strict.f1 == 0.0
        assert relaxed.f1 == 1.0

    def test_paired_diff_detects_consistent_gap(self):
        from scripts.triple_bench.metrics import paired_bootstrap_diff

        gold = [_gold(f"d{i}", [("A", "release", f"P{i}")]) for i in range(20)]
        strong = {f"d{i}": [Triple("A", "release", f"P{i}")] for i in range(20)}
        weak = {f"d{i}": [Triple("A", "release", f"P{i}")] if i < 4 else [] for i in range(20)}
        left = score_all(gold, strong).by_split("test")
        right = score_all(gold, weak).by_split("test")
        mean, low, _high, significant = paired_bootstrap_diff(left, right, rounds=200)
        assert mean > 0
        assert low > 0
        assert significant is True

    def test_paired_diff_reports_not_significant_for_identical(self):
        from scripts.triple_bench.metrics import paired_bootstrap_diff

        gold = [_gold(f"d{i}", [("A", "release", f"P{i}")]) for i in range(10)]
        preds = {f"d{i}": [Triple("A", "release", f"P{i}")] for i in range(10)}
        left = score_all(gold, preds).by_split("test")
        right = score_all(gold, preds).by_split("test")
        _mean, low, high, significant = paired_bootstrap_diff(left, right, rounds=200)
        assert low == 0 and high == 0
        assert significant is False


class TestGoldValidation:
    def test_validate_flags_evidence_not_in_text(self):
        from scripts.triple_bench.gold import validate_gold

        doc = Doc(item_id="a", title="t", text="OpenAI 发布了 GPT-5。", lang="zh")
        item = _gold("a", [("OpenAI", "release", "GPT-6")])
        item.triples[0].evidence = "OpenAI 发布了 GPT-5。"
        issues = validate_gold([item], {"a": doc})
        reasons = {issue.reason for issue in issues}
        assert "object_not_in_evidence" in reasons
