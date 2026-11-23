"""词典缺口报告：候选专名的抽取与过滤。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "kg_lexicon_gaps", ROOT / "scripts" / "kg_lexicon_gaps.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclass 需要模块已注册到 sys.modules 才能解析注解。
    sys.modules["kg_lexicon_gaps"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestExtract:
    def test_extracts_latin_name_after_action(self):
        module = _load_script()
        got = module.extract_candidates("字节跳动发布 Seedance 2，支持多镜头生成。")
        assert "Seedance 2" in got

    def test_stops_at_punctuation(self):
        module = _load_script()
        got = module.extract_candidates("OpenAI 推出 ChatGPT Wallet，支持支付。")
        assert "ChatGPT Wallet" in got

    def test_ignores_pure_chinese_object(self):
        module = _load_script()
        assert module.extract_candidates("百度发布大模型交易平台。") == []

    def test_keeps_decimal_version(self):
        """小数点不能被当分隔符，否则会记下错误的半截名字。"""
        module = _load_script()
        got = module.extract_candidates("字节发布 Seedance 2.5 flash，支持 720P。")
        assert "Seedance 2.5 flash" in got
        assert "Seedance 2" not in got

    def test_noise_filter(self):
        module = _load_script()
        assert module.looks_like_noise("AI")
        assert module.looks_like_noise("api")
        assert module.looks_like_noise("skills")
        assert not module.looks_like_noise("Seedance 2")
        assert not module.looks_like_noise("LongCat-2")


class TestCollect:
    def test_skips_names_already_in_lexicon(self, tmp_path):
        module = _load_script()

        class _FakeLexicon:
            def find_spans(self, text):
                return [object()] if text == "Claude" else []

        day = tmp_path / "ocean" / "2026-11-16"
        day.mkdir(parents=True)
        (day / "items.json").write_text(
            '[{"title": "Anthropic 发布 Claude", "full_text": "x"},'
            ' {"title": "字节发布 Seedance 2", "full_text": "x"}]',
            encoding="utf-8",
        )
        gaps = module.collect_gaps(tmp_path, ["2026-11-16"], _FakeLexicon())
        names = [gap.name for gap in gaps]
        assert names == ["Seedance 2"]
        assert gaps[0].count == 1
        assert gaps[0].days == 1
