# 单音色合成 (Single-Variant TTS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or execute inline. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 消除每日 TTS 的双音色冗余——新增 `tts.cosyvoice.synth_variants` 配置开关,只合成配置中列出的音色变体(默认空 = 合成全部,保持向后兼容);网页播放器的音色按钮随配置自动收敛,缺文件的回退逻辑(已存在)兜底。

**Architecture:** 过滤点只有两处:合成循环(`tts_backends/cosyvoice2.py`,决定真实合成哪些变体)与 UI 标签(`site_builder/html_gen.py`,决定渲染哪些按钮)。配置流经 `models.CosyVoiceConfig` → asdict → `pipeline/cosyvoice_backend.load_cosyvoice_config` 解析进运行时 dataclass。最终 MP3 拼装取 `default_variant`(professional 优先,否则列表第一个),发布音频的音色特征零变化。

**Tech Stack:** 纯 Python 配置传递 + 纯函数过滤;本地真实合成冒烟用 `~/cosyvoice_venv`。

**决策背景:** 调查报告(`2026-09-09-tts-investigation-notes.md`)证实每句台词的 lively 变体纯冗余(发布只用 professional,播放器缺文件时自动回退),用户拍板保留单音色;懒合成方案被否。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/ai_news_podcast/config/models.py` | 修改 | `CosyVoiceConfig.synth_variants` 字段 + `_build_tts` 透传 |
| `src/ai_news_podcast/pipeline/cosyvoice_backend.py` | 修改 | 运行时 dataclass 增加 `synth_variants` 并在 `load_cosyvoice_config` 解析 |
| `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py` | 修改 | 抽取 `_select_variants()` 纯函数并在 `synthesize` 使用 |
| `src/ai_news_podcast/site_builder/html_gen.py` | 修改 | 抽取 `_voice_labels()` 纯函数,按 `synth_variants` 过滤注入 UI 的标签 |
| `config/config.yaml` | 修改 | `tts.cosyvoice.synth_variants: ["professional"]` |
| `AGENTS.md` | 修改 | gotcha 补充开关说明 |
| `tests/test_cosyvoice_backend.py` / `tests/test_html_gen.py` | 修改 | 新增单测 |

层级合规:`site_builder` 不 import `pipeline`(契约),过滤逻辑在两层各自内联为纯函数。

---

### Task 1: 配置层 — models + 运行时 dataclass + config.yaml

**Files:**
- Modify: `src/ai_news_podcast/config/models.py`、`src/ai_news_podcast/pipeline/cosyvoice_backend.py`、`config/config.yaml`
- Test: `tests/test_cosyvoice_backend.py`

- [ ] **Step 1: 失败测试(追加到 tests/test_cosyvoice_backend.py)**

```python
class TestSynthVariantsConfig:
    def test_appconfig_passthrough(self) -> None:
        from ai_news_podcast.config.models import AppConfig

        cfg = AppConfig.from_dict(
            {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}
        )
        assert cfg.tts.cosyvoice.synth_variants == ["professional"]

    def test_default_is_empty(self) -> None:
        from ai_news_podcast.config.models import AppConfig

        cfg = AppConfig.from_dict({})
        assert cfg.tts.cosyvoice.synth_variants == []

    def test_load_cosyvoice_config_parses(self, tmp_path) -> None:
        from ai_news_podcast.pipeline.cosyvoice_backend import load_cosyvoice_config

        cfg = {"tts": {"cosyvoice": {"synth_variants": ["professional"], "model_dir": ""}}}
        parsed = load_cosyvoice_config(cfg, project_root=tmp_path)
        assert parsed.synth_variants == ("professional",)

    def test_load_cosyvoice_config_default_empty(self, tmp_path) -> None:
        from ai_news_podcast.pipeline.cosyvoice_backend import load_cosyvoice_config

        parsed = load_cosyvoice_config({}, project_root=tmp_path)
        assert parsed.synth_variants == ()
```

- [ ] **Step 2: 跑测试确认失败**(字段不存在 → AttributeError/TypeError)

- [ ] **Step 3: 实现**

`config/models.py` — `CosyVoiceConfig` 增加字段:
```python
    synth_variants: list[str] = field(default_factory=list)
```
`_build_tts` 的 `CosyVoiceConfig(...)` 构造增加:
```python
        synth_variants=cosyvoice_data.get("synth_variants", []),
```

`pipeline/cosyvoice_backend.py` — 运行时 dataclass 增加:
```python
    synth_variants: tuple[str, ...] = ()
```
`load_cosyvoice_config` 返回处改为:
```python
    synth_variants_raw = cosy.get("synth_variants") or []
    synth_variants = tuple(str(v).strip() for v in synth_variants_raw if str(v).strip())

    return CosyVoiceConfig(
        model_dir=model_dir,
        refs=parsed_refs,
        synth_variants=synth_variants,
    )
```

`config/config.yaml` — `tts.cosyvoice` 块内(`ref_audio:` 之前)插入:
```yaml
    synth_variants: ["professional"]  # 只合成列出的音色变体;留空则合成 ref_audio 的全部(旧行为)
```

- [ ] **Step 4: 跑测试确认通过;全量 pytest 不回归**

- [ ] **Step 5: Commit** `feat(tts): add synth_variants config knob`

---

### Task 2: 合成过滤 — `_select_variants` 纯函数

**Files:**
- Modify: `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py`
- Test: `tests/test_cosyvoice_backend.py`

- [ ] **Step 1: 失败测试**

```python
class TestSelectVariants:
    def _cfg(self, synth_variants, refs_keys=("professional", "lively")):
        from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoiceConfig
        from pathlib import Path

        return CosyVoiceConfig(
            model_dir=Path(),
            refs={"A": {k: (Path(), "") for k in refs_keys},
                  "B": {k: (Path(), "") for k in refs_keys}},
            synth_variants=tuple(synth_variants),
        )

    def test_empty_config_synthesizes_all(self):
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants

        assert _select_variants(self._cfg(())) == ["professional", "lively"]

    def test_configured_filter(self):
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants

        assert _select_variants(self._cfg(["professional"])) == ["professional"]

    def test_unknown_variant_falls_back_to_all(self):
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants

        assert _select_variants(self._cfg(["warm"])) == ["professional", "lively"]

    def test_partial_overlap_keeps_intersection(self):
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants

        assert _select_variants(self._cfg(["lively", "warm"])) == ["lively"]

    def test_no_refs_default_pairs(self):
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants

        assert _select_variants(self._cfg((), refs_keys=())) == ["professional", "lively"]
```

- [ ] **Step 2: 跑失败 → 实现 → 跑通过**

`cosyvoice2.py` 增加(模块级纯函数):
```python
def _select_variants(cv_cfg: CosyVoiceConfig) -> list[str]:
    """决定真实要合成的音色变体;配置与 ref_audio 无交集时回退为全部(防呆)。"""
    available = (
        list(cv_cfg.refs["A"].keys()) if "A" in cv_cfg.refs else ["professional", "lively"]
    )
    configured = [v for v in cv_cfg.synth_variants if v in available]
    if configured:
        return configured
    if cv_cfg.synth_variants:
        log.warning(
            "tts.cosyvoice.synth_variants=%s 与 ref_audio 无交集,回退为合成全部音色",
            list(cv_cfg.synth_variants),
        )
    return available
```
import 行增加 `CosyVoiceConfig`(`from ai_news_podcast.pipeline.cosyvoice_backend import CosyVoice2Engine, CosyVoiceConfig, load_cosyvoice_config`)。`synthesize` 中 L128-130 替换为:
```python
        variants = _select_variants(cv_cfg)
```
`default_variant` 行不变(professional 优先,否则列表第一个)。

- [ ] **Step 3: Commit** `feat(tts): synthesize only configured voice variants`

---

### Task 3: UI 标签收敛 — `_voice_labels` 过滤

**Files:**
- Modify: `src/ai_news_podcast/site_builder/html_gen.py`
- Test: `tests/test_html_gen.py`

- [ ] **Step 1: 失败测试(追加到 tests/test_html_gen.py)**

```python
class TestVoiceLabels:
    def test_defaults_without_config(self) -> None:
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        labels = _voice_labels({})
        assert set(labels["host_a"]) == {"professional", "lively"}

    def test_filtered_by_synth_variants(self) -> None:
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        cfg = {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"professional"}
        assert set(labels["host_b"]) == {"professional"}

    def test_appconfig_cfg(self) -> None:
        from ai_news_podcast.config.models import AppConfig
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        cfg = AppConfig.from_dict(
            {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}
        )
        labels = _voice_labels(cfg)
        assert set(labels["host_b"]) == {"professional"}
```

- [ ] **Step 2: 跑失败 → 实现 → 跑通过**

`html_gen.py` 增加模块级函数,并把现有 `voices_config` 赋值块(L75-90)整体替换:
```python
_DEFAULT_VOICE_NAMES = {
    "host_a": {"professional": "亲切女声", "lively": "青春女声"},
    "host_b": {"professional": "专业男声", "lively": "活力男声"},
}


def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    """注入播放器的音色标签;按 synth_variants 收敛,保证按钮只指向真实存在的音频。"""
    if cfg is None:
        return {k: dict(v) for k, v in _DEFAULT_VOICE_NAMES.items()}

    if isinstance(cfg, dict):
        custom = cfg.get("tts", {}).get("voice_names")
        synth_variants = cfg.get("tts", {}).get("cosyvoice", {}).get("synth_variants") or []
    else:
        custom = None
        synth_variants = list(cfg.tts.cosyvoice.synth_variants)

    labels = custom or {k: dict(v) for k, v in _DEFAULT_VOICE_NAMES.items()}
    if synth_variants:
        labels = {
            host: {var: name for var, name in variants.items() if var in synth_variants}
            for host, variants in labels.items()
        }
    return labels
```
调用处:`voices_config_json = json.dumps(_voice_labels(cfg), ensure_ascii=False)`。
注意:`Any` 若未导入则补 `from typing import Any`。

- [ ] **Step 3: Commit** `feat(site): render only configured voice pills`

---

### Task 4: 文档 + 全量验证 + 真实冒烟

**Files:** Modify `AGENTS.md`

- [ ] **Step 1:** AGENTS.md `ref_audio` 交叉映射 gotcha 后追加一句:
```markdown
  `tts.cosyvoice.synth_variants` 控制真实合成的变体(默认空=全部);当前设为
  `["professional"]` 以省一半 CPU 推理——播放器未合成变体的按钮按 professional 回退
  (`player.js`),切换按钮只渲染配置内的标签(html_gen)。
```

- [ ] **Step 2: 全量门禁**
```bash
uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/ scripts/ && uv run lint-imports && uv run pytest tests/ -q && uv run pre-commit run --all-files
```

- [ ] **Step 3: 真实合成冒烟(本机 cosyvoice venv,2 行对话,~3-4 分钟)**
```bash
COSYVOICE_MODEL_DIR="$HOME/cosyvoice_models/CosyVoice2-0.5B" \
PYTHONPATH="$HOME/cosyvoice_src:$HOME/cosyvoice_src/third_party/Matcha-TTS" \
"$HOME/cosyvoice_venv/bin/python" - <<'EOF'
import asyncio
import json
from pathlib import Path
import tempfile

from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import CosyVoice2Backend


async def main() -> None:
    out = Path(tempfile.mkdtemp()) / "2026-09-09.mp3"
    cfg = {"tts": {"cosyvoice": {
        "synth_variants": ["professional"],
        "ref_audio": {
            "host_a": {"professional": "assets/audio_samples/host_b_v1.mp3",
                       "professional_text": "assets/audio_samples/host_b_v1.txt"},
            "host_b": {"professional": "assets/audio_samples/host_a_v1.mp3",
                       "professional_text": "assets/audio_samples/host_a_v1.txt"},
        },
    }, "audio": {}}}
    text = "[Host A] 大家好,这是单音色冒烟测试。\n\n[Host B] 好的,只有一种音色。"
    await CosyVoice2Backend().synthesize(text, output_path=out, cfg=cfg,
                                         project_root=Path.cwd())
    chunk_dir = out.with_suffix("")
    files = sorted(p.name for p in chunk_dir.iterdir())
    print("files:", files)
    pl = json.loads((chunk_dir / "playlist.json").read_text(encoding="utf-8"))
    keys = {tuple(sorted(c["audios"])) for c in pl["chunks"]}
    print("playlist audio keys:", keys)
    assert out.exists() and out.stat().st_size > 100_000
    assert all(k == ("professional",) for k in keys), keys
    print("SMOKE OK")


asyncio.run(main())
EOF
```
Expected: `SMOKE OK`;chunk 目录只有 `_professional` 文件;playlist 的 audios 键只有 professional。

- [ ] **Step 4: Commit** `docs: note single-variant tts in agents guide`(或并入 Task 3 提交)

---

## Self-Review

1. **覆盖**:合成过滤(Task 2)、配置贯通(Task 1)、UI 收敛(Task 3)、防呆回退(无交集→全部)、默认空=旧行为(向后兼容)✓;冒烟验证端到端 ✓。
2. **占位符**:无。
3. **一致性**:`synth_variants` 在 models(list)/runtime(tuple)/yaml(list) 三处语义一致;`default_variant` 选择逻辑未动,professional 缺席时用列表第一个(与既有行为一致)。
