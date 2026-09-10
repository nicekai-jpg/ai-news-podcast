# 按主持人指定音色 (Per-Host Voice Casting) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or execute inline. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `tts.cosyvoice.synth_variants` 从"全局统一列表"升级为"支持按 host 指定"——每句台词只真实合成该主持人选定的变体(耗时仍为单音色水平),发布 MP3 = 苏晴青春女声(host_a.lively)+ 周航专业男声(host_b.professional)的拼装;列表形式语义不变(全体统一),留空仍为全部(向后兼容)。

**Architecture:** 配置解析归一化为 `dict[host, tuple[variant, ...]]`(host 键接受 `host_a`/`A`/`host_b`/`B`);合成循环按 `chunk.host` 取该主持人的变体列表,逐 (chunk, variant) 合成;发布拼装取每位主持人的默认变体(professional 优先,否则列表第一个)对应的分段;playlist 每个 chunk 的 `audios` 只含该主持人实际合成的键;player.js 回退链补"任一可用音色"防 404;html_gen 按 host 分别过滤音色按钮。

**Tech Stack:** 同前。真实冒烟用本机 `~/cosyvoice_venv`(配方见 Task 4),冒烟产物放 `data/_preview/voice_cast_preview.mp3` 供用户试听拍板。

**决策背景:** 用户拍板苏晴换 lively(青春女声)、周航保持 professional。**发布音色将改变且与历史剧集不连续**——冒烟文件须先给用户试听确认后再推送。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/ai_news_podcast/config/models.py` | 修改 | `synth_variants` 类型放宽为 `list[str] \| dict[str, list[str]]` |
| `src/ai_news_podcast/pipeline/cosyvoice_backend.py` | 修改 | 运行时 dataclass 改为 `dict[str, tuple[str, ...]]`;解析归一化(大小写/别名/未知键告警) |
| `src/ai_news_podcast/pipeline/tts_backends/cosyvoice2.py` | 修改 | `_select_variants_for_host`、`_host_variant_plan` 纯函数;合成循环按 host 分派;拼装取各 host 默认变体;playlist 写 per-chunk audios |
| `src/ai_news_podcast/site_builder/html_gen.py` | 修改 | `_voice_labels` 按 host 分别过滤 |
| `src/ai_news_podcast/site_builder/static/player.js` | 修改 | 回退链补"任一可用音色" |
| `config/config.yaml` | 修改 | `synth_variants` 改为字典形式 |
| `AGENTS.md` | 修改 | gotcha 更新 |
| `tests/test_cosyvoice_backend.py`、`tests/test_html_gen.py` | 修改 | 测试改写/新增 |

---

### Task 1: 配置层归一化

**Files:** models.py、cosyvoice_backend.py、config.yaml;Test: tests/test_cosyvoice_backend.py

- [ ] **Step 1: 失败测试**(改写 `TestSynthVariantsConfig`,追加)

```python
class TestSynthVariantsConfig:
    def test_appconfig_list_passthrough(self) -> None:
        from ai_news_podcast.config.models import AppConfig

        cfg = AppConfig.from_dict({"tts": {"cosyvoice": {"synth_variants": ["professional"]}}})
        assert cfg.tts.cosyvoice.synth_variants == ["professional"]

    def test_appconfig_dict_passthrough(self) -> None:
        from ai_news_podcast.config.models import AppConfig

        cfg = AppConfig.from_dict(
            {"tts": {"cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}}}
        )
        assert cfg.tts.cosyvoice.synth_variants == {"host_a": ["lively"], "host_b": ["professional"]}

    def test_default_is_empty_list(self) -> None:
        from ai_news_podcast.config.models import AppConfig

        assert AppConfig.from_dict({}).tts.cosyvoice.synth_variants == []

    def test_runtime_normalizes_list_to_both_hosts(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}, project_root=tmp_path
        )
        assert parsed.synth_variants == {"A": ("professional",), "B": ("professional",)}

    def test_runtime_normalizes_dict_with_aliases(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {"tts": {"cosyvoice": {"synth_variants": {"host_a": ["lively"], "B": ["professional"]}}}},
            project_root=tmp_path,
        )
        assert parsed.synth_variants == {"A": ("lively",), "B": ("professional",)}

    def test_runtime_unknown_host_key_ignored(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config(
            {"tts": {"cosyvoice": {"synth_variants": {"host_c": ["lively"]}}}}, project_root=tmp_path
        )
        assert parsed.synth_variants == {}

    def test_runtime_default_empty(self, tmp_path: Path) -> None:
        parsed = load_cosyvoice_config({}, project_root=tmp_path)
        assert parsed.synth_variants == {"A": (), "B": ()}
```

- [ ] **Step 2: 跑失败 → 实现 → 跑通过**

`models.py` — 类型放宽(构造透传 `_build_tts` 原样可用,无需改):
```python
    synth_variants: list[str] | dict[str, list[str]] = field(default_factory=list)
```

`cosyvoice_backend.py` — 运行时 dataclass:
```python
    synth_variants: dict[str, tuple[str, ...]] = field(default_factory=dict)
```
`load_cosyvoice_config` 的解析段替换为:
```python
    raw_variants = cosy.get("synth_variants") or []

    def _norm_host(key: str) -> str | None:
        k = str(key).strip().lower()
        if k in ("host_a", "a"):
            return "A"
        if k in ("host_b", "b"):
            return "B"
        return None

    def _clean_variants(values: Any) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(str(v).strip() for v in values if str(v).strip())

    if isinstance(raw_variants, dict):
        synth_variants: dict[str, tuple[str, ...]] = {}
        for key, values in raw_variants.items():
            host = _norm_host(key)
            if host is None:
                logger.warning("synth_variants 未知 host 键 %r,已忽略", key)
                continue
            synth_variants[host] = _clean_variants(values)
    else:
        global_variants = _clean_variants(raw_variants)
        synth_variants = {"A": global_variants, "B": global_variants}
```
(`return CosyVoiceConfig(..., synth_variants=synth_variants)` 保持;函数末尾的 `synth_variants = tuple(...)` 旧两行删除。)

`config/config.yaml` — 替换为:
```yaml
    synth_variants:                   # 每位主持人的音色;列表=全体统一,字典=按 host 指定,留空=全部
      host_a: ["lively"]              # 苏晴 → 青春女声
      host_b: ["professional"]        # 周航 → 专业男声
```

- [ ] **Step 3: Commit** `feat(tts): normalize synth_variants to per-host mapping`

---

### Task 2: 合成分派 — `_select_variants_for_host` + `_host_variant_plan` + 循环改造

**Files:** cosyvoice2.py;Test: tests/test_cosyvoice_backend.py

- [ ] **Step 1: 失败测试**(改写 `TestSelectVariants`)

```python
class TestSelectVariants:
    def _cfg(self, synth_variants, refs_keys=("professional", "lively")) -> CosyVoiceConfig:
        return CosyVoiceConfig(
            model_dir=Path(),
            refs={
                "A": {k: (Path(), "") for k in refs_keys},
                "B": {k: (Path(), "") for k in refs_keys},
            },
            synth_variants=synth_variants,
        )

    def test_empty_config_all_variants_both_hosts(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants_for_host

        cfg = self._cfg({})
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional", "lively"]

    def test_global_list_applies_to_both_hosts(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants_for_host

        cfg = self._cfg({"A": ("professional",), "B": ("professional",)})
        assert _select_variants_for_host(cfg, "A") == ["professional"]
        assert _select_variants_for_host(cfg, "B") == ["professional"]

    def test_per_host_selection(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants_for_host

        cfg = self._cfg({"A": ("lively",), "B": ("professional",)})
        assert _select_variants_for_host(cfg, "A") == ["lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional"]

    def test_no_overlap_falls_back_to_all(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants_for_host

        cfg = self._cfg({"A": ("warm",), "B": ()})
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]
        assert _select_variants_for_host(cfg, "B") == ["professional", "lively"]

    def test_empty_refs_default_pairs(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _select_variants_for_host

        cfg = self._cfg({}, refs_keys=())
        assert _select_variants_for_host(cfg, "A") == ["professional", "lively"]

    def test_host_plan_defaults(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _host_variant_plan

        cfg = self._cfg({"A": ("lively", "professional"), "B": ("professional",)})
        variants, defaults = _host_variant_plan(cfg)
        assert variants["A"] == ["lively", "professional"]
        assert defaults["A"] == "lively"        # professional 缺席 → 列表第一个
        assert variants["B"] == ["professional"]
        assert defaults["B"] == "professional"  # professional 在列 → 优先

    def test_host_plan_empty_config(self) -> None:
        from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import _host_variant_plan

        variants, defaults = _host_variant_plan(self._cfg({}))
        assert defaults == {"A": "professional", "B": "professional"}
```

- [ ] **Step 2: 跑失败 → 实现 → 跑通过**

`cosyvoice2.py` — 替换 `_select_variants` 为两个函数:

```python
def _select_variants_for_host(cv_cfg: CosyVoiceConfig, host: str) -> list[str]:
    """决定某位主持人真实要合成的变体;配置与 ref_audio 无交集时回退为全部(防呆)。"""
    available = list(cv_cfg.refs.get(host, {}).keys()) or ["professional", "lively"]
    configured = [v for v in cv_cfg.synth_variants.get(host, ()) if v in available]
    if configured:
        return configured
    if cv_cfg.synth_variants.get(host):
        log.warning(
            "synth_variants[%s]=%s 与 ref_audio 无交集,回退为合成全部音色",
            host,
            list(cv_cfg.synth_variants.get(host, ())),
        )
    return available


def _host_variant_plan(cv_cfg: CosyVoiceConfig) -> tuple[dict[str, list[str]], dict[str, str]]:
    """返回 (每位主持人的合成变体列表, 每位主持人的发布默认变体)。"""
    variants = {host: _select_variants_for_host(cv_cfg, host) for host in ("A", "B")}
    defaults = {
        host: ("professional" if "professional" in vs else (vs[0] if vs else "professional"))
        for host, vs in variants.items()
    }
    return variants, defaults
```

`synthesize` 中替换:
- 旧 `variants = _select_variants(cv_cfg)` 与 `voice_maps = {...}` 块 →
```python
        host_variant_map, host_default = _host_variant_plan(cv_cfg)
        all_variants = sorted({v for vs in host_variant_map.values() for v in vs})
        voice_maps = {var: {"A": f"host_a_{var}", "B": f"host_b_{var}"} for var in all_variants}
```
- 合成循环 →
```python
        segments_by_variant: dict[str, list[Any]] = {}
        chunk_variant_names: list[list[str]] = []

        for idx, chunk in enumerate(chunks, start=1):
            sentences = split_text_into_sentences(chunk.text, max_chars=80)
            vars_for_chunk = host_variant_map[chunk.host]
            chunk_variant_names.append(vars_for_chunk)

            for var in vars_for_chunk:
                chunk_segments: list[Any] = []
                for s_idx, sentence in enumerate(sentences):
                    s_text = sentence.strip()
                    if not s_text:
                        continue

                    tensor = cosy_engine.synthesize_chunk(
                        text=s_text, host=chunk.host, variant=var
                    )
                    wav_path = tmp_root / f"chunk_{idx:03d}_{var}_{s_idx:03d}.wav"
                    torchaudio.save(str(wav_path), tensor, cv_cfg.sample_rate)
                    chunk_segments.append(audio_segment_cls.from_file(str(wav_path)))

                if chunk_segments:
                    combined_chunk = chunk_segments[0]
                    for next_seg in chunk_segments[1:]:
                        combined_chunk += audio_segment_cls.silent(duration=150) + next_seg
                else:
                    combined_chunk = audio_segment_cls.silent(duration=100)
                segments_by_variant.setdefault(var, [None] * len(chunks))[idx - 1] = combined_chunk
```
(注意:原代码把 `combined_chunk` 追加进 `segments_by_variant[var]`(append),新结构是按 chunk 下标定位的稀疏列表——这正是 per-host 分派的关键。旧 `cv2_text = s_text` 中间变量内联掉。)
- 拼装段 →
```python
            combined, timestamps = assemble_dialogue_audio(
                chunks,
                [segments_by_variant[host_default[chunk.host]][i] for i, chunk in enumerate(chunks)],
                chunk_silence_base=int(audio_cfg.get("chunk_silence_base", 300)),
                vocal_pad_ms=int(audio_cfg.get("vocal_pad_ms", 1000)),
                silence_min=int(audio_cfg.get("chunk_silence_min", 400)),
                silence_max=int(audio_cfg.get("chunk_silence_max", 800)),
                silence_jitter=int(audio_cfg.get("chunk_silence_jitter", 100)),
            )
```
- `_write_chunks_and_playlist` 调用处增加 `chunk_variant_names=chunk_variant_names`;其函数体改签名并在 audios 循环处改:
```python
def _write_chunks_and_playlist(
    chunks: list[DialogueChunk],
    segments_by_variant: dict[str, list[Any]],
    chunk_variant_names: list[list[str]],
    timestamps: list[tuple[float, float]],
    voice_maps: dict[str, dict[str, str]],
    output_path: Path,
) -> None:
    ...
    for idx, chunk in enumerate(chunks, start=1):
        start_sec, duration_sec = timestamps[idx - 1]

        audios = {}
        voices = {}
        for var in chunk_variant_names[idx - 1]:
            fn = f"chunk_{idx:03d}_{var}.mp3"
            segments = segments_by_variant[var][idx - 1]
            if segments is None:
                continue

            pydub = importlib.import_module("pydub")
            audio_segment_cls = pydub.AudioSegment
            silence_pad = audio_segment_cls.silent(duration=300)
            padded_seg = silence_pad + segments + silence_pad

            try:
                padded_seg.export(str(chunks_dir / fn), format="mp3", bitrate="64k")
            except TypeError:
                padded_seg.export(str(chunks_dir / fn), format="mp3")

            audios[var] = fn
            voices[var] = voice_maps[var].get(chunk.host, "unknown")
```
(其余 playlist 字段不变。)

- [ ] **Step 3: Commit** `feat(tts): cast per-host voice variants in synthesis and assembly`

---

### Task 3: UI 与播放器兜底

**Files:** html_gen.py、static/player.js;Test: tests/test_html_gen.py

- [ ] **Step 1: 失败测试**(改写 `TestVoiceLabels`)

```python
class TestVoiceLabels:
    def test_defaults_without_config(self) -> None:
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        labels = _voice_labels({})
        assert set(labels["host_a"]) == {"professional", "lively"}

    def test_global_list_filters_both_hosts(self) -> None:
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        cfg = {"tts": {"cosyvoice": {"synth_variants": ["professional"]}}}
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"professional"}
        assert set(labels["host_b"]) == {"professional"}

    def test_per_host_dict_filters_independently(self) -> None:
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        cfg = {"tts": {"cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}}}
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"lively"}
        assert set(labels["host_b"]) == {"professional"}

    def test_appconfig_cfg(self) -> None:
        from ai_news_podcast.config.models import AppConfig
        from ai_news_podcast.site_builder.html_gen import _voice_labels

        cfg = AppConfig.from_dict(
            {"tts": {"cosyvoice": {"synth_variants": {"host_a": ["lively"], "host_b": ["professional"]}}}}
        )
        labels = _voice_labels(cfg)
        assert set(labels["host_a"]) == {"lively"}
        assert set(labels["host_b"]) == {"professional"}
```

- [ ] **Step 2: 跑失败 → 实现 → 跑通过**

`html_gen.py` — `_voice_labels` 的过滤段替换为按 host 归一化(与 pipeline 同规则,site_builder 层不 import pipeline,此处内联):

```python
def _norm_host_key(key: str) -> str | None:
    k = str(key).strip().lower()
    if k in ("host_a", "a"):
        return "host_a"
    if k in ("host_b", "b"):
        return "host_b"
    return None


def _voice_labels(cfg: Any) -> dict[str, dict[str, str]]:
    """注入播放器的音色标签;按 synth_variants(每 host)收敛,按钮只指向真实存在的音频。"""
    if cfg is None:
        return {k: dict(v) for k, v in _DEFAULT_VOICE_NAMES.items()}

    if isinstance(cfg, dict):
        custom = cfg.get("tts", {}).get("voice_names")
        raw_variants = cfg.get("tts", {}).get("cosyvoice", {}).get("synth_variants") or []
    else:
        custom = None
        raw_variants = cfg.tts.cosyvoice.synth_variants

    labels = custom or {k: dict(v) for k, v in _DEFAULT_VOICE_NAMES.items()}
    if isinstance(raw_variants, dict):
        allowed: dict[str, list[str]] = {}
        for key, values in raw_variants.items():
            host = _norm_host_key(key)
            if host is not None:
                allowed[host] = [str(v).strip() for v in (values or []) if str(v).strip()]
    elif raw_variants:
        allowed = {
            "host_a": [str(v).strip() for v in raw_variants],
            "host_b": [str(v).strip() for v in raw_variants],
        }
    else:
        allowed = {}

    for host in ("host_a", "host_b"):
        if allowed.get(host):
            labels[host] = {
                var: name for var, name in labels[host].items() if var in allowed[host]
            }
    return labels
```

`player.js` 的 `loadChunk` 回退链(592-593 附近)改为:
```js
      var audioFile = (chunk.audios && (chunk.audios[variant] || chunk.audios['professional'] || Object.values(chunk.audios)[0])) || chunk.audio || `chunk_${String(index + 1).padStart(3, '0')}.mp3`;
```
(选中的变体没有文件时:先回退 professional,再回退该 chunk 任一可用音色——保证每句都有声可放。)

- [ ] **Step 3: Commit** `feat(site): per-host voice pills and audio fallback`

---

### Task 4: 文档 + 门禁 + 真实冒烟(试听件)

**Files:** AGENTS.md;产物:`data/_preview/voice_cast_preview.mp3`

- [ ] **Step 1:** AGENTS.md 的 `synth_variants` gotcha 更新为:
```markdown
  `tts.cosyvoice.synth_variants` 支持列表(全体统一)或字典(按 host 指定,
  键接受 host_a/A/host_b/B),空=全部合成;当前为 host_a→lively(青春女声)、
  host_b→professional(专业男声)的组合。播放器对未合成变体逐级回退
  (professional → 任一可用),html_gen 按 host 渲染音色按钮。
```

- [ ] **Step 2: 全量门禁**
```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ scripts/ && uv run lint-imports && uv run pytest tests/ -q && uv run pre-commit run --all-files
```

- [ ] **Step 3: 真实冒烟(产物留 data/_preview/ 给用户试听)**
```bash
mkdir -p data/_preview && COSYVOICE_MODEL_DIR="$HOME/cosyvoice_models/CosyVoice2-0.5B" \
PYTHONPATH="$PWD/src:$HOME/cosyvoice_src:$HOME/cosyvoice_src/third_party/Matcha-TTS" \
"$HOME/cosyvoice_venv/bin/python" - <<'EOF' 2>/dev/null | tail -6
import asyncio
import json
from pathlib import Path

from ai_news_podcast.pipeline.tts_backends.cosyvoice2 import CosyVoice2Backend


async def main() -> None:
    out = Path("data/_preview/voice_cast_preview.mp3")
    cfg = {  # 与 config.yaml 同构:dict 形式
        "tts": {"cosyvoice": {
            "synth_variants": {"host_a": ["lively"], "host_b": ["professional"]},
            "ref_audio": {
                "host_a": {"lively": "assets/audio_samples/host_b_v2.mp3",
                           "lively_text": "assets/audio_samples/host_b_v2.txt",
                           "professional": "assets/audio_samples/host_b_v1.mp3",
                           "professional_text": "assets/audio_samples/host_b_v1.txt"},
                "host_b": {"professional": "assets/audio_samples/host_a_v1.mp3",
                           "professional_text": "assets/audio_samples/host_a_v1.txt",
                           "lively": "assets/audio_samples/host_a_v2.mp3",
                           "lively_text": "assets/audio_samples/host_a_v2.txt"},
            },
        }, "audio": {}}
    }
    text = ("[Host A] 大家好,我是苏晴,这一版换成活泼音色啦。\n\n"
            "[Host B] 我是周航,我保持原来的专业男声。")
    await CosyVoice2Backend().synthesize(text, output_path=out, cfg=cfg,
                                         project_root=Path.cwd())
    chunk_dir = out.with_suffix("")
    files = sorted(p.name for p in chunk_dir.iterdir())
    pl = json.loads((chunk_dir / "playlist.json").read_text(encoding="utf-8"))
    print("files:", files)
    print("chunk audios:", [c["audios"] for c in pl["chunks"]])
    assert out.exists() and out.stat().st_size > 10_000
    assert "chunk_001_lively.mp3" in files and "chunk_002_professional.mp3" in files
    assert not any(f.endswith("_professional.mp3") and f.startswith("chunk_001") for f in files)
    print("SMOKE OK — 试听:", out)


asyncio.run(main())
EOF
```
Expected: `chunk_001_lively.mp3`(苏晴活泼版)+ `chunk_002_professional.mp3`(周航专业版),`SMOKE OK`,产物留在 `data/_preview/voice_cast_preview.mp3`。

- [ ] **Step 4:** 把试听文件路径报告给用户,等试听确认后再推送(发布音色将改变,不可回退已发布历史)。

- [ ] **Step 5: Commit** `docs: note per-host voice casting in agents guide`

---

## Self-Review

1. **覆盖**:per-host 分派(Task 2)、列表形式向后兼容(归一化为双 host 同值)、发布默认变体规则(professional 优先否则第一个)、playlist per-chunk audios、播放器 404 兜底、UI 按 host 收敛、试听件 ✓。
2. **占位符**:无。
3. **一致性**:`synth_variants` 三处类型(models 联合 / runtime dict / yaml 字典)经归一化对齐;`_norm_host` 在 pipeline 与 site_builder 各内联一份(层级契约禁止跨层 import),键规则一致(host_a/A、host_b/B);旧 `TestSelectVariants._cfg` 的 runtime 形状(元组→字典)已随新测试改写。
