#!/usr/bin/env python3
"""GHA Job 2 entry: synthesize episode MP3 with CosyVoice2."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _bootstrap_imports():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from ai_news_podcast.pipeline.tts_engine import synthesize
    from ai_news_podcast.utils import read_yaml

    return root, synthesize, read_yaml


async def main() -> int:
    root, synthesize, read_yaml = _bootstrap_imports()
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to episode .txt script")
    ap.add_argument("--output", required=True, help="Output .mp3 path")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL_DIR", ""))
    args = ap.parse_args()

    cfg = read_yaml(root / args.config)
    if args.model_dir:
        cfg.setdefault("tts", {}).setdefault("cosyvoice", {})["model_dir"] = args.model_dir

    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = root / script_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    script_text = script_path.read_text(encoding="utf-8")
    
    cfg_bgm = cfg.get("tts", {}).get("bgm_path", "")
    bgm_resolved = None
    if cfg_bgm:
        bgm_p = Path(cfg_bgm)
        bgm_resolved = bgm_p if bgm_p.is_absolute() else root / bgm_p

    await synthesize(
        script_text,
        backend="cosyvoice2",
        output_path=output_path,
        transcript_path=script_path,
        cfg=cfg,
        project_root=root,
        bgm_path=str(bgm_resolved) if (bgm_resolved and bgm_resolved.exists()) else None,
    )
    print(f"Audio saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
