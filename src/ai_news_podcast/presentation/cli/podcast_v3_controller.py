"""Stage 1–5 新流水线 CLI 入口：podcast-pipeline-v3。

采集入海 → 新闻潮汐 → 选题推荐 → 编辑点评 → 播客文稿（15 分钟）。
旧入口 ``podcast-pipeline`` 保持不变，可随时回滚。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from ai_news_podcast.business.pipeline.tts_engine_service import synthesize
from ai_news_podcast.business.pipeline.v3_runner_service import V3RunResult, run_v3_pipeline
from ai_news_podcast.data.config.models_dao import AppConfig
from ai_news_podcast.presentation.cli.base_controller import AsyncCommand, entrypoint_for
from ai_news_podcast.presentation.cli.podcast_publish_controller import PublishCommand

log = logging.getLogger("cli")


def _load_stage_models(path: Path) -> dict[str, str]:
    """读取 pipeline_v3.yaml 的 llm.stage_models（缺失返回空 dict）。"""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = ((data.get("llm") or {}).get("stage_models")) or {}
    return {str(k): str(v) for k, v in models.items()}


def _load_stage_options(path: Path) -> dict[str, dict[str, Any]]:
    """读取 pipeline_v3.yaml 的 llm.stage_options（温度/最大 token 等）。"""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    options = ((data.get("llm") or {}).get("stage_options")) or {}
    return {str(k): dict(v) for k, v in options.items() if isinstance(v, dict)}


class V3PipelineCommand(AsyncCommand):
    """运行 Stage 1–5 新流水线。"""

    description = "新流水线 Stage 1–5（采集→新闻潮汐→推荐→点评→文稿）"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--date", default=None, help="指定日期 YYYY-MM-DD（默认今天，Asia/Shanghai）"
        )
        parser.add_argument("--data-dir", default="data", help="数据根目录（默认 data）")
        parser.add_argument("--config-dir", default="config", help="配置目录（默认 config）")
        parser.add_argument(
            "--rsshub-base", default=None, help="本地 RSSHub 实例地址（如 http://127.0.0.1:1200）"
        )
        parser.add_argument("--window-hours", type=int, default=72, help="采集时间窗口（默认 72h）")
        parser.add_argument(
            "--no-fulltext",
            dest="fetch_fulltext",
            action="store_false",
            help="不回源抓正文（快速验证用；正文质量下降）",
        )
        parser.add_argument("--max-workers", type=int, default=8, help="采集源级并发（默认 8）")
        parser.add_argument("--skip-script", action="store_true", help="只跑到点评，不生成文稿")
        parser.add_argument("--no-ledger", action="store_true", help="不写入覆盖账本")
        parser.add_argument("--dry-run", action="store_true", help="只打印配置，不执行")
        parser.add_argument(
            "--with-audio", action="store_true", help="Stage 6：合成音频（需 CosyVoice 环境）"
        )
        parser.add_argument("--publish", action="store_true", help="Stage 7：发布站点（隐含音频）")

    async def _stage6_7(
        self,
        args: argparse.Namespace,
        cfg: AppConfig,
        root: Path,
        date_str: str,
        result: V3RunResult,
    ) -> int:
        """Stage 6 音频 + Stage 7 发布；返回 0 表示成功。"""
        if args.with_audio or args.publish:
            if result.script_path is None:
                print("没有文稿（--skip-script），无法合成音频")
                return 1
            episodes_dir = root / cfg.build.episodes_dir
            mp3_path = episodes_dir / f"{date_str}.mp3"
            bgm_rel = cfg.tts.bgm_path
            bgm_path = str(root / bgm_rel) if bgm_rel and (root / bgm_rel).exists() else None
            try:
                print(f"Stage 6: 合成音频 -> {mp3_path}")
                await synthesize(
                    result.script_path.read_text(encoding="utf-8"),
                    backend=cfg.tts.backend,
                    output_path=mp3_path,
                    bgm_path=bgm_path,
                    transcript_path=result.script_path,
                    cfg=cfg,
                    project_root=root,
                )
            except Exception:
                log.exception("Stage 6 音频合成失败")
                return 1

        if args.publish:
            print("Stage 7: 发布站点")
            pub_args = argparse.Namespace(
                date=date_str, base_url=None, sources="config/sources.yaml"
            )
            try:
                rc = await PublishCommand().execute_async(pub_args, cfg, root)
            except Exception:
                log.exception("Stage 7 发布失败")
                return 1
            if rc != 0:
                return rc

        return 0

    async def execute_async(self, args: argparse.Namespace, cfg: AppConfig, root: Path) -> int:
        date_str = args.date or datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
        data_dir = (root / args.data_dir).resolve()
        config_dir = (root / args.config_dir).resolve()

        stage_models = _load_stage_models(config_dir / "pipeline_v3.yaml")
        stage_options = _load_stage_options(config_dir / "pipeline_v3.yaml")
        llm_cfg = cfg.llm
        if stage_models:
            llm_cfg = dataclasses.replace(
                cfg.llm, stage_models={**cfg.llm.stage_models, **stage_models}
            )

        print("=" * 55)
        print(f"新流水线 v3 | date={date_str}")
        print(f"data_dir={data_dir}")
        print(f"config_dir={config_dir}")
        print(f"rsshub_base={args.rsshub_base or '(未提供，C3 将跳过)'}")
        print(f"fetch_fulltext={args.fetch_fulltext}；max_workers={args.max_workers}")
        print(f"模型：默认 {llm_cfg.model}；按 Stage {llm_cfg.stage_models}")
        print("=" * 55)

        if args.dry_run:
            print("dry-run：不执行采集与生成。")
            return 0

        try:
            result = run_v3_pipeline(
                date_str,
                data_dir=data_dir,
                config_dir=config_dir,
                rsshub_base=args.rsshub_base,
                window_hours=args.window_hours,
                fetch_fulltext=args.fetch_fulltext,
                max_workers=args.max_workers,
                llm_cfg=llm_cfg,
                stage_options=stage_options,
                skip_script=args.skip_script,
                update_ledger=not args.no_ledger,
            )
        except Exception:
            log.exception("新流水线执行失败")
            return 1

        rc = await self._stage6_7(args, cfg, root, date_str, result)
        if rc != 0:
            return rc

        print("\n产出：")
        for label, path in (
            ("Stage1 raw", result.raw_path),
            ("Stage2 events", result.events_path),
            ("Stage3 shortlist", result.shortlist_path),
            ("Stage4 commentary", result.commentary_path),
            ("Stage4 report", result.report_path),
            ("Stage5 script", result.script_path),
            ("Stage4/5 brief(compat)", result.brief_path),
        ):
            if path is not None:
                print(f"  {label}: {path}")
        return 0


def entrypoint() -> int:
    return entrypoint_for(V3PipelineCommand)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
