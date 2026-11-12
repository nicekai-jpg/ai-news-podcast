#!/usr/bin/env python3
"""给评测队列做规则预标注：人工只需确认/修正，不用从零填。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_news_podcast.business.ocean.config import load_ocean_config
from ai_news_podcast.business.ocean.labels import label_text, load_lexicon


def build_prelabels(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    lexicon: Any,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        labeled = label_text(
            str(row.get("title") or ""),
            str(row.get("summary") or ""),
            "",
            lexicon=lexicon,
            config=config,
        )
        labels = labeled.labels
        enriched = dict(row)
        enriched.update(
            {
                "prelabel_ai_relevance": labels.ai_relevance,
                "prelabel_primary_domain": labels.primary_domain,
                "prelabel_industry": labels.industry,
                "prelabel_topics": labels.topics,
                "prelabel_tech_stack": labels.tech_stack,
                "prelabel_actions": labels.actions,
                "prelabel_narrative": labels.narrative_angle,
                "prelabel_confidence": labels.label_confidence,
            }
        )
        output.append(enriched)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="评测队列规则预标注")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = load_ocean_config(Path(args.config_dir) / "ocean_v3.yaml")
    lexicon = load_lexicon(Path(str(config.get("_nlp_dir"))))
    rows = [
        json.loads(line)
        for line in Path(args.queue).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    enriched = build_prelabels(rows, config=config, lexicon=lexicon)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in enriched) + "\n",
        encoding="utf-8",
    )
    print(f"预标注 {len(enriched)} 条 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
