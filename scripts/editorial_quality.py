#!/usr/bin/env python3
"""点评质量检查：日报/播客点评结构完整性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_news_podcast.business.editorial.review_quality import review_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="点评质量检查")
    parser.add_argument("--editorial-dir", required=True)
    args = parser.parse_args()
    root = Path(args.editorial_dir)
    failed = False
    for name, kind in (("report_review.json", "report"), ("podcast_review.json", "podcast")):
        path = root / name
        if not path.is_file():
            print(f"{name}: 缺失")
            failed = True
            continue
        report = review_quality(json.loads(path.read_text(encoding="utf-8")), kind=kind)
        print(f"{name}: score={report['score']} issues={len(report['issues'])}")
        for issue in report["issues"]:
            print(f"  - {issue}")
        if report["score"] < 0.75:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
