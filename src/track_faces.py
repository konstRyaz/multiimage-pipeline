#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.tracking import run_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Объединить последовательные появления лиц в дорожки")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-frame-gap", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.38)
    parser.add_argument("--appearance-weight", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_tracking(args.run_dir, args.overwrite, args.max_frame_gap,
                           args.min_similarity, args.appearance_weight)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
