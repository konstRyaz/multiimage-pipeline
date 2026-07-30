#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.selection import run_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Выбрать качественные и разнообразные лица")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-per-identity", type=int, default=20)
    parser.add_argument("--duplicate-hamming", type=int, default=4)
    parser.add_argument("--min-frame-gap", type=int, default=8)
    parser.add_argument("--source-margin", type=float, default=0.35)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_selection(args.run_dir, args.overwrite, args.max_per_identity,
                            args.duplicate_hamming, args.min_frame_gap,
                            args.source_margin, args.include_unknown)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
