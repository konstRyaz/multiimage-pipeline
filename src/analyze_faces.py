#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Диагностика качества и сходства подготовленных лиц")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--contact-sheet-limit", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(args.run_dir, args.overwrite, args.max_pairs, args.contact_sheet_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
