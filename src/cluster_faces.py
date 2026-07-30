#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.clustering import run_clustering


def main() -> None:
    parser = argparse.ArgumentParser(description="Объединить дорожки в личности")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-cluster-faces", type=int, default=3)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_clustering(args.run_dir, args.overwrite, args.threshold,
                             args.min_cluster_faces, args.corrections)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
