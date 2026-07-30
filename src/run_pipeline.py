#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.analysis import run_analysis
from face_pipeline.clustering import run_clustering
from face_pipeline.selection import run_selection
from face_pipeline.tracking import run_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить все этапы после process_faces.py")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--track-threshold", type=float, default=0.38)
    parser.add_argument("--cluster-threshold", type=float, default=0.45)
    parser.add_argument("--max-frame-gap", type=int, default=8)
    parser.add_argument("--min-cluster-faces", type=int, default=3)
    parser.add_argument("--max-per-identity", type=int, default=20)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {}
    if not args.skip_analysis:
        print("[1/4] Диагностика")
        result["analysis"] = run_analysis(args.run_dir, args.overwrite)
    print("[2/4] Дорожки")
    result["tracking"] = run_tracking(
        args.run_dir, args.overwrite, args.max_frame_gap, args.track_threshold
    )
    print("[3/4] Кластеризация личностей")
    result["clustering"] = run_clustering(
        args.run_dir, args.overwrite, args.cluster_threshold,
        args.min_cluster_faces, args.corrections,
    )
    print("[4/4] Отбор изображений")
    result["selection"] = run_selection(
        args.run_dir, args.overwrite, max_per_identity=args.max_per_identity
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
