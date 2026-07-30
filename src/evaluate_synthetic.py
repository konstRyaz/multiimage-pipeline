#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.io import read_csv, write_json


def evaluate(run_dir: Path) -> dict[str, object]:
    truth_rows = read_csv(run_dir / "ground_truth.csv")
    predicted_rows = read_csv(run_dir / "clustering" / "faces_clustered.csv")
    truth = {row["face_id"]: row for row in truth_rows}
    prediction = {row["face_id"]: row["cluster_id"] for row in predicted_rows}
    real_ids = [face_id for face_id, row in truth.items() if row["is_noise"] == "0"]
    tp = fp = fn = tn = 0
    for left, right in itertools.combinations(real_ids, 2):
        same_truth = truth[left]["true_person"] == truth[right]["true_person"]
        same_prediction = prediction[left] == prediction[right]
        if same_truth and same_prediction:
            tp += 1
        elif not same_truth and same_prediction:
            fp += 1
        elif same_truth and not same_prediction:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    people_to_clusters: dict[str, set[str]] = {}
    for face_id in real_ids:
        people_to_clusters.setdefault(truth[face_id]["true_person"], set()).add(prediction[face_id])
    result = {
        "pairwise_precision": round(precision, 6),
        "pairwise_recall": round(recall, 6),
        "pairwise_f1": round(f1, 6),
        "predicted_clusters_for_each_person": {key: sorted(value) for key, value in people_to_clusters.items()},
        "passed": f1 >= 0.98 and all(len(value) == 1 for value in people_to_clusters.values()),
    }
    write_json(run_dir / "synthetic_evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Сверить синтетическую разметку с результатом")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
