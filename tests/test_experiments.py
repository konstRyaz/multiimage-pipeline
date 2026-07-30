from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face_pipeline.experiment_config import load_experiment_config, resolve_paths
from face_pipeline.experiment_metrics import aggregate_metrics, classification_metrics
from face_pipeline.experiments import compare_experiments, run_experiment
from face_pipeline.io import REQUIRED_FACE_FIELDS, read_csv, save_npy, write_csv
from face_pipeline.selection import greedy_select_detailed
from face_pipeline.clustering import run_clustering
from face_pipeline.selection import run_selection
from face_pipeline.tracking import run_tracking
from generate_synthetic import run as generate_synthetic


def resolved_config(path: Path, run_dir: Path, experiments_dir: Path, labels: Path | None = None):
    return resolve_paths(
        load_experiment_config(path),
        path,
        run_dir,
        experiments_dir,
        labels,
    )


class ExperimentTests(unittest.TestCase):
    def test_deduplication_and_nonselection_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("RGB", (16, 16), (120, 120, 120))
            image.save(root / "same.png")
            image.save(root / "same2.png")
            rows = [
                {
                    "face_id": "a", "aligned_path": "same.png", "quality_score": "0.9",
                    "video_name": "v", "frame_number": "1", "yaw": "0",
                },
                {
                    "face_id": "b", "aligned_path": "same2.png", "quality_score": "0.8",
                    "video_name": "v", "frame_number": "2", "yaw": "0",
                },
            ]
            selected, decisions = greedy_select_detailed(
                rows, root, maximum=2, duplicate_hamming=0, min_frame_gap=0
            )
            self.assertEqual([row["face_id"] for row, _ in selected], ["a"])
            by_id = {row["face_id"]: row for row in decisions}
            self.assertIn("duplicate", by_id["b"]["not_selected_reasons"])

    def test_classification_metrics_positive_means_acceptable(self) -> None:
        rows = [
            {"face_id": "tp", "accepted": True},
            {"face_id": "fp", "accepted": True},
            {"face_id": "fn", "accepted": False},
            {"face_id": "tn", "accepted": False},
            {"face_id": "unlabeled", "accepted": True},
        ]
        result = classification_metrics(
            rows, {"tp": True, "fp": False, "fn": True, "tn": False}
        )
        assert result is not None
        self.assertEqual(result["confusion_matrix"], {
            "true_positive": 1, "false_positive": 1,
            "false_negative": 1, "true_negative": 1,
        })
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["unlabeled_faces"], 1)

    def test_aggregate_metrics_empty(self) -> None:
        result = aggregate_metrics([], [], [], [], {"total": 0.1}, {"cache_hit": True})
        self.assertEqual(result["counts"]["faces"], 0)
        self.assertEqual(result["counts"]["accepted_rate"], 0.0)
        self.assertEqual(result["identities"]["distribution_before_filter"]["identities"], 0)

    def test_single_face_and_empty_experiments_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for empty in (True, False):
                run_dir = root / ("empty" if empty else "single")
                run_dir.mkdir()
                fields = list(REQUIRED_FACE_FIELDS) + [
                    "frame_width", "frame_height", "border_margin", "pitch", "yaw", "roll"
                ]
                if empty:
                    write_csv(run_dir / "faces.csv", [], fields)
                    save_npy(run_dir / "embeddings.npy", np.empty((0, 2), dtype=np.float32))
                else:
                    Image.new("RGB", (16, 16), (128, 128, 128)).save(run_dir / "face.png")
                    row = {
                        "face_id": "f", "source_frame": "face.png", "video_name": "v",
                        "frame_number": 0, "face_index": 0, "bbox_x1": 1, "bbox_y1": 1,
                        "bbox_x2": 15, "bbox_y2": 15, "face_width": 14, "face_height": 14,
                        "frame_width": 16, "frame_height": 16, "border_margin": 1,
                        "det_score": 0.99, "blur_score": 100, "aligned_path": "face.png",
                        "embedding_index": 0, "pitch": "", "yaw": "", "roll": "",
                    }
                    write_csv(run_dir / "faces.csv", [row])
                    save_npy(run_dir / "embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
                config = resolved_config(
                    ROOT / "configs" / "baseline_v1.json",
                    run_dir,
                    run_dir / "experiments",
                )
                result = run_experiment(config, ROOT)
                self.assertEqual(result["metadata"]["status"], "complete")
                self.assertEqual(result["metrics"]["counts"]["faces"], 0 if empty else 1)
                self.assertTrue((Path(result["run_dir"]) / "resolved_config.json").is_file())

    def test_comparison_reuses_cache_and_excludes_rejected_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "synthetic"
            generate_synthetic(run_dir, seed=42, overwrite=False)
            run_tracking(run_dir)
            run_clustering(run_dir)
            run_selection(run_dir)
            experiments_dir = run_dir / "experiments"
            baseline = resolved_config(
                ROOT / "configs" / "baseline_v1.json",
                run_dir,
                experiments_dir,
                run_dir / "ground_truth.csv",
            )
            hard = resolved_config(
                ROOT / "configs" / "hard_filter_v1_soft_ranking_v1.json",
                run_dir,
                experiments_dir,
                run_dir / "ground_truth.csv",
            )
            result = compare_experiments([baseline, hard], ROOT)
            absolute = result["absolute"]
            self.assertEqual(absolute["baseline_v1"]["counts"]["rejected"], 0)
            self.assertEqual(
                absolute["hard_filter_v1_soft_ranking_v1"]["counts"]["rejected"], 7
            )
            self.assertTrue(
                absolute["hard_filter_v1_soft_ranking_v1"]["feature_cache"]["hit"]
            )
            self.assertGreaterEqual(
                absolute["hard_filter_v1_soft_ranking_v1"]["classification"]["f1"], 0.98
            )
            hard_run = next(
                Path(row["run_dir"])
                for row in result["runs"]
                if row["experiment_name"] == "hard_filter_v1_soft_ranking_v1"
            )
            baseline_run = next(
                Path(row["run_dir"])
                for row in result["runs"]
                if row["experiment_name"] == "baseline_v1"
            )
            legacy_tracked = read_csv(run_dir / "tracking" / "faces_tracked.csv")
            baseline_tracked = read_csv(
                baseline_run / "pipeline" / "tracking" / "faces_tracked.csv"
            )
            self.assertEqual(
                [(row["face_id"], row["track_id"]) for row in baseline_tracked],
                [(row["face_id"], row["track_id"]) for row in legacy_tracked],
            )
            np.testing.assert_allclose(
                [float(row["quality_score"]) for row in baseline_tracked],
                [float(row["quality_score"]) for row in legacy_tracked],
                atol=1e-6,
                rtol=0,
            )
            self.assertEqual(
                [row["face_id"] for row in read_csv(baseline_run / "pipeline" / "selected" / "selection.csv")],
                [row["face_id"] for row in read_csv(run_dir / "selected" / "selection.csv")],
            )
            policy = {
                json.loads(line)["face_id"]: json.loads(line)
                for line in (hard_run / "policy_results.jsonl").read_text(encoding="utf-8").splitlines()
            }
            clustered = {
                row["face_id"]
                for row in read_csv(hard_run / "pipeline" / "clustering" / "faces_clustered.csv")
            }
            rejected = {face_id for face_id, row in policy.items() if not row["accepted"]}
            self.assertTrue(rejected)
            self.assertTrue(rejected.isdisjoint(clustered))
            self.assertTrue((Path(result["comparison_dir"]) / "comparison.json").is_file())
            self.assertIn(
                "counts.rejected",
                result["deltas_from_baseline"]["hard_filter_v1_soft_ranking_v1"],
            )


if __name__ == "__main__":
    unittest.main()
