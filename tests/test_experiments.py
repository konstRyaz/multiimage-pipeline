from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
import select_faces as select_faces_cli
import track_faces as track_faces_cli


def resolved_config(path: Path, run_dir: Path, experiments_dir: Path, labels: Path | None = None):
    return resolve_paths(
        load_experiment_config(path),
        path,
        run_dir,
        experiments_dir,
        labels,
    )


def _manual_run(root: Path, rows: list[dict[str, object]]) -> Path:
    run_dir = root / "manual"
    run_dir.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    for name in {"source_good.png", "aligned_good.png", "source_bad.png", "aligned_bad.png"}:
        image.save(run_dir / name)
    write_csv(run_dir / "faces.csv", rows)
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]][: len(rows)], dtype=np.float32)
    if len(rows) == 1:
        embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    save_npy(run_dir / "embeddings.npy", embeddings)
    return run_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            self.assertEqual(
                absolute["hard_filter_v1_soft_ranking_v1"]["classification"]["f1"], 1.0
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

    def test_rejected_face_is_excluded_from_ranking_and_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {
                    "face_id": "good",
                    "source_frame": "source_good.png",
                    "video_name": "v",
                    "frame_number": 1,
                    "face_index": 0,
                    "bbox_x1": 8,
                    "bbox_y1": 8,
                    "bbox_x2": 48,
                    "bbox_y2": 48,
                    "face_width": 40,
                    "face_height": 40,
                    "frame_width": 64,
                    "frame_height": 64,
                    "border_margin": 8,
                    "det_score": 0.99,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "blur_score": 40.0,
                    "aligned_path": "aligned_good.png",
                    "embedding_index": 0,
                    "cluster_id": "",
                    "rejection_reason": "",
                },
                {
                    "face_id": "bad",
                    "source_frame": "source_bad.png",
                    "video_name": "v",
                    "frame_number": 2,
                    "face_index": 1,
                    "bbox_x1": 8,
                    "bbox_y1": 8,
                    "bbox_x2": 48,
                    "bbox_y2": 48,
                    "face_width": 40,
                    "face_height": 40,
                    "frame_width": 64,
                    "frame_height": 64,
                    "border_margin": 8,
                    "det_score": 0.10,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "blur_score": 1000.0,
                    "aligned_path": "aligned_bad.png",
                    "embedding_index": 1,
                    "cluster_id": "",
                    "rejection_reason": "",
                },
            ]
            run_dir = _manual_run(root, rows)
            config = resolved_config(
                ROOT / "configs" / "hard_filter_v1_soft_ranking_v1.json",
                run_dir,
                run_dir / "experiments",
            )
            result = run_experiment(config, ROOT)
            experiment_dir = Path(result["run_dir"])
            ranking_rows = _read_jsonl(experiment_dir / "ranking_results.jsonl")
            self.assertEqual(len(ranking_rows), 1)
            self.assertEqual(ranking_rows[0]["face_id"], "good")
            self.assertEqual(
                [row["face_id"] for row in read_csv(experiment_dir / "pipeline" / "tracking" / "faces_tracked.csv")],
                ["good"],
            )
            self.assertEqual(
                [row["face_id"] for row in read_csv(experiment_dir / "pipeline" / "clustering" / "faces_clustered.csv")],
                ["good"],
            )
            self.assertNotIn(
                "bad",
                [row["face_id"] for row in read_csv(experiment_dir / "pipeline" / "selected" / "selection.csv")],
            )

    def test_rejected_face_does_not_change_ranking_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_rows = [
                {
                    "face_id": "good",
                    "source_frame": "source_good.png",
                    "video_name": "v",
                    "frame_number": 1,
                    "face_index": 0,
                    "bbox_x1": 8,
                    "bbox_y1": 8,
                    "bbox_x2": 48,
                    "bbox_y2": 48,
                    "face_width": 40,
                    "face_height": 40,
                    "frame_width": 64,
                    "frame_height": 64,
                    "border_margin": 8,
                    "det_score": 0.99,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "blur_score": 40.0,
                    "aligned_path": "aligned_good.png",
                    "embedding_index": 0,
                    "cluster_id": "",
                    "rejection_reason": "",
                }
            ]
            combined_rows = accepted_rows + [
                {
                    "face_id": "bad",
                    "source_frame": "source_bad.png",
                    "video_name": "v",
                    "frame_number": 2,
                    "face_index": 1,
                    "bbox_x1": 8,
                    "bbox_y1": 8,
                    "bbox_x2": 48,
                    "bbox_y2": 48,
                    "face_width": 40,
                    "face_height": 40,
                    "frame_width": 64,
                    "frame_height": 64,
                    "border_margin": 8,
                    "det_score": 0.10,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "blur_score": 1000.0,
                    "aligned_path": "aligned_bad.png",
                    "embedding_index": 1,
                    "cluster_id": "",
                    "rejection_reason": "",
                },
            ]
            single_dir = _manual_run(root / "single", accepted_rows)
            combined_dir = _manual_run(root / "combined", combined_rows)
            config_single = resolved_config(
                ROOT / "configs" / "hard_filter_v1_soft_ranking_v1.json",
                single_dir,
                single_dir / "experiments",
            )
            config_combined = resolved_config(
                ROOT / "configs" / "hard_filter_v1_soft_ranking_v1.json",
                combined_dir,
                combined_dir / "experiments",
            )
            single = run_experiment(config_single, ROOT)
            combined = run_experiment(config_combined, ROOT)
            single_ranking = _read_jsonl(Path(single["run_dir"]) / "ranking_results.jsonl")
            combined_ranking = _read_jsonl(Path(combined["run_dir"]) / "ranking_results.jsonl")
            self.assertEqual(len(single_ranking), 1)
            self.assertEqual(len(combined_ranking), 1)
            self.assertEqual(single_ranking[0]["face_id"], "good")
            self.assertEqual(combined_ranking[0]["face_id"], "good")
            self.assertEqual(single_ranking[0]["soft_rating"], combined_ranking[0]["soft_rating"])
            self.assertEqual(
                single_ranking[0]["detection_component"],
                combined_ranking[0]["detection_component"],
            )
            self.assertEqual(
                single_ranking[0]["size_component"],
                combined_ranking[0]["size_component"],
            )
            self.assertEqual(
                single_ranking[0]["sharpness_component"],
                combined_ranking[0]["sharpness_component"],
            )
            self.assertEqual(
                single_ranking[0]["pose_component"],
                combined_ranking[0]["pose_component"],
            )

    def test_empty_after_policy_filter_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {
                    "face_id": "bad",
                    "source_frame": "source_bad.png",
                    "video_name": "v",
                    "frame_number": 1,
                    "face_index": 0,
                    "bbox_x1": 8,
                    "bbox_y1": 8,
                    "bbox_x2": 48,
                    "bbox_y2": 48,
                    "face_width": 40,
                    "face_height": 40,
                    "frame_width": 64,
                    "frame_height": 64,
                    "border_margin": 8,
                    "det_score": 0.10,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "blur_score": 1000.0,
                    "aligned_path": "aligned_bad.png",
                    "embedding_index": 0,
                    "cluster_id": "",
                    "rejection_reason": "",
                }
            ]
            run_dir = _manual_run(root, rows)
            config = resolved_config(
                ROOT / "configs" / "hard_filter_v1_soft_ranking_v1.json",
                run_dir,
                run_dir / "experiments",
            )
            result = run_experiment(config, ROOT)
            experiment_dir = Path(result["run_dir"])
            self.assertEqual(result["metadata"]["status"], "complete")
            self.assertEqual(result["metrics"]["counts"]["accepted"], 0)
            self.assertEqual(result["metrics"]["counts"]["selected"], 0)
            self.assertEqual(
                _read_jsonl(experiment_dir / "ranking_results.jsonl"),
                [],
            )
            self.assertEqual(
                read_csv(experiment_dir / "pipeline" / "selected" / "selection.csv"),
                [],
            )

    def test_synthetic_ground_truth_marks_quality_and_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "synthetic"
            generate_synthetic(run_dir, seed=42, overwrite=False)
            truth_rows = read_csv(run_dir / "ground_truth.csv")
            face_rows = read_csv(run_dir / "faces.csv")
            truth_by_face = {row["face_id"]: row for row in truth_rows}
            face_by_face = {row["face_id"]: row for row in face_rows}
            self.assertIn("is_acceptable", truth_rows[0])
            self.assertTrue(all("is_acceptable" in row for row in truth_rows))
            self.assertTrue(all(
                int(truth_by_face[face_id]["is_acceptable"]) == int(
                    row["true_person"] != "noise" and float(face_by_face[face_id]["blur_score"]) >= 20.0
                )
                for face_id, row in truth_by_face.items()
            ))
            noise_faces = [
                row["face_id"] for row in truth_rows if row["true_person"] == "noise"
            ]
            self.assertTrue(noise_faces)
            self.assertTrue(all(int(truth_by_face[face_id]["is_acceptable"]) == 0 for face_id in noise_faces))
            blurred_faces = [
                row["face_id"]
                for row in face_rows
                if float(row["blur_score"]) < 20.0
            ]
            self.assertTrue(blurred_faces)
            self.assertTrue(all(int(truth_by_face[face_id]["is_acceptable"]) == 0 for face_id in blurred_faces))
            real_faces = [
                row["face_id"]
                for row in truth_rows
                if row["true_person"] != "noise" and float(face_by_face[row["face_id"]]["blur_score"]) >= 20.0
            ]
            self.assertTrue(real_faces)
            self.assertTrue(all(int(truth_by_face[face_id]["is_acceptable"]) == 1 for face_id in real_faces))

    def test_cli_commands_forward_new_arguments_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            with patch.object(select_faces_cli, "run_selection", return_value={"ok": True}) as mocked:
                with patch.object(sys, "argv", ["select_faces.py", "--run-dir", str(run_dir)]):
                    select_faces_cli.main()
                self.assertEqual(mocked.call_args.kwargs["hash_size"], 8)
                with patch.object(sys, "argv", ["select_faces.py", "--run-dir", str(run_dir), "--hash-size", "12"]):
                    select_faces_cli.main()
                self.assertEqual(mocked.call_args.kwargs["hash_size"], 12)

            with patch.object(track_faces_cli, "run_tracking", return_value={"ok": True}) as mocked:
                with patch.object(sys, "argv", ["track_faces.py", "--run-dir", str(run_dir)]):
                    track_faces_cli.main()
                self.assertEqual(mocked.call_args.kwargs["quality_weight_offset"], 0.5)
                with patch.object(sys, "argv", ["track_faces.py", "--run-dir", str(run_dir), "--quality-weight-offset", "0.75"]):
                    track_faces_cli.main()
                self.assertEqual(mocked.call_args.kwargs["quality_weight_offset"], 0.75)


if __name__ == "__main__":
    unittest.main()
