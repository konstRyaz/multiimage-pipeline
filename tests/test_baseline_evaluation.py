from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from face_pipeline.baseline_data import (
    BaselineDataError,
    CelebARecord,
    load_celeba_records,
    parse_wider_annotations,
    parse_xqlfw_pairs,
    validate_identity_disjoint,
)
from face_pipeline.baseline_metrics import (
    match_main_face,
    metrics_all_and_multi,
    scalable_threshold_clusters,
    xqlfw_cross_validation,
)
from face_pipeline.baseline_workflow import (
    CONFIG_SCHEMA,
    FROZEN_SCHEMA,
    DiskCache,
    ProgressReporter,
    evaluate,
    experiment_fingerprint,
    validate_frozen,
)
from face_pipeline.baseline_wider import detector_curve, wider_compatible_ap


def minimal_config(root: Path) -> dict[str, object]:
    return {
        "config_schema_version": CONFIG_SCHEMA,
        "experiment_name": "fixture",
        "seed": 42,
        "paths": {"data_root": str(root)},
        "model": {"name": "fixture", "provider": "cpu", "device": -1, "detection_size": 640, "minimum_detection_score": 0.01},
        "processing": {"batch_size": 2, "limit": None},
        "wider": {"iou_threshold": 0.5},
        "xqlfw": {},
        "celeba": {"main_face_min_iou": 0.3},
    }


class WiderParserTests(unittest.TestCase):
    def test_parser_reads_images_and_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text("0--Parade/a.jpg\n2\n1 2 3 4 0 0 0 0 0 0\n5 6 0 7 0 0 0 0 0 0\nb.jpg\n0\n", encoding="utf-8")
            records = parse_wider_annotations(path)
            self.assertEqual([item.relative_path for item in records], ["0--Parade/a.jpg", "b.jpg"])
            self.assertEqual(records[0].boxes, ((1.0, 2.0, 4.0, 6.0),))

    def test_parser_skips_official_placeholder_after_zero_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text(
                "zero.jpg\n"
                "0\n"
                "0 0 0 0 0 0 0 0 0 0\n"
                "next.jpg\n"
                "1\n"
                "1 2 3 4 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            records = parse_wider_annotations(path)
            self.assertEqual(
                [item.relative_path for item in records],
                ["zero.jpg", "next.jpg"],
            )
            self.assertEqual(records[0].boxes, ())
            self.assertEqual(
                records[1].boxes,
                ((1.0, 2.0, 4.0, 6.0),),
            )

    def test_detector_curve_and_compatible_ap_on_perfect_fixture(self) -> None:
        predictions = [np.asarray([[20, 20, 30, 30, 0.1], [0, 0, 9, 9, 0.9]], dtype=float)]
        actual = [np.asarray([[0, 0, 9, 9]], dtype=float)]
        curve = detector_curve(predictions, actual, [0.5])
        self.assertEqual((curve[0]["tp"], curve[0]["fp"], curve[0]["fn"]), (1, 0, 0))
        ap = wider_compatible_ap(["event/a.jpg"], predictions, actual, {"event/a.jpg": {0}})
        self.assertAlmostEqual(ap, 1.0)


class XqlfwProtocolTests(unittest.TestCase):
    @staticmethod
    def write_pairs(path: Path, count: int = 6000) -> None:
        lines = ["10 300"]
        for index in range(count):
            within = index % 600
            if within < 300:
                lines.append(f"person_{index:04d} 1 2")
            else:
                lines.append(f"left_{index:04d} 1 right_{index:04d} 1")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_rejects_incomplete_file_even_with_valid_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.txt"
            self.write_pairs(path, 5999)
            with self.assertRaisesRegex(BaselineDataError, "6000"):
                parse_xqlfw_pairs(path, Path(directory), require_images=False)

    def test_splits_ten_blocks_of_600(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.txt"
            self.write_pairs(path)
            pairs = parse_xqlfw_pairs(path, Path(directory), require_images=False)
            self.assertEqual([sum(pair.fold == fold for pair in pairs) for fold in range(10)], [600] * 10)
            self.assertEqual([sum(pair.fold == fold and pair.same for pair in pairs) for fold in range(10)], [300] * 10)

    def test_threshold_for_fold_uses_only_other_nine_blocks(self) -> None:
        labels = np.tile(np.r_[np.ones(300, dtype=bool), np.zeros(300, dtype=bool)], 10)
        folds = np.repeat(np.arange(10), 600)
        scores = np.where(labels, 0.8, 0.2).astype(float)
        first = xqlfw_cross_validation(scores, labels, folds)
        changed = scores.copy()
        changed[folds == 0] = np.linspace(-100, 100, 600)
        second = xqlfw_cross_validation(changed, labels, folds)
        self.assertEqual(first["folds"][0]["threshold"], second["folds"][0]["threshold"])


class CelebATests(unittest.TestCase):
    def test_reads_identity_partition_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "identity_CelebA.txt").write_text("a.jpg 1\nb.jpg 2\nc.jpg 3\n", encoding="utf-8")
            (root / "list_eval_partition.txt").write_text("a.jpg 0\nb.jpg 1\nc.jpg 2\n", encoding="utf-8")
            (root / "list_bbox_celeba.txt").write_text("3\nimage_id x_1 y_1 width height\na.jpg 1 2 3 4\nb.jpg 2 3 4 5\nc.jpg 3 4 5 6\n", encoding="utf-8")
            records = load_celeba_records(root, expected=3)
            self.assertEqual([(item.filename, item.identity, item.partition) for item in records], [("a.jpg", 1, "train"), ("b.jpg", 2, "val"), ("c.jpg", 3, "test")])
            self.assertEqual(records[0].bbox, (1.0, 2.0, 4.0, 6.0))

    def test_preserves_zero_sized_official_bbox_as_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "identity_CelebA.txt").write_text(
                "zero.jpg 1\n",
                encoding="utf-8",
            )
            (root / "list_eval_partition.txt").write_text(
                "zero.jpg 0\n",
                encoding="utf-8",
            )
            (root / "list_bbox_celeba.txt").write_text(
                "1\n"
                "image_id x_1 y_1 width height\n"
                "zero.jpg 320 828 0 0\n",
                encoding="utf-8",
            )

            records = load_celeba_records(root, expected=1)

            self.assertEqual(len(records), 1)
            self.assertIsNone(records[0].bbox)

    def test_rejects_partially_degenerate_official_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "identity_CelebA.txt").write_text(
                "bad.jpg 1\n",
                encoding="utf-8",
            )
            (root / "list_eval_partition.txt").write_text(
                "bad.jpg 0\n",
                encoding="utf-8",
            )
            (root / "list_bbox_celeba.txt").write_text(
                "1\n"
                "image_id x_1 y_1 width height\n"
                "bad.jpg 10 20 0 30\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                BaselineDataError,
                "неположительная рамка",
            ):
                load_celeba_records(root, expected=1)

    def test_rejects_identity_overlap(self) -> None:
        records = [CelebARecord("a", 1, "train", (0, 0, 1, 1)), CelebARecord("b", 1, "test", (0, 0, 1, 1))]
        with self.assertRaisesRegex(BaselineDataError, "пересекаются"):
            validate_identity_disjoint(records)

    def test_matches_main_face_by_maximum_iou(self) -> None:
        detections = [(50, 50, 90, 90, 0.99), (1, 1, 11, 11, 0.8)]
        self.assertEqual(match_main_face((0, 0, 10, 10), detections, 0.3), 1)
        self.assertIsNone(match_main_face((0, 0, 10, 10), [(50, 50, 60, 60, 1)], 0.3))

    def test_clustering_metrics_known_example_and_singletons(self) -> None:
        result = metrics_all_and_multi(["a", "a", "b", "c"], [0, 0, 0, 1])
        self.assertAlmostEqual(result["all_identities"]["pair_recall"], 1.0)
        self.assertAlmostEqual(result["all_identities"]["pair_precision"], 1 / 3)
        self.assertEqual(result["all_identities"]["fragmented_identity_fraction"], 0.0)
        self.assertEqual(result["identities_with_at_least_two_images"]["pair_f1"], 1.0)

    def test_scalable_clustering_never_materializes_n_by_n_fixture(self) -> None:
        vectors = np.eye(7, dtype=np.float32)
        shapes = []
        original = np.where
        def checked(value: np.ndarray):
            shapes.append(value.shape)
            return original(value)
        with mock.patch("face_pipeline.baseline_metrics.np.where", side_effect=checked):
            scalable_threshold_clusters(vectors, 0.9, block_size=2)
        self.assertTrue(shapes)
        self.assertTrue(all(shape != (7, 7) for shape in shapes))


class FreezeAndCacheTests(unittest.TestCase):
    def test_frozen_parameters_validate_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = minimal_config(root)
            frozen = {"frozen_schema_version": FROZEN_SCHEMA, "experiment_fingerprint": experiment_fingerprint(config),
                      "wider": {"confidence_threshold": 0.5, "iou_threshold": 0.5},
                      "xqlfw": {"verification_threshold": 0.4},
                      "celeba": {"clustering_threshold": 0.45, "main_face_min_iou": 0.3}}
            path = root / "frozen.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            validate_frozen(json.loads(path.read_text(encoding="utf-8")), config)

    def test_evaluate_refuses_to_run_without_frozen_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            (run_dir / "stages").mkdir(parents=True)
            (run_dir / "stages" / "prepare.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            with self.assertRaisesRegex(BaselineDataError, "не найден"):
                evaluate(minimal_config(root), run_dir, root / "missing.json")

    def test_evaluate_source_contains_no_calibration_calls(self) -> None:
        source = inspect.getsource(evaluate)
        self.assertNotIn("select_detector_threshold(", source)
        self.assertNotIn("xqlfw_cross_validation(", source)
        self.assertNotIn("_select_celeba(", source)

    def test_cache_resumes_without_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "image.jpg"
            source.write_bytes(b"fixture")
            cache = DiskCache(root / "cache", "model-v1")
            calls = []
            first, first_hit = cache.get_or_compute(source, lambda: calls.append(1) or {"boxes": np.ones((1, 5), dtype=np.float32)})
            second, second_hit = cache.get_or_compute(source, lambda: calls.append(2) or {"boxes": np.zeros((1, 5), dtype=np.float32)})
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(calls, [1])
            np.testing.assert_array_equal(first["boxes"], second["boxes"])


class ProgressReporterTests(unittest.TestCase):
    def test_writes_atomic_snapshot_history_and_intermediate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = ProgressReporter(root, "calibrate", interval_seconds=1)
            reporter.begin_phase("celeba_train", 10)
            reporter.advance(4, {"items": 10, "cache_hits": 1, "computed": 3, "errors": 0},
                             {"текущее_покрытие": 0.75}, force=True)
            reporter.metric("fixture", {"f1": 0.8})
            current = json.loads((root / "progress" / "current.json").read_text(encoding="utf-8"))
            self.assertEqual(current["stage"], "calibrate")
            self.assertEqual(current["phase"], "celeba_train")
            self.assertEqual(current["processed"], 4)
            self.assertAlmostEqual(current["percent"], 40.0)
            self.assertIn("eta_seconds", current)
            self.assertEqual(current["technical_metrics"]["текущее_покрытие"], 0.75)
            metric = json.loads((root / "progress" / "intermediate_metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(metric["component"], "fixture")
            self.assertEqual(metric["metrics"]["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
