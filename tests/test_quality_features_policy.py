from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from face_pipeline.experiment_config import DEFAULT_CONFIG, validate_experiment_config
from face_pipeline.io import save_npy, write_csv
from face_pipeline.policy import apply_policy, soft_ratings
from face_pipeline.quality import quality_scores
from face_pipeline.quality_features import (
    extract_quality_features,
    image_features,
    load_or_build_feature_cache,
)


def base_feature(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "det_score": 0.9,
        "min_face_size": 100.0,
        "blur_score": 100.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "pose_available": True,
        "border_margin_relative": 0.2,
        "brightness_mean": 128.0,
        "dark_fraction": 0.0,
        "bright_fraction": 0.0,
        "image_available": True,
    }
    value.update(changes)
    return value


def policy_with_modes(**modes: str) -> dict[str, object]:
    rules = {}
    definitions = {
        "low_detection_score": ("det_score", "lt", 0.65),
        "small_face": ("min_face_size", "lt", 32.0),
        "large_yaw": ("yaw", "abs_gt", 55.0),
        "pose_unavailable": ("pose_available", "eq", False),
    }
    for name, (feature, operator, threshold) in definitions.items():
        rules[name] = {
            "feature": feature,
            "operator": operator,
            "threshold": threshold,
            "mode": modes.get(name, "off"),
            "missing": "ignore",
        }
    return {"version": "test_v1", "rules": rules}


class QualityFeaturePolicyTests(unittest.TestCase):
    def test_baseline_rating_matches_original_formula(self) -> None:
        rows = [
            {
                "det_score": "0.82", "face_width": "70", "face_height": "60",
                "blur_score": "40", "yaw": "20", "pitch": "", "roll": "-5",
            },
            {
                "det_score": "0.97", "face_width": "180", "face_height": "150",
                "blur_score": "160", "yaw": "", "pitch": "", "roll": "",
            },
        ]
        features = [
            base_feature(
                det_score=float(row["det_score"]),
                min_face_size=min(float(row["face_width"]), float(row["face_height"])),
                blur_score=float(row["blur_score"]),
                yaw=float(row["yaw"]) if row["yaw"] else None,
                pitch=float(row["pitch"]) if row["pitch"] else None,
                roll=float(row["roll"]) if row["roll"] else None,
            )
            for row in rows
        ]
        calculated, _ = soft_ratings(features, DEFAULT_CONFIG["ranking"])
        expected = quality_scores(rows)
        np.testing.assert_allclose(
            [row["soft_rating"] for row in calculated],
            expected,
            rtol=0,
            atol=2e-7,
        )

    def test_off_shadow_and_hard_modes(self) -> None:
        feature = base_feature(det_score=0.2)
        off = apply_policy(feature, policy_with_modes(low_detection_score="off"))
        shadow = apply_policy(feature, policy_with_modes(low_detection_score="shadow"))
        hard = apply_policy(feature, policy_with_modes(low_detection_score="hard"))
        self.assertTrue(off.accepted)
        self.assertEqual(off.shadow_reasons, ())
        self.assertTrue(shadow.accepted)
        self.assertEqual(shadow.shadow_reasons, ("low_detection_score",))
        self.assertFalse(hard.accepted)
        self.assertEqual(hard.reject_reasons, ("low_detection_score",))

    def test_multiple_reasons_and_threshold_boundaries(self) -> None:
        decision = apply_policy(
            base_feature(det_score=0.2, min_face_size=10.0),
            policy_with_modes(low_detection_score="hard", small_face="hard"),
        )
        self.assertEqual(
            decision.reject_reasons,
            ("low_detection_score", "small_face"),
        )
        boundary = apply_policy(
            base_feature(det_score=0.65, min_face_size=32.0, yaw=55.0),
            policy_with_modes(
                low_detection_score="hard", small_face="hard", large_yaw="hard"
            ),
        )
        self.assertTrue(boundary.accepted)

    def test_missing_pose_is_explicit_and_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            image = Image.new("RGB", (16, 16), (128, 128, 128))
            image.save(run_dir / "face.png")
            row = {
                "face_id": "f", "embedding_index": "0", "aligned_path": "face.png",
                "face_width": "40", "face_height": "50", "frame_width": "100",
                "frame_height": "100", "border_margin": "4", "det_score": "0.9",
                "blur_score": "10", "yaw": "", "pitch": "", "roll": "",
            }
            features = extract_quality_features(row, run_dir)
            self.assertFalse(features["pose_available"])
            self.assertIsNone(features["yaw"])
            decision = apply_policy(
                features,
                policy_with_modes(pose_unavailable="shadow"),
            )
            self.assertEqual(decision.shadow_reasons, ("pose_unavailable",))

    def test_illumination_and_sharpness_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dark = root / "dark.png"
            checker = root / "checker.png"
            Image.new("L", (20, 20), 5).save(dark)
            pattern = np.indices((20, 20)).sum(axis=0) % 2 * 255
            Image.fromarray(pattern.astype(np.uint8), mode="L").save(checker)
            dark_features = image_features(dark)
            checker_features = image_features(checker)
            self.assertLess(dark_features["brightness_mean"], 10)
            self.assertEqual(dark_features["dark_fraction"], 1.0)
            self.assertGreater(checker_features["derived_sharpness"], dark_features["derived_sharpness"])

    def test_soft_rating_known_formula(self) -> None:
        features = [base_feature(det_score=0.75, min_face_size=90.0, blur_score=100.0)]
        result, context = soft_ratings(features, DEFAULT_CONFIG["ranking"])
        self.assertEqual(context["blur_scale"], 100.0)
        expected = 0.30 * 0.5 + 0.25 * 0.5 + 0.25 * 0.5 + 0.20 * 1.0
        self.assertAlmostEqual(result[0]["soft_rating"], expected)

    def test_invalid_config_reports_unknown_rule_and_negative_size(self) -> None:
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        config["policy"]["rules"]["mystery"] = {
            "feature": "x", "operator": "lt", "threshold": 1, "mode": "hard"
        }
        with self.assertRaisesRegex(ValueError, "Неизвестные правила"):
            validate_experiment_config(config)
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        config["policy"]["rules"]["small_face"]["threshold"] = -1
        with self.assertRaisesRegex(ValueError, "не может быть отрицательным"):
            validate_experiment_config(config)

    def test_cache_hit_corruption_and_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            Image.new("RGB", (16, 16), (100, 100, 100)).save(run_dir / "face.png")
            row = {
                "face_id": "f", "source_frame": "face.png", "video_name": "v",
                "frame_number": 0, "face_index": 0,
                "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10,
                "face_width": 10, "face_height": 10, "frame_width": 16,
                "frame_height": 16, "border_margin": 0, "det_score": 0.9,
                "pitch": "", "yaw": "", "roll": "", "blur_score": 50,
                "aligned_path": "face.png", "embedding_index": 0,
            }
            write_csv(run_dir / "faces.csv", [row])
            save_npy(run_dir / "embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
            _, first = load_or_build_feature_cache(run_dir)
            _, second = load_or_build_feature_cache(run_dir)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])

            cache_file = run_dir / "derived" / "quality_features_v1" / "features.jsonl"
            cache_file.write_text("{broken", encoding="utf-8")
            _, repaired = load_or_build_feature_cache(run_dir)
            self.assertFalse(repaired["cache_hit"])
            self.assertIn("Повреждённый", repaired["warning"])

            row["det_score"] = 0.8
            write_csv(run_dir / "faces.csv", [row])
            _, incompatible = load_or_build_feature_cache(run_dir)
            self.assertFalse(incompatible["cache_hit"])
            self.assertIn("Несовместимый", incompatible["warning"])


if __name__ == "__main__":
    unittest.main()
