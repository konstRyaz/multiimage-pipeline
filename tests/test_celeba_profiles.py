from __future__ import annotations

import json
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from face_pipeline.baseline_data import (
    BaselineDataError,
    CelebARecord,
    WiderImage,
    XqlfwPair,
)
from face_pipeline.baseline_workflow import (
    CELEBA_CACHE_SCHEMA,
    DiskCache,
    InsightFaceRuntime,
    _celeba_embeddings,
    evaluate,
    experiment_fingerprint,
    validate,
)
from face_pipeline.celeba_profiles import (
    MANIFEST_SCHEMA,
    PROFILE_LIMITS,
    SMOKE_WIDER_LIMITS,
    SMOKE_XQLFW_PAIRS_PER_CLASS_PER_FOLD,
    apply_profile,
    build_profile_selections,
    ensure_research_manifests,
    manifest_filename,
    records_from_manifest,
    select_wider_records,
    select_xqlfw_pairs,
    validate_manifest,
    xqlfw_expected_fold_size,
)
from src.baseline_evaluate import build_parser


def fixture_records() -> list[CelebARecord]:
    records = []
    for partition, count in (("train", 8), ("val", 6)):
        for identity in range(count):
            global_identity = identity + (100 if partition == "val" else 0)
            for image in range(3):
                records.append(
                    CelebARecord(
                        f"{partition}_{global_identity}_{image}.jpg",
                        global_identity,
                        partition,
                        (0.0, 0.0, 10.0, 10.0),
                    )
                )
    return records


LIMIT_FIXTURE = {
    "smoke": {"train": 4, "val": 2},
    "dev-a": {"train": 8, "val": 6},
    "dev-b": {"train": 8, "val": 6},
}


class CelebAProfileManifestTests(unittest.TestCase):
    def test_selection_is_deterministic_bounded_whole_and_disjoint(self) -> None:
        records = fixture_records()
        with mock.patch.dict(PROFILE_LIMITS, LIMIT_FIXTURE):
            first = build_profile_selections(records)
            second = build_profile_selections(list(reversed(records)))
        self.assertEqual(first, second)
        for profile in ("smoke", "dev-a", "dev-b"):
            for partition in ("train", "val"):
                selected = first[(profile, partition)]
                self.assertLessEqual(len(selected), LIMIT_FIXTURE[profile][partition])
                self.assertTrue(all(item.partition == partition for item in selected))
                selected_names = {item.filename for item in selected}
                for identity in {item.identity for item in selected}:
                    source_names = {
                        item.filename
                        for item in records
                        if item.partition == partition and item.identity == identity
                    }
                    self.assertEqual(selected_names & source_names, source_names)
        for partition in ("train", "val"):
            dev_a = {item.identity for item in first[("dev-a", partition)]}
            dev_b = {item.identity for item in first[("dev-b", partition)]}
            smoke = {item.identity for item in first[("smoke", partition)]}
            self.assertFalse(dev_a & dev_b)
            self.assertTrue(smoke <= dev_a)

    def test_manifests_are_named_hashed_reused_and_corruption_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            annotations.mkdir()
            for name in (
                "identity_CelebA.txt",
                "list_eval_partition.txt",
                "list_bbox_celeba.txt",
            ):
                (annotations / name).write_text(name + "\n", encoding="utf-8")
            with mock.patch.dict(PROFILE_LIMITS, LIMIT_FIXTURE):
                first = ensure_research_manifests(root / "run", fixture_records(), annotations)
                second = ensure_research_manifests(root / "run", fixture_records(), annotations)
                selections = build_profile_selections(fixture_records())
            self.assertEqual(first, second)
            for profile in ("smoke", "dev-a", "dev-b"):
                for partition in ("train", "val"):
                    path = root / "run" / "manifests" / manifest_filename(profile, partition)
                    self.assertTrue(path.is_file())
                    document = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(document["schema_version"], MANIFEST_SCHEMA)
                    validate_manifest(document)
                    self.assertEqual(
                        records_from_manifest(document, fixture_records()),
                        selections[(profile, partition)],
                    )
            broken = root / "run" / "manifests" / "celeba_smoke_train.json"
            document = json.loads(broken.read_text(encoding="utf-8"))
            document["records"][0]["identity"] = -1
            broken.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.dict(PROFILE_LIMITS, LIMIT_FIXTURE):
                with self.assertRaisesRegex(BaselineDataError, "повреждён|SHA-256"):
                    ensure_research_manifests(root / "run", fixture_records(), annotations)

    def test_changed_source_annotation_makes_existing_manifest_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            annotations.mkdir()
            names = (
                "identity_CelebA.txt",
                "list_eval_partition.txt",
                "list_bbox_celeba.txt",
            )
            for name in names:
                (annotations / name).write_text("v1\n", encoding="utf-8")
            with mock.patch.dict(PROFILE_LIMITS, LIMIT_FIXTURE):
                ensure_research_manifests(root / "run", fixture_records(), annotations)
                (annotations / names[0]).write_text("v2\n", encoding="utf-8")
                with self.assertRaisesRegex(BaselineDataError, "несовместим"):
                    ensure_research_manifests(root / "run", fixture_records(), annotations)


def write_small_celeba(root: Path) -> list[str]:
    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    rows = [
        ("a.jpg", 1, 0),
        ("b.jpg", 2, 0),
        ("c.jpg", 101, 1),
        ("d.jpg", 102, 1),
        ("test.jpg", 201, 2),
    ]
    (annotations / "identity_CelebA.txt").write_text(
        "\n".join(f"{name} {identity}" for name, identity, _ in rows) + "\n",
        encoding="utf-8",
    )
    (annotations / "list_eval_partition.txt").write_text(
        "\n".join(f"{name} {partition}" for name, _, partition in rows) + "\n",
        encoding="utf-8",
    )
    (annotations / "list_bbox_celeba.txt").write_text(
        "5\nimage_id x_1 y_1 width height\n"
        + "\n".join(f"{name} 0 0 10 10" for name, _, _ in rows)
        + "\n",
        encoding="utf-8",
    )
    for name, _, partition in rows:
        if partition != 2:
            (images / name).write_bytes(name.encode("ascii"))
    return [name for name, _, partition in rows if partition == 0]


def small_config(root: Path) -> dict[str, object]:
    return {
        "config_schema_version": "baseline_evaluation_config_v1",
        "experiment_name": "fixture",
        "seed": 42,
        "evaluation_profile": "smoke",
        "paths": {
            "data_root": str(root),
            "celeba_images": "images",
            "celeba_annotations": "annotations",
        },
        "model": {
            "name": "fixture",
            "provider": "cpu",
            "device": -1,
            "detection_size": 640,
            "minimum_detection_score": 0.01,
        },
        "processing": {"batch_size": 2, "limit": None},
        "wider": {},
        "xqlfw": {},
        "celeba": {
            "expected_records": 5,
            "main_face_min_iou": 0.3,
        },
    }


class FakeCelebARuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def infer_celeba(self, path: Path, *_args: object) -> dict[str, np.ndarray]:
        self.calls.append(path.name)
        selected = int(path.name == "a.jpg")
        return {
            "boxes": np.ones((2, 5), dtype=np.float32),
            "selected_box": np.ones((selected, 5), dtype=np.float32),
            "embeddings": np.ones((selected, 512), dtype=np.float32),
            "detector_candidates": np.asarray(2, dtype=np.int64),
            "main_face_selected": np.asarray(selected, dtype=np.int64),
        }


class CelebARecognitionAndCacheTests(unittest.TestCase):
    def test_celeba_computes_at_most_one_embedding_and_resumes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_names = write_small_celeba(root)
            config = small_config(root)
            runtime = FakeCelebARuntime()
            vectors, _, _, coverage, _ = _celeba_embeddings(
                config, root / "run", {"train"}, runtime, 0.5
            )
            self.assertEqual(set(runtime.calls), set(train_names))
            self.assertLessEqual(len(vectors), len(train_names))
            self.assertLessEqual(
                coverage["embeddings_computed"], coverage["main_faces_selected"]
            )
            self.assertLessEqual(
                coverage["main_faces_selected"], coverage["processed_images"]
            )
            self.assertEqual(coverage["detector_candidates"], 2 * len(train_names))
            self.assertEqual(coverage["cache_computed"], len(train_names))

            resumed = FakeCelebARuntime()
            _, _, _, second_coverage, _ = _celeba_embeddings(
                config, root / "run", {"train"}, resumed, 0.5
            )
            self.assertEqual(resumed.calls, [])
            self.assertEqual(second_coverage["cache_hits"], len(train_names))

    def test_old_celeba_cache_schema_is_not_reused_or_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_names = write_small_celeba(root)
            source = root / "images" / train_names[0]
            old = DiskCache(root / "run" / "cache" / "celeba", "old-schema")
            old.get_or_compute(
                source,
                lambda: {"boxes": np.ones((9, 5), dtype=np.float32)},
            )
            old_files = list((root / "run" / "cache" / "celeba").rglob("*.npz"))
            runtime = FakeCelebARuntime()
            _celeba_embeddings(small_config(root), root / "run", {"train"}, runtime, 0.5)
            self.assertTrue(runtime.calls)
            self.assertTrue(all(path.is_file() for path in old_files))
            self.assertTrue(
                (root / "run" / "cache" / "celeba" / CELEBA_CACHE_SCHEMA).is_dir()
            )

    def test_runtime_recognizes_only_selected_main_face(self) -> None:
        recognition_calls: list[object] = []

        class Face:
            def __init__(self, **values: object):
                self.__dict__.update(values)

        class Detector:
            def detect(self, _image: object, **_kwargs: object):
                return (
                    np.asarray(
                        [
                            [50, 50, 60, 60, 0.99],
                            [0, 0, 10, 10, 0.8],
                            [2, 2, 9, 9, 0.2],
                        ],
                        dtype=np.float32,
                    ),
                    np.ones((3, 5, 2), dtype=np.float32),
                )

        class Recognition:
            def get(self, _image: object, face: Face):
                recognition_calls.append(face)
                face.normed_embedding = np.ones(512, dtype=np.float32)

        app = types.SimpleNamespace(
            det_model=Detector(), models={"recognition": Recognition()}
        )
        cv2 = types.SimpleNamespace(imread=lambda _path: np.ones((10, 10, 3)))
        common = types.SimpleNamespace(Face=Face)
        insightface_app = types.SimpleNamespace(common=common)
        with mock.patch.dict(
            sys.modules,
            {
                "cv2": cv2,
                "insightface.app.common": common,
                "insightface.app": insightface_app,
            },
        ):
            runtime = InsightFaceRuntime(
                {
                    "name": "fixture",
                    "provider": "cpu",
                    "device": -1,
                    "detection_size": 640,
                    "minimum_detection_score": 0.01,
                }
            )
            with mock.patch.object(runtime, "_app", return_value=app):
                result = runtime.infer_celeba(Path("fixture.jpg"), (0, 0, 10, 10), 0.3, 0.5)
                missing = runtime.infer_celeba(
                    Path("fixture.jpg"), (100, 100, 110, 110), 0.3, 0.5
                )
        self.assertEqual(int(result["detector_candidates"]), 3)
        self.assertEqual(len(result["embeddings"]), 1)
        self.assertEqual(len(recognition_calls), 1)
        self.assertEqual(len(missing["embeddings"]), 0)


class ProfileInterfaceTests(unittest.TestCase):
    def test_smoke_bounds_wider_and_xqlfw_deterministically(self) -> None:
        config = {"evaluation_profile": "smoke"}
        wider = [WiderImage(f"event/{index}.jpg", ()) for index in range(200)]
        first_wider = select_wider_records(config, wider, "train")
        second_wider = select_wider_records(config, list(reversed(wider)), "train")
        self.assertEqual(first_wider, second_wider)
        self.assertEqual(len(first_wider), SMOKE_WIDER_LIMITS["train"])

        pairs = []
        for fold in range(10):
            for same in (True, False):
                for index in range(20):
                    pairs.append(
                        XqlfwPair(
                            Path(f"{fold}/{int(same)}/{index}_left.jpg"),
                            Path(f"{fold}/{int(same)}/{index}_right.jpg"),
                            same,
                            fold,
                        )
                    )
        first_pairs = select_xqlfw_pairs(config, pairs)
        second_pairs = select_xqlfw_pairs(config, list(reversed(pairs)))
        self.assertEqual(first_pairs, second_pairs)
        self.assertEqual(
            len(first_pairs), 20 * SMOKE_XQLFW_PAIRS_PER_CLASS_PER_FOLD
        )
        for fold in range(10):
            fold_pairs = [pair for pair in first_pairs if pair.fold == fold]
            self.assertEqual(sum(pair.same for pair in fold_pairs), 5)
            self.assertEqual(sum(not pair.same for pair in fold_pairs), 5)
        self.assertEqual(xqlfw_expected_fold_size(config), 10)

    def test_non_smoke_profiles_keep_complete_wider_and_xqlfw(self) -> None:
        wider = [WiderImage(f"{index}.jpg", ()) for index in range(100)]
        pairs = [XqlfwPair(Path("left"), Path("right"), True, 0)]
        for profile in ("dev-a", "dev-b", "full"):
            config = {} if profile == "full" else {"evaluation_profile": profile}
            self.assertEqual(select_wider_records(config, wider, "train"), wider)
            self.assertEqual(select_xqlfw_pairs(config, pairs), pairs)
            self.assertEqual(xqlfw_expected_fold_size(config), 600)

    def test_cli_parses_profiles_and_full_keeps_old_fingerprint(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare",
                "--config",
                "config.json",
                "--run-dir",
                "run",
                "--profile",
                "dev-a",
            ]
        )
        self.assertEqual(args.profile, "dev-a")
        config = small_config(Path("/tmp/fixture"))
        config.pop("evaluation_profile")
        before = experiment_fingerprint(config)
        apply_profile(config, "full")
        self.assertNotIn("evaluation_profile", config)
        self.assertEqual(before, experiment_fingerprint(config))

    def test_smoke_has_a_separate_cache_fingerprint_schema(self) -> None:
        config = small_config(Path("/tmp/fixture"))
        smoke_fingerprint = experiment_fingerprint(config)
        config["evaluation_profile"] = "dev-a"
        self.assertNotEqual(smoke_fingerprint, experiment_fingerprint(config))

    def test_research_evaluate_fails_before_test_or_frozen_access(self) -> None:
        config = small_config(Path("/path/that/must/not/be/read"))
        with self.assertRaisesRegex(BaselineDataError, "запрещает evaluate"):
            evaluate(config, Path("/also/not/read"), Path("/missing/frozen.json"))

    def test_research_validate_routes_complete_wider_val_protocol(self) -> None:
        source = inspect.getsource(validate)
        self.assertIn("if is_research_profile(config):", source)
        self.assertIn("_wider_validation_metrics(", source)


if __name__ == "__main__":
    unittest.main()
