from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .policy import validate_policy, validate_ranking
from .quality_features import FEATURE_SCHEMA_VERSION


CONFIG_SCHEMA_VERSION = "experiment_config_v1"


DEFAULT_CONFIG: dict[str, Any] = {
    "config_schema_version": CONFIG_SCHEMA_VERSION,
    "experiment_name": "baseline_v1",
    "description": "Функционально совместимая фиксация исходного поведения.",
    "random_seed": 42,
    "feature_extraction": {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dark_pixel_threshold": 30,
        "bright_pixel_threshold": 225,
        "batch_size": 256,
    },
    "policy": {
        "version": "baseline_v1",
        "rules": {
            "low_detection_score": {
                "feature": "det_score", "operator": "lt", "threshold": 0.65,
                "mode": "shadow", "missing": "violate",
            },
            "small_face": {
                "feature": "min_face_size", "operator": "lt", "threshold": 32.0,
                "mode": "shadow", "missing": "violate",
            },
            "blurred": {
                "feature": "blur_score", "operator": "lt", "threshold": 20.0,
                "mode": "shadow", "missing": "violate",
            },
            "large_yaw": {
                "feature": "yaw", "operator": "abs_gt", "threshold": 55.0,
                "mode": "shadow", "missing": "ignore",
            },
            "large_pitch": {
                "feature": "pitch", "operator": "abs_gt", "threshold": 45.0,
                "mode": "shadow", "missing": "ignore",
            },
            "large_roll": {
                "feature": "roll", "operator": "abs_gt", "threshold": 45.0,
                "mode": "shadow", "missing": "ignore",
            },
            "pose_unavailable": {
                "feature": "pose_available", "operator": "eq", "threshold": False,
                "mode": "off", "missing": "violate",
            },
            "face_near_border": {
                "feature": "border_margin_relative", "operator": "lt", "threshold": 0.01,
                "mode": "off", "missing": "ignore",
            },
            "too_dark": {
                "feature": "brightness_mean", "operator": "lt", "threshold": 40.0,
                "mode": "off", "missing": "ignore",
            },
            "too_bright": {
                "feature": "brightness_mean", "operator": "gt", "threshold": 215.0,
                "mode": "off", "missing": "ignore",
            },
            "extreme_dark_pixels": {
                "feature": "dark_fraction", "operator": "gt", "threshold": 0.60,
                "mode": "off", "missing": "ignore",
            },
            "extreme_bright_pixels": {
                "feature": "bright_fraction", "operator": "gt", "threshold": 0.60,
                "mode": "off", "missing": "ignore",
            },
            "image_unavailable": {
                "feature": "image_available", "operator": "eq", "threshold": False,
                "mode": "off", "missing": "violate",
            },
        },
    },
    "ranking": {
        "version": "baseline_quality_v1",
        "weights": {
            "detection_component": 0.30,
            "size_component": 0.25,
            "sharpness_component": 0.25,
            "pose_component": 0.20,
        },
        "normalization": {
            "detection": {"method": "linear_clip", "minimum": 0.5, "maximum": 1.0},
            "size": {"method": "linear_clip", "minimum": 20.0, "maximum": 160.0},
            "sharpness": {
                "method": "median_ratio",
                "fallback_scale": 100.0,
                "minimum_scale": 1.0,
                "scale_multiplier": 2.0,
                "scale": 100.0,
            },
            "pose": {
                "method": "mean_angular_penalty",
                "yaw_scale": 75.0,
                "pitch_scale": 60.0,
                "roll_scale": 60.0,
                "missing_angle": 0.0,
            },
        },
    },
    "tracking": {
        "max_frame_gap": 8,
        "min_similarity": 0.38,
        "appearance_weight": 0.85,
        "quality_weight_offset": 0.5,
    },
    "clustering": {
        "threshold": 0.45,
        "min_cluster_faces": 3,
        "corrections": None,
    },
    "selection": {
        "max_per_identity": 20,
        "perceptual_hash_size": 8,
        "duplicate_hamming_threshold": 4,
        "min_frame_gap": 8,
        "source_margin": 0.35,
        "include_unknown": False,
        "materialize_images": True,
        "diversity_weights": {
            "quality": 0.65,
            "hash": 0.20,
            "pose": 0.10,
            "time": 0.05,
        },
        "pose_scale": 60.0,
        "time_scale": 100.0,
        "cross_video_time_distance": 100.0,
        "minimum_pose_distance": 0.0,
    },
    "paths": {
        "run_dir": None,
        "experiments_dir": None,
        "labels": None,
    },
    "labels": {
        "face_id_column": "face_id",
        "positive_column": "is_acceptable",
        "positive_values": ["1", "true", "yes", "acceptable", "accepted"],
        "identity_column": "true_person",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_experiment_config(config: dict[str, Any]) -> None:
    if config.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Ожидалась схема конфигурации {CONFIG_SCHEMA_VERSION}, "
            f"получено {config.get('config_schema_version')!r}"
        )
    if not isinstance(config.get("experiment_name"), str) or not config["experiment_name"]:
        raise ValueError("experiment_name должен быть непустой строкой")
    if not isinstance(config.get("random_seed"), int) or config["random_seed"] < 0:
        raise ValueError("random_seed должен быть неотрицательным целым")
    extraction = config["feature_extraction"]
    if extraction.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Поддерживается только схема признаков {FEATURE_SCHEMA_VERSION}")
    dark = int(extraction["dark_pixel_threshold"])
    bright = int(extraction["bright_pixel_threshold"])
    if not 0 <= dark < bright <= 255:
        raise ValueError("Нужны пороги пикселей 0 <= dark < bright <= 255")
    if int(extraction["batch_size"]) <= 0:
        raise ValueError("feature_extraction.batch_size должен быть положительным")
    validate_policy(config["policy"])
    validate_ranking(config["ranking"])

    tracking = config["tracking"]
    if int(tracking["max_frame_gap"]) < 0:
        raise ValueError("tracking.max_frame_gap не может быть отрицательным")
    if not -1.0 <= float(tracking["min_similarity"]) <= 1.0:
        raise ValueError("tracking.min_similarity должен быть в диапазоне -1..1")
    if not 0.0 <= float(tracking["appearance_weight"]) <= 1.0:
        raise ValueError("tracking.appearance_weight должен быть в диапазоне 0..1")
    if float(tracking["quality_weight_offset"]) < 0:
        raise ValueError("tracking.quality_weight_offset не может быть отрицательным")

    clustering = config["clustering"]
    if not -1.0 <= float(clustering["threshold"]) <= 1.0:
        raise ValueError("clustering.threshold должен быть в диапазоне -1..1")
    if int(clustering["min_cluster_faces"]) <= 0:
        raise ValueError("clustering.min_cluster_faces должен быть положительным")

    selection = config["selection"]
    for key in ("max_per_identity", "perceptual_hash_size"):
        if int(selection[key]) <= 0:
            raise ValueError(f"selection.{key} должен быть положительным")
    for key in ("duplicate_hamming_threshold", "min_frame_gap"):
        if int(selection[key]) < 0:
            raise ValueError(f"selection.{key} не может быть отрицательным")
    if float(selection["source_margin"]) < 0:
        raise ValueError("selection.source_margin не может быть отрицательным")
    hash_bits = int(selection["perceptual_hash_size"]) ** 2
    if int(selection["duplicate_hamming_threshold"]) > hash_bits:
        raise ValueError("Порог Хэмминга не может быть больше числа битов хеша")
    weights = selection["diversity_weights"]
    if set(weights) != {"quality", "hash", "pose", "time"}:
        raise ValueError("selection.diversity_weights должен содержать quality/hash/pose/time")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("Веса разнообразия не могут быть отрицательными")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Сумма весов разнообразия должна быть равна 1")
    for key in ("pose_scale", "time_scale", "cross_video_time_distance"):
        if float(selection[key]) <= 0:
            raise ValueError(f"selection.{key} должен быть положительным")
    if float(selection["minimum_pose_distance"]) < 0:
        raise ValueError("selection.minimum_pose_distance не может быть отрицательным")


def load_experiment_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать конфигурацию {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Корень конфигурации должен быть объектом")
    resolved = _deep_merge(DEFAULT_CONFIG, value)
    validate_experiment_config(resolved)
    return resolved


def resolve_paths(
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path | None = None,
    experiments_dir: Path | None = None,
    labels: Path | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    base = config_path.resolve().parent

    def resolved(value: str | Path | None, fallback: Path | None) -> str | None:
        if fallback is not None:
            return str(Path(fallback).expanduser().resolve())
        item = value
        if item is None:
            return None
        path = Path(item).expanduser()
        return str((base / path).resolve() if not path.is_absolute() else path.resolve())

    paths = result["paths"]
    paths["run_dir"] = resolved(paths.get("run_dir"), run_dir)
    if paths["run_dir"] is None:
        raise ValueError("Каталог тяжёлого кэша задаётся через --run-dir или paths.run_dir")
    default_experiments = Path(paths["run_dir"]) / "experiments"
    paths["experiments_dir"] = resolved(
        paths.get("experiments_dir"), experiments_dir or default_experiments
    )
    paths["labels"] = resolved(paths.get("labels"), labels)
    corrections = result["clustering"].get("corrections")
    result["clustering"]["corrections"] = resolved(corrections, None)
    validate_experiment_config(result)
    return result
