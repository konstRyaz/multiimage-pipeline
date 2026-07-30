from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .io import DataValidationError, as_float, as_int, load_face_run, read_csv, resolve_path


FEATURE_SCHEMA_VERSION = "quality_features_v1"
DEFAULT_EXTRACTION = {
    "dark_pixel_threshold": 30,
    "bright_pixel_threshold": 225,
}


class FeatureCacheError(ValueError):
    """Кэш дешёвых признаков отсутствует, повреждён или несовместим."""


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = as_float(row, key, math.nan)
    return float(value) if math.isfinite(value) else None


def _positive_dimension(row: dict[str, Any], key: str) -> float | None:
    value = _optional_float(row, key)
    return value if value is not None and value > 0 else None


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Дешёвая оценка резкости без OpenCV."""
    values = np.asarray(gray, dtype=np.float32)
    if values.ndim != 2 or min(values.shape, default=0) < 3:
        return 0.0
    center = values[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
    )
    return float(np.var(laplacian, dtype=np.float64))


def image_features(
    path: Path,
    dark_pixel_threshold: int = 30,
    bright_pixel_threshold: int = 225,
) -> dict[str, float]:
    if not 0 <= dark_pixel_threshold <= 255:
        raise ValueError("dark_pixel_threshold должен быть в диапазоне 0..255")
    if not 0 <= bright_pixel_threshold <= 255:
        raise ValueError("bright_pixel_threshold должен быть в диапазоне 0..255")
    if dark_pixel_threshold >= bright_pixel_threshold:
        raise ValueError("dark_pixel_threshold должен быть меньше bright_pixel_threshold")
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "brightness_mean": float(np.mean(gray)) if gray.size else 0.0,
        "dark_fraction": float(np.mean(gray <= dark_pixel_threshold)) if gray.size else 0.0,
        "bright_fraction": float(np.mean(gray >= bright_pixel_threshold)) if gray.size else 0.0,
        "derived_sharpness": variance_of_laplacian(gray),
    }


def extract_quality_features(
    row: dict[str, Any],
    run_dir: Path,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = {**DEFAULT_EXTRACTION, **(extraction or {})}
    width = _positive_dimension(row, "face_width")
    height = _positive_dimension(row, "face_height")
    x1, y1 = _optional_float(row, "bbox_x1"), _optional_float(row, "bbox_y1")
    x2, y2 = _optional_float(row, "bbox_x2"), _optional_float(row, "bbox_y2")
    if width is None and x1 is not None and x2 is not None and x2 > x1:
        width = x2 - x1
    if height is None and y1 is not None and y2 is not None and y2 > y1:
        height = y2 - y1
    min_face_size = min(width, height) if width is not None and height is not None else None

    frame_width = _positive_dimension(row, "frame_width")
    frame_height = _positive_dimension(row, "frame_height")
    min_frame_size = (
        min(frame_width, frame_height)
        if frame_width is not None and frame_height is not None
        else None
    )
    relative_face_size = (
        min_face_size / min_frame_size
        if min_face_size is not None and min_frame_size is not None
        else None
    )
    border_margin = _optional_float(row, "border_margin")
    if (
        border_margin is None
        and None not in (x1, y1, x2, y2, frame_width, frame_height)
    ):
        border_margin = min(
            float(x1), float(y1), float(frame_width) - float(x2), float(frame_height) - float(y2)
        )
    border_margin_relative = (
        border_margin / min_frame_size
        if border_margin is not None and min_frame_size is not None
        else None
    )

    pitch = _optional_float(row, "pitch")
    yaw = _optional_float(row, "yaw")
    roll = _optional_float(row, "roll")
    pose_available = pitch is not None and yaw is not None and roll is not None
    aligned = resolve_path(str(row.get("aligned_path", "")), run_dir)
    image_values: dict[str, Any] = {
        "brightness_mean": None,
        "dark_fraction": None,
        "bright_fraction": None,
        "derived_sharpness": None,
    }
    image_error: str | None = None
    if aligned is not None:
        try:
            image_values.update(
                image_features(
                    aligned,
                    int(settings["dark_pixel_threshold"]),
                    int(settings["bright_pixel_threshold"]),
                )
            )
        except (OSError, ValueError) as exc:
            image_error = f"{type(exc).__name__}: {exc}"

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "face_id": str(row.get("face_id", "")),
        "embedding_index": as_int(row, "embedding_index", -1),
        "source_frame": str(row.get("source_frame", "")),
        "aligned_path": str(row.get("aligned_path", "")),
        "video_name": str(row.get("video_name", "")),
        "frame_number": as_int(row, "frame_number", 0),
        "det_score": _optional_float(row, "det_score"),
        "bbox_width": width,
        "bbox_height": height,
        "min_face_size": min_face_size,
        "relative_face_size": relative_face_size,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "border_margin": border_margin,
        "border_margin_relative": border_margin_relative,
        "blur_score": _optional_float(row, "blur_score"),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "pose_available": pose_available,
        "image_available": aligned is not None and image_error is None,
        "image_error": image_error,
        **image_values,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprint(run_dir: Path, extraction: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = {**DEFAULT_EXTRACTION, **(extraction or {})}
    faces_path = run_dir / "faces.csv"
    embeddings_path = run_dir / "embeddings.npy"
    if not faces_path.is_file() or not embeddings_path.is_file():
        raise FeatureCacheError("Для кэша нужны faces.csv и embeddings.npy")
    rows = read_csv(faces_path)
    aligned_entries: list[dict[str, Any]] = []
    for row in rows:
        path = resolve_path(str(row.get("aligned_path", "")), run_dir)
        if path is None:
            aligned_entries.append({"face_id": row.get("face_id", ""), "missing": True})
            continue
        stat = path.stat()
        aligned_entries.append({
            "face_id": row.get("face_id", ""),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "faces_sha256": _hash_file(faces_path),
        "embeddings_sha256": _hash_file(embeddings_path),
        "aligned": aligned_entries,
        "extraction": settings,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), **payload}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FeatureCacheError(f"Строка {line_number} кэша не является объектом")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"Не удалось прочитать кэш признаков: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_source(run_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(run_dir / "faces.csv")
    embeddings = np.load(run_dir / "embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise DataValidationError(
            f"Несогласованный тяжёлый кэш: faces={len(rows)}, embeddings={embeddings.shape}"
        )
    if rows:
        load_face_run(run_dir, require_images=False)
    return rows


def load_or_build_feature_cache(
    run_dir: Path,
    extraction: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.resolve()
    settings = {**DEFAULT_EXTRACTION, **(extraction or {})}
    # Валидация параметров выполняется даже при пустом наборе.
    if not 0 <= int(settings["dark_pixel_threshold"]) < int(settings["bright_pixel_threshold"]) <= 255:
        raise ValueError("Пороги тёмных/светлых пикселей должны удовлетворять 0 <= dark < bright <= 255")
    source_rows = _validate_source(run_dir)
    fingerprint = input_fingerprint(run_dir, settings)
    cache_dir = run_dir / "derived" / FEATURE_SCHEMA_VERSION
    manifest_path = cache_dir / "manifest.json"
    rows_path = cache_dir / "features.jsonl"
    warning: str | None = None

    if not force and manifest_path.is_file() and rows_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cached = _read_jsonl(rows_path)
            valid = (
                manifest.get("status") == "complete"
                and manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
                and manifest.get("input_fingerprint", {}).get("sha256") == fingerprint["sha256"]
                and manifest.get("face_count") == len(source_rows) == len(cached)
                and [row.get("face_id") for row in cached]
                == [row.get("face_id") for row in source_rows]
                and [row.get("embedding_index") for row in cached]
                == [as_int(row, "embedding_index", -1) for row in source_rows]
            )
            if valid:
                return cached, {
                    "cache_hit": True,
                    "cache_dir": str(cache_dir),
                    "manifest": manifest,
                    "warning": None,
                }
            warning = "Несовместимый кэш признаков пересчитан"
        except (OSError, json.JSONDecodeError, FeatureCacheError) as exc:
            warning = f"Повреждённый кэш признаков пересчитан: {exc}"
    elif not force and cache_dir.exists():
        warning = "Частичный кэш признаков пересчитан"

    features = [extract_quality_features(row, run_dir, settings) for row in source_rows]
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "complete",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": created,
        "face_count": len(features),
        "input_fingerprint": fingerprint,
        "extraction": settings,
    }
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_dir.with_name(f".{cache_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    _write_jsonl(temp_dir / "features.jsonl", features)
    (temp_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    temp_dir.replace(cache_dir)
    return features, {
        "cache_hit": False,
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "warning": warning,
    }
