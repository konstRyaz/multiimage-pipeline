from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .io import as_float


@dataclass(frozen=True)
class QualityThresholds:
    min_det_score: float = 0.65
    min_face_size: float = 32.0
    min_blur_score: float = 20.0
    max_abs_yaw: float = 55.0
    max_abs_pitch: float = 45.0
    max_abs_roll: float = 45.0


def _pose_value(row: dict[str, Any], name: str) -> float:
    value = as_float(row, name, math.nan)
    return value if math.isfinite(value) else 0.0


def quality_scores(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0,), dtype=np.float32)
    blur_values = np.asarray([max(0.0, as_float(row, "blur_score", 0.0)) for row in rows])
    positive = blur_values[blur_values > 0]
    blur_scale = float(np.median(positive)) if positive.size else 100.0
    blur_scale = max(blur_scale, 1.0)
    scores = []
    for row, blur in zip(rows, blur_values):
        det = np.clip((as_float(row, "det_score", 0.0) - 0.5) / 0.5, 0.0, 1.0)
        side = min(as_float(row, "face_width", 0.0), as_float(row, "face_height", 0.0))
        size = np.clip((side - 20.0) / 140.0, 0.0, 1.0)
        sharpness = np.clip(blur / (2.0 * blur_scale), 0.0, 1.0)
        yaw = abs(_pose_value(row, "yaw"))
        pitch = abs(_pose_value(row, "pitch"))
        roll = abs(_pose_value(row, "roll"))
        pose = np.clip(1.0 - (yaw / 75.0 + pitch / 60.0 + roll / 60.0) / 3.0, 0.0, 1.0)
        scores.append(0.30 * det + 0.25 * size + 0.25 * sharpness + 0.20 * pose)
    return np.asarray(scores, dtype=np.float32)


def suspicious_reasons(row: dict[str, Any], thresholds: QualityThresholds) -> list[str]:
    reasons: list[str] = []
    if as_float(row, "det_score", 0.0) < thresholds.min_det_score:
        reasons.append("low_detection_score")
    if min(as_float(row, "face_width", 0.0), as_float(row, "face_height", 0.0)) < thresholds.min_face_size:
        reasons.append("small_face")
    if as_float(row, "blur_score", 0.0) < thresholds.min_blur_score:
        reasons.append("blurred")
    if abs(_pose_value(row, "yaw")) > thresholds.max_abs_yaw:
        reasons.append("large_yaw")
    if abs(_pose_value(row, "pitch")) > thresholds.max_abs_pitch:
        reasons.append("large_pitch")
    if abs(_pose_value(row, "roll")) > thresholds.max_abs_roll:
        reasons.append("large_roll")
    return reasons
