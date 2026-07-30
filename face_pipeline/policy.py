from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


POLICY_MODES = {"off", "shadow", "hard"}
OPERATORS = {"lt", "gt", "abs_gt", "eq"}
KNOWN_RULES = {
    "low_detection_score",
    "small_face",
    "blurred",
    "large_yaw",
    "large_pitch",
    "large_roll",
    "pose_unavailable",
    "face_near_border",
    "too_dark",
    "too_bright",
    "extreme_dark_pixels",
    "extreme_bright_pixels",
    "image_unavailable",
}


@dataclass(frozen=True)
class PolicyDecision:
    accepted: bool
    reject_reasons: tuple[str, ...]
    shadow_reasons: tuple[str, ...]


def validate_policy(policy: dict[str, Any]) -> None:
    if not isinstance(policy.get("version"), str) or not policy["version"]:
        raise ValueError("У политики должна быть непустая строковая version")
    rules = policy.get("rules")
    if not isinstance(rules, dict):
        raise ValueError("policy.rules должен быть объектом")
    unknown = sorted(set(rules) - KNOWN_RULES)
    if unknown:
        raise ValueError("Неизвестные правила политики: " + ", ".join(unknown))
    for rule_id, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Правило {rule_id} должно быть объектом")
        mode = rule.get("mode")
        if mode not in POLICY_MODES:
            raise ValueError(f"Некорректный режим правила {rule_id}: {mode!r}")
        if mode == "off":
            continue
        if not isinstance(rule.get("feature"), str) or not rule["feature"]:
            raise ValueError(f"У правила {rule_id} не задан feature")
        operator = rule.get("operator")
        if operator not in OPERATORS:
            raise ValueError(f"Некорректный operator правила {rule_id}: {operator!r}")
        if "threshold" not in rule:
            raise ValueError(f"У правила {rule_id} не задан threshold")
        threshold = rule["threshold"]
        if operator != "eq":
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError(f"Порог правила {rule_id} должен быть числом")
            if not math.isfinite(float(threshold)):
                raise ValueError(f"Порог правила {rule_id} должен быть конечным")
        if rule_id == "small_face" and float(threshold) < 0:
            raise ValueError("Минимальный размер лица не может быть отрицательным")
        missing = rule.get("missing", "ignore")
        if missing not in {"ignore", "violate"}:
            raise ValueError(f"Некорректное missing у правила {rule_id}: {missing!r}")


def _violates(value: Any, operator: str, threshold: Any, missing: str) -> bool:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return missing == "violate"
    if operator == "lt":
        return float(value) < float(threshold)
    if operator == "gt":
        return float(value) > float(threshold)
    if operator == "abs_gt":
        return abs(float(value)) > float(threshold)
    if operator == "eq":
        return value == threshold
    raise ValueError(f"Неизвестный оператор: {operator}")


def apply_policy(features: dict[str, Any], policy: dict[str, Any]) -> PolicyDecision:
    validate_policy(policy)
    rejected: list[str] = []
    shadow: list[str] = []
    for rule_id, rule in policy["rules"].items():
        mode = rule["mode"]
        if mode == "off":
            continue
        if _violates(
            features.get(rule["feature"]),
            rule["operator"],
            rule["threshold"],
            rule.get("missing", "ignore"),
        ):
            if mode == "hard":
                rejected.append(rule_id)
            else:
                shadow.append(rule_id)
    return PolicyDecision(not rejected, tuple(rejected), tuple(shadow))


def apply_policy_rows(
    features: Sequence[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in features:
        decision = apply_policy(row, policy)
        results.append({
            **row,
            "policy_version": policy["version"],
            "decision": "accepted" if decision.accepted else "rejected",
            "accepted": decision.accepted,
            "reject_reasons": list(decision.reject_reasons),
            "shadow_reasons": list(decision.shadow_reasons),
        })
    return results


def normalized_rating_components(
    features: Sequence[dict[str, Any]],
    ranking: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    validate_ranking(ranking)
    normalizers = ranking["normalization"]
    blur_values = np.asarray(
        [
            max(0.0, float(row["blur_score"]))
            for row in features
            if row.get("blur_score") is not None
        ],
        dtype=np.float64,
    )
    positive = blur_values[blur_values > 0]
    sharp = normalizers["sharpness"]
    if sharp["method"] == "median_ratio":
        blur_scale = float(np.median(positive)) if positive.size else float(sharp["fallback_scale"])
    else:
        blur_scale = float(sharp["scale"])
    blur_scale = max(blur_scale, float(sharp["minimum_scale"]))

    output: list[dict[str, float]] = []
    for row in features:
        det_raw = float(row["det_score"]) if row.get("det_score") is not None else 0.0
        size_raw = float(row["min_face_size"]) if row.get("min_face_size") is not None else 0.0
        blur_raw = float(row["blur_score"]) if row.get("blur_score") is not None else 0.0
        det_cfg = normalizers["detection"]
        size_cfg = normalizers["size"]
        det = float(np.clip(
            (det_raw - float(det_cfg["minimum"])) /
            (float(det_cfg["maximum"]) - float(det_cfg["minimum"])), 0.0, 1.0
        ))
        size = float(np.clip(
            (size_raw - float(size_cfg["minimum"])) /
            (float(size_cfg["maximum"]) - float(size_cfg["minimum"])), 0.0, 1.0
        ))
        sharpness = float(np.clip(
            blur_raw / (float(sharp["scale_multiplier"]) * blur_scale), 0.0, 1.0
        ))
        pose_cfg = normalizers["pose"]
        missing_angle = float(pose_cfg["missing_angle"])
        yaw = abs(float(row["yaw"])) if row.get("yaw") is not None else missing_angle
        pitch = abs(float(row["pitch"])) if row.get("pitch") is not None else missing_angle
        roll = abs(float(row["roll"])) if row.get("roll") is not None else missing_angle
        pose = float(np.clip(
            1.0 - (
                yaw / float(pose_cfg["yaw_scale"])
                + pitch / float(pose_cfg["pitch_scale"])
                + roll / float(pose_cfg["roll_scale"])
            ) / 3.0,
            0.0,
            1.0,
        ))
        output.append({
            "detection_component": det,
            "size_component": size,
            "sharpness_component": sharpness,
            "pose_component": pose,
        })
    return output, {"blur_scale": blur_scale}


def soft_ratings(
    features: Sequence[dict[str, Any]], ranking: dict[str, Any]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    components, context = normalized_rating_components(features, ranking)
    weights = ranking["weights"]
    for row in components:
        row["soft_rating"] = float(sum(float(weights[name]) * row[name] for name in weights))
    return components, context


def validate_ranking(ranking: dict[str, Any]) -> None:
    if not isinstance(ranking.get("version"), str) or not ranking["version"]:
        raise ValueError("У рейтинга должна быть непустая строковая version")
    weights = ranking.get("weights")
    expected = {
        "detection_component",
        "size_component",
        "sharpness_component",
        "pose_component",
    }
    if not isinstance(weights, dict) or set(weights) != expected:
        raise ValueError("ranking.weights должен содержать ровно четыре компонента рейтинга")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("Веса рейтинга не могут быть отрицательными")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Сумма весов рейтинга должна быть равна 1")
    normalization = ranking.get("normalization", {})
    for name in ("detection", "size", "sharpness", "pose"):
        if name not in normalization or not isinstance(normalization[name], dict):
            raise ValueError(f"Не задана нормализация {name}")
    for name in ("detection", "size"):
        low = float(normalization[name]["minimum"])
        high = float(normalization[name]["maximum"])
        if not high > low:
            raise ValueError(f"У нормализации {name} maximum должен быть больше minimum")
    sharp = normalization["sharpness"]
    if sharp.get("method") not in {"median_ratio", "fixed"}:
        raise ValueError("Неизвестный метод нормализации резкости")
    for key in ("fallback_scale", "minimum_scale", "scale_multiplier"):
        if float(sharp[key]) <= 0:
            raise ValueError(f"sharpness.{key} должен быть положительным")
    pose = normalization["pose"]
    for key in ("yaw_scale", "pitch_scale", "roll_scale"):
        if float(pose[key]) <= 0:
            raise ValueError(f"pose.{key} должен быть положительным")
