from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import read_csv


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    data = list(values)
    if not data:
        return {"identities": 0, "total": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "identities": len(data),
        "total": sum(data),
        "min": min(data),
        "max": max(data),
        "mean": round(statistics.fmean(data), 6),
        "median": round(float(statistics.median(data)), 6),
    }


def _reason_metrics(policy_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(policy_rows)
    counts: Counter[str] = Counter()
    intersections: Counter[str] = Counter()
    for row in policy_rows:
        reasons = sorted(set(str(value) for value in row.get(key, [])))
        counts.update(reasons)
        for size in range(2, len(reasons) + 1):
            intersections.update("&".join(values) for values in itertools.combinations(reasons, size))
    return {
        "counts": dict(sorted(counts.items())),
        "rates": {
            reason: round(count / total, 8) if total else 0.0
            for reason, count in sorted(counts.items())
        },
        "intersections": dict(sorted(intersections.items())),
    }


def load_labels(
    path: Path | None,
    face_id_column: str,
    positive_column: str,
    positive_values: list[str],
    identity_column: str | None = None,
) -> tuple[dict[str, bool], dict[str, str], list[str]]:
    if path is None:
        return {}, {}, []
    rows = read_csv(path)
    if rows and face_id_column not in rows[0]:
        raise ValueError(f"В разметке нужен столбец {face_id_column!r}")
    accepted_values = {value.strip().lower() for value in positive_values}
    has_positive_column = bool(rows and positive_column in rows[0])
    has_noise_column = bool(rows and "is_noise" in rows[0] and not has_positive_column)
    labels: dict[str, bool] = {}
    identities: dict[str, str] = {}
    warnings: list[str] = []
    for row in rows:
        face_id = row[face_id_column].strip()
        if not face_id:
            warnings.append("В разметке пропущена строка с пустым face_id")
            continue
        if face_id in labels:
            raise ValueError(f"Повторяющийся face_id в разметке: {face_id}")
        if has_positive_column:
            labels[face_id] = row[positive_column].strip().lower() in accepted_values
        elif has_noise_column:
            labels[face_id] = row["is_noise"].strip().lower() not in accepted_values
        else:
            raise ValueError(
                f"В разметке нужны столбцы {face_id_column!r} и {positive_column!r}"
            )
        if identity_column and row.get(identity_column, "").strip():
            identities[face_id] = row[identity_column].strip()
    return labels, identities, warnings


def classification_metrics(
    policy_rows: list[dict[str, Any]], labels: dict[str, bool]
) -> dict[str, Any] | None:
    if not labels:
        return None
    tp = fp = fn = tn = labeled = 0
    seen: set[str] = set()
    for row in policy_rows:
        face_id = row["face_id"]
        if face_id not in labels:
            continue
        labeled += 1
        seen.add(face_id)
        truth = labels[face_id]
        prediction = bool(row["accepted"])
        if prediction and truth:
            tp += 1
        elif prediction and not truth:
            fp += 1
        elif not prediction and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "positive_class": "пригодное/принимаемое изображение",
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "labeled_faces": labeled,
        "unlabeled_faces": len(policy_rows) - labeled,
        "labels_without_face": len(set(labels) - seen),
    }


def _identity_counts(
    face_ids: Iterable[str], identity_by_face: dict[str, str], identities: set[str]
) -> dict[str, int]:
    counts = {identity: 0 for identity in identities}
    for face_id in face_ids:
        identity = identity_by_face.get(face_id)
        if identity is not None:
            counts[identity] = counts.get(identity, 0) + 1
    return counts


def aggregate_metrics(
    policy_rows: list[dict[str, Any]],
    clustered_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    selection_decisions: list[dict[str, Any]],
    timings: dict[str, float],
    cache_info: dict[str, Any],
    labels: dict[str, bool] | None = None,
    identity_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    labels = labels or {}
    identity_by_face = dict(identity_labels or {})
    for row in clustered_rows:
        identity_by_face.setdefault(row["face_id"], row["cluster_id"])
    identities = set(identity_by_face.values())
    all_ids = [row["face_id"] for row in policy_rows]
    accepted_ids = [row["face_id"] for row in policy_rows if row["accepted"]]
    selected_ids = [row["face_id"] for row in selection_rows]
    before_counts = _identity_counts(all_ids, identity_by_face, identities)
    after_counts = _identity_counts(accepted_ids, identity_by_face, identities)
    selected_counts = _identity_counts(selected_ids, identity_by_face, identities)
    rejected = len(policy_rows) - len(accepted_ids)

    exclusion_counts: Counter[str] = Counter()
    for row in selection_decisions:
        reasons = row.get("not_selected_reasons", [])
        if isinstance(reasons, str):
            reasons = [value for value in reasons.split(";") if value]
        exclusion_counts.update(reasons)

    return {
        "counts": {
            "images": len({row.get("source_frame", "") for row in policy_rows}),
            "faces": len(policy_rows),
            "accepted": len(accepted_ids),
            "rejected": rejected,
            "accepted_rate": round(len(accepted_ids) / len(policy_rows), 8) if policy_rows else 0.0,
            "rejected_rate": round(rejected / len(policy_rows), 8) if policy_rows else 0.0,
            "selected": len(selection_rows),
        },
        "reject_reasons": _reason_metrics(policy_rows, "reject_reasons"),
        "shadow_reasons": _reason_metrics(policy_rows, "shadow_reasons"),
        "identities": {
            "counts_before_filter": dict(sorted(before_counts.items())),
            "counts_after_filter": dict(sorted(after_counts.items())),
            "counts_after_selection": dict(sorted(selected_counts.items())),
            "without_faces_after_filter": sum(value == 0 for value in after_counts.values()),
            "without_faces_after_selection": sum(value == 0 for value in selected_counts.values()),
            "distribution_before_filter": _distribution(before_counts.values()),
            "distribution_after_filter": _distribution(after_counts.values()),
            "distribution_after_selection": _distribution(selected_counts.values()),
            "identity_source": "labels_and_clusters" if identity_labels else "resulting_clusters",
        },
        "selection_exclusions": dict(sorted(exclusion_counts.items())),
        "classification": classification_metrics(policy_rows, labels),
        "timings_seconds": {
            key: round(float(value), 6)
            for key, value in timings.items()
            if math.isfinite(float(value))
        },
        "feature_cache": {
            "hit": bool(cache_info.get("cache_hit")),
            "path": cache_info.get("cache_dir"),
            "fingerprint": cache_info.get("manifest", {})
            .get("input_fingerprint", {})
            .get("sha256"),
        },
    }
