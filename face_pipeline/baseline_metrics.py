from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def binary_counts(labels: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    return (
        int(np.sum(labels & predicted)),
        int(np.sum(~labels & predicted)),
        int(np.sum(labels & ~predicted)),
        int(np.sum(~labels & ~predicted)),
    )


def precision_recall_f1(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn, tn = binary_counts(labels, predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_sweep(scores: Sequence[float], labels: Sequence[bool]) -> list[dict[str, float | int]]:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(scores_array))[::-1], [-np.inf]))
    rows = []
    for threshold in thresholds:
        metrics = precision_recall_f1(labels_array, scores_array >= threshold)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def best_threshold(scores: Sequence[float], labels: Sequence[bool], min_precision: float | None = None) -> tuple[float, dict[str, float | int]]:
    rows = threshold_sweep(scores, labels)
    eligible = [row for row in rows if min_precision is None or float(row["precision"]) >= min_precision]
    if not eligible:
        raise ValueError("Нет порога, удовлетворяющего ограничению precision")
    # Детерминированно: максимум F1, затем recall, затем более высокий порог.
    best = max(eligible, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["threshold"])))
    return float(best["threshold"]), best


def best_accuracy_threshold(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, dict[str, float | int]]:
    rows = threshold_sweep(scores, labels)
    best = max(rows, key=lambda row: (int(row["tp"]) + int(row["tn"]), float(row["f1"]), float(row["threshold"])))
    return float(best["threshold"]), best


def roc_points(scores: Sequence[float], labels: Sequence[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = threshold_sweep(scores, labels)
    positives = max(1, sum(bool(value) for value in labels))
    negatives = max(1, len(labels) - sum(bool(value) for value in labels))
    fpr = np.asarray([float(row["fp"]) / negatives for row in rows])
    tpr = np.asarray([float(row["tp"]) / positives for row in rows])
    thresholds = np.asarray([float(row["threshold"]) for row in rows])
    return fpr, tpr, thresholds


def verification_metrics(scores: Sequence[float], labels: Sequence[bool], fars: Sequence[float] = (0.1, 0.01, 0.001, 0.0001)) -> dict[str, object]:
    fpr, tpr, _ = roc_points(scores, labels)
    auc = float(_trapz(tpr, fpr))
    fnr = 1.0 - tpr
    crossing = int(np.argmin(np.abs(fpr - fnr)))
    negatives = len(labels) - sum(bool(value) for value in labels)
    tar = {}
    warnings = []
    for far in fars:
        valid = np.where(fpr <= far)[0]
        tar[str(far)] = float(np.max(tpr[valid])) if len(valid) else 0.0
        if negatives * far < 10:
            warnings.append(f"TAR@FAR={far:g}: {negatives} отрицательных пар недостаточно для устойчивой оценки (ожидается менее 10 ошибок)")
    return {"roc_auc": auc, "eer": float((fpr[crossing] + fnr[crossing]) / 2.0), "tar_at_far": tar, "warnings": warnings}


def xqlfw_cross_validation(
    scores: Sequence[float],
    labels: Sequence[bool],
    folds: Sequence[int],
    fars: Sequence[float] = (0.1, 0.01, 0.001, 0.0001),
) -> dict[str, object]:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=bool)
    folds_array = np.asarray(folds, dtype=np.int64)
    results = []
    for fold in range(10):
        train = folds_array != fold
        test = folds_array == fold
        if int(test.sum()) != 600 or int(train.sum()) != 5400:
            raise ValueError(f"XQLFW: блок {fold} имеет неверный размер")
        threshold, calibration = best_accuracy_threshold(scores_array[train], labels_array[train])
        metrics = precision_recall_f1(labels_array[test], scores_array[test] >= threshold)
        results.append({"fold": fold, "calibration_records": 5400, "evaluation_records": 600,
                        "threshold": threshold, "calibration": calibration, "evaluation": metrics,
                        "accuracy": (int(metrics["tp"]) + int(metrics["tn"])) / 600.0})
    accuracies = np.asarray([float(item["accuracy"]) for item in results])
    thresholds = np.asarray([float(item["threshold"]) for item in results])
    return {"folds": results, "accuracy_mean": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies)),
            "verification_threshold": float(np.median(thresholds)), **verification_metrics(scores_array, labels_array, fars)}


def _choose2(value: int) -> int:
    return value * (value - 1) // 2


def clustering_metrics(true_labels: Sequence[object], predicted_labels: Sequence[object]) -> dict[str, float | int]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Число истинных и предсказанных меток не совпадает")
    n = len(true_labels)
    if n == 0:
        return {name: 0 for name in ("pair_precision", "pair_recall", "pair_f1", "bcubed_precision", "bcubed_recall", "bcubed_f1", "ari", "predicted_clusters", "wrongly_merged_identity_fraction", "fragmented_identity_fraction")}
    cells = Counter(zip(true_labels, predicted_labels))
    true_sizes = Counter(true_labels)
    predicted_sizes = Counter(predicted_labels)
    tp = sum(_choose2(value) for value in cells.values())
    predicted_pairs = sum(_choose2(value) for value in predicted_sizes.values())
    true_pairs = sum(_choose2(value) for value in true_sizes.values())
    pp = tp / predicted_pairs if predicted_pairs else 1.0
    pr = tp / true_pairs if true_pairs else 1.0
    pair_f1 = 2 * pp * pr / (pp + pr) if pp + pr else 0.0
    bp = sum(value * value / predicted_sizes[predicted] for (true, predicted), value in cells.items()) / n
    br = sum(value * value / true_sizes[true] for (true, predicted), value in cells.items()) / n
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    total_pairs = _choose2(n)
    expected = true_pairs * predicted_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (true_pairs + predicted_pairs)
    ari = (tp - expected) / (maximum - expected) if maximum != expected else 1.0
    true_per_pred: dict[object, set[object]] = defaultdict(set)
    pred_per_true: dict[object, set[object]] = defaultdict(set)
    for true, predicted in cells:
        true_per_pred[predicted].add(true)
        pred_per_true[true].add(predicted)
    merged_identities = {identity for identities in true_per_pred.values() if len(identities) > 1 for identity in identities}
    eligible_fragmented = {identity for identity, size in true_sizes.items() if size >= 2}
    fragmented = {identity for identity in eligible_fragmented if len(pred_per_true[identity]) > 1}
    return {"pair_precision": pp, "pair_recall": pr, "pair_f1": pair_f1,
            "bcubed_precision": bp, "bcubed_recall": br, "bcubed_f1": bf, "ari": ari,
            "predicted_clusters": len(predicted_sizes),
            "wrongly_merged_identity_fraction": len(merged_identities) / len(true_sizes),
            "fragmented_identity_fraction": len(fragmented) / len(eligible_fragmented) if eligible_fragmented else 0.0}


def metrics_all_and_multi(true_labels: Sequence[object], predicted_labels: Sequence[object]) -> dict[str, object]:
    all_metrics = clustering_metrics(true_labels, predicted_labels)
    counts = Counter(true_labels)
    mask = [counts[label] >= 2 for label in true_labels]
    return {"all_identities": all_metrics,
            "identities_with_at_least_two_images": clustering_metrics([x for x, keep in zip(true_labels, mask) if keep], [x for x, keep in zip(predicted_labels, mask) if keep])}


def match_main_face(ground_truth: Sequence[float], detections: Sequence[Sequence[float]], min_iou: float = 0.3) -> int | None:
    gx1, gy1, gx2, gy2 = map(float, ground_truth)
    def overlap(box: Sequence[float]) -> float:
        x1, y1, x2, y2 = map(float, box[:4])
        intersection = max(0.0, min(gx2, x2) - max(gx1, x1)) * max(0.0, min(gy2, y2) - max(gy1, y1))
        union = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1) + max(0.0, x2 - x1) * max(0.0, y2 - y1) - intersection
        return intersection / union if union else 0.0
    if len(detections) == 0:
        return None
    values = [overlap(box) for box in detections]
    best = int(np.argmax(values))
    return best if values[best] >= min_iou else None


def blockwise_threshold_clusters(vectors: np.ndarray, threshold: float, block_size: int = 2048) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    parent = np.arange(n, dtype=np.int64)
    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item
    def union(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a
    for left_start in range(0, n, block_size):
        left_stop = min(n, left_start + block_size)
        for right_start in range(left_start, n, block_size):
            right_stop = min(n, right_start + block_size)
            similarities = vectors[left_start:left_stop] @ vectors[right_start:right_stop].T
            rows, columns = np.where(similarities >= threshold)
            for row, column in zip(rows.tolist(), columns.tolist()):
                left, right = left_start + row, right_start + column
                if left < right:
                    union(left, right)
    roots = np.asarray([root(index) for index in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def scalable_threshold_clusters(
    vectors: np.ndarray,
    threshold: float,
    block_size: int = 2048,
    neighbors: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Кластеризация порогового графа без полной матрицы сходства.

    Для малых фикстур используется точный блочный путь. Для полного CelebA —
    детерминированный HNSW из FAISS и объединение найденных соседей.
    """
    return scalable_threshold_cluster_sweep(vectors, [threshold], block_size, neighbors, seed)[float(threshold)]


def scalable_threshold_cluster_sweep(
    vectors: np.ndarray,
    thresholds: Sequence[float],
    block_size: int = 2048,
    neighbors: int = 64,
    seed: int = 42,
) -> dict[float, np.ndarray]:
    np.random.seed(seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    threshold_values = sorted(set(map(float, thresholds)), reverse=True)
    if not threshold_values:
        return {}
    minimum = min(threshold_values)
    parents = [np.arange(len(vectors), dtype=np.int64) for _ in threshold_values]
    def root(parent: np.ndarray, item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item
    def add_edge(score: float, left: int, right: int) -> None:
        for threshold, parent in zip(threshold_values, parents):
            if score < threshold:
                continue
            a, b = root(parent, left), root(parent, right)
            if a != b:
                if a > b:
                    a, b = b, a
                parent[b] = a
    if len(vectors) <= 10_000:
        for left_start in range(0, len(vectors), block_size):
            left_stop = min(len(vectors), left_start + block_size)
            for right_start in range(left_start, len(vectors), block_size):
                right_stop = min(len(vectors), right_start + block_size)
                similarities = vectors[left_start:left_stop] @ vectors[right_start:right_stop].T
                rows, columns = np.where(similarities >= minimum)
                for row, column in zip(rows.tolist(), columns.tolist()):
                    left, right = left_start + row, right_start + column
                    if left < right:
                        add_edge(float(similarities[row, column]), left, right)
    else:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "Для масштабируемой кластеризации полного CelebA установите зависимости "
                "из requirements-baseline.txt (нужен faiss-cpu)"
            ) from exc
        faiss.omp_set_num_threads(1)
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = max(128, neighbors)
        index.add(vectors)
        for start in range(0, len(vectors), block_size):
            stop = min(len(vectors), start + block_size)
            scores, indices = index.search(vectors[start:stop], neighbors)
            for local, (row_scores, row_indices) in enumerate(zip(scores, indices)):
                left = start + local
                for score, right in zip(row_scores.tolist(), row_indices.tolist()):
                    if right > left and score >= minimum:
                        add_edge(float(score), left, int(right))
    output: dict[float, np.ndarray] = {}
    for threshold, parent in zip(threshold_values, parents):
        roots = np.asarray([root(parent, index) for index in range(len(vectors))])
        _, labels = np.unique(roots, return_inverse=True)
        output[threshold] = labels
    return output
