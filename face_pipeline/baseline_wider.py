from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def box_iou_matrix(predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if len(predictions) == 0 or len(ground_truth) == 0:
        return np.zeros((len(predictions), len(ground_truth)), dtype=np.float64)
    left = np.maximum(predictions[:, None, :2], ground_truth[None, :, :2])
    right = np.minimum(predictions[:, None, 2:4], ground_truth[None, :, 2:4])
    # Официальный код WIDER использует включительные координаты (+1).
    intersection = np.prod(np.maximum(0.0, right - left + 1.0), axis=2)
    pred_area = np.prod(np.maximum(0.0, predictions[:, 2:4] - predictions[:, :2] + 1.0), axis=1)
    gt_area = np.prod(np.maximum(0.0, ground_truth[:, 2:4] - ground_truth[:, :2] + 1.0), axis=1)
    return intersection / np.maximum(pred_area[:, None] + gt_area[None, :] - intersection, 1e-12)


def detection_counts(
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for predicted, actual in zip(predictions, ground_truth):
        kept = np.asarray(predicted, dtype=np.float64)
        kept = kept[kept[:, 4] >= threshold] if len(kept) else np.empty((0, 5))
        kept = kept[np.argsort(kept[:, 4])[::-1]] if len(kept) else kept
        overlaps = box_iou_matrix(kept[:, :4], np.asarray(actual, dtype=np.float64))
        matched: set[int] = set()
        for row in range(len(kept)):
            if overlaps.shape[1] == 0:
                fp += 1
                continue
            order = np.argsort(overlaps[row])[::-1]
            match = next((int(index) for index in order if index not in matched and overlaps[row, index] >= iou_threshold), None)
            if match is None:
                fp += 1
            else:
                matched.add(match)
                tp += 1
        fn += len(actual) - len(matched)
    return tp, fp, fn


def detector_curve(
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    thresholds: Sequence[float],
    iou_threshold: float = 0.5,
) -> list[dict[str, float | int]]:
    output = []
    for threshold in sorted(set(map(float, thresholds))):
        tp, fp, fn = detection_counts(predictions, ground_truth, threshold, iou_threshold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output.append({"threshold": threshold, "precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                       "tp": tp, "fp": fp, "fn": fn})
    return output


def select_detector_threshold(curve: Sequence[dict[str, float | int]], min_precision: float | None = None) -> dict[str, float | int] | None:
    eligible = [row for row in curve if min_precision is None or float(row["precision"]) >= min_precision]
    return max(eligible, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["threshold"]))) if eligible else None


def _cell_strings(value: np.ndarray) -> list[str]:
    result = []
    for item in value.ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        result.append(str(item))
    return result


def load_wider_subset(eval_dir: Path, subset: str) -> dict[str, set[int]]:
    """Прочитать официальные 1-based индексы Easy/Medium/Hard из MAT-файлов."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("Для официально совместимого WIDER AP нужна scipy") from exc
    ground = loadmat(eval_dir / "wider_face_val.mat")
    subset_data = loadmat(eval_dir / f"wider_{subset}_val.mat")
    events = _cell_strings(ground["event_list"])
    file_lists = ground["file_list"].ravel()
    gt_lists = subset_data["gt_list"].ravel()
    output: dict[str, set[int]] = {}
    for event_index, event in enumerate(events):
        filenames = _cell_strings(file_lists[event_index])
        keep_cells = gt_lists[event_index].ravel()
        for image_index, filename in enumerate(filenames):
            keep = np.asarray(keep_cells[image_index]).astype(np.int64).ravel()
            output[f"{event}/{filename}.jpg"] = {int(value) - 1 for value in keep if int(value) > 0}
    return output


def _image_eval(predicted: np.ndarray, actual: np.ndarray, keep: set[int], iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    predicted = predicted[np.argsort(predicted[:, 4])[::-1]] if len(predicted) else predicted
    overlaps = box_iou_matrix(predicted[:, :4], actual)
    proposal = np.ones(len(predicted), dtype=np.int64)
    recalled: set[int] = set()
    cumulative = np.zeros(len(predicted), dtype=np.int64)
    for index in range(len(predicted)):
        if overlaps.shape[1]:
            match = int(np.argmax(overlaps[index]))
            if overlaps[index, match] >= iou_threshold:
                if match in keep:
                    recalled.add(match)
                else:
                    proposal[index] = -1
        cumulative[index] = len(recalled)
    return proposal, cumulative


def wider_compatible_ap(
    relative_paths: Sequence[str],
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    keep_by_path: dict[str, set[int]],
    iou_threshold: float = 0.5,
    points: int = 1000,
) -> float:
    """Совместимый с официальным widerface_evaluate дискретный протокол AP."""
    all_scores = np.concatenate([item[:, 4] for item in predictions if len(item)]) if any(len(item) for item in predictions) else np.asarray([])
    if len(all_scores) == 0:
        return 0.0
    minimum, maximum = float(all_scores.min()), float(all_scores.max())
    denominator = maximum - minimum
    normalized = []
    for item in predictions:
        if not len(item):
            normalized.append(item.copy())
        elif denominator > 0:
            value = np.column_stack((item[:, :4], (item[:, 4] - minimum) / denominator))
            normalized.append(value[np.argsort(value[:, 4])[::-1]])
        else:
            value = np.column_stack((item[:, :4], np.ones(len(item), dtype=np.float64)))
            normalized.append(value[np.argsort(value[:, 4])[::-1]])
    total_gt = sum(len(keep_by_path.get(path, set())) for path in relative_paths)
    pr = np.zeros((points, 2), dtype=np.float64)
    for path, predicted, actual in zip(relative_paths, normalized, ground_truth):
        proposal, recall = _image_eval(predicted, actual, keep_by_path.get(path, set()), iou_threshold)
        for index in range(points):
            cutoff = 1.0 - (index + 1) / points
            selected = np.where(predicted[:, 4] >= cutoff)[0]
            if len(selected):
                last = selected[-1]
                pr[index, 0] += int(np.sum(proposal[:last + 1] == 1))
                pr[index, 1] += recall[last]
    precision = np.divide(pr[:, 1], pr[:, 0], out=np.zeros(points), where=pr[:, 0] > 0)
    recall = pr[:, 1] / max(total_gt, 1)
    augmented_recall = np.concatenate(([0.0], recall, [1.0]))
    augmented_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(augmented_precision) - 2, -1, -1):
        augmented_precision[index] = max(augmented_precision[index], augmented_precision[index + 1])
    changes = np.where(augmented_recall[1:] != augmented_recall[:-1])[0]
    return float(np.sum((augmented_recall[changes + 1] - augmented_recall[changes]) * augmented_precision[changes + 1]))
