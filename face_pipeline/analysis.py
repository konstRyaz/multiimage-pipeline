from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contact_sheets import make_contact_sheet
from .io import as_int, load_face_run, prepare_output_dir, write_csv, write_json
from .quality import QualityThresholds, quality_scores, suspicious_reasons


def nearest_neighbors(vectors: np.ndarray, block_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    count = len(vectors)
    best_index = np.full(count, -1, dtype=np.int64)
    best_score = np.full(count, -np.inf, dtype=np.float32)
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        similarities = vectors[start:stop] @ vectors.T
        local = np.arange(stop - start)
        similarities[local, np.arange(start, stop)] = -np.inf
        indices = np.argmax(similarities, axis=1)
        best_index[start:stop] = indices
        best_score[start:stop] = similarities[local, indices]
    return best_index, best_score


def sampled_pair_similarities(vectors: np.ndarray, max_pairs: int, seed: int = 42) -> tuple[np.ndarray, str]:
    count = len(vectors)
    total = count * (count - 1) // 2
    if total <= max_pairs:
        upper = np.triu_indices(count, k=1)
        return np.sum(vectors[upper[0]] * vectors[upper[1]], axis=1), "exact"
    rng = np.random.default_rng(seed)
    left = rng.integers(0, count, size=max_pairs * 2)
    right = rng.integers(0, count, size=max_pairs * 2)
    mask = left != right
    left, right = left[mask][:max_pairs], right[mask][:max_pairs]
    return np.sum(vectors[left] * vectors[right], axis=1), "sampled"


def component_counts(similarity: np.ndarray, thresholds: Sequence[float]) -> list[dict[str, Any]]:
    count = similarity.shape[0]
    results = []
    for threshold in thresholds:
        parent = list(range(count))
        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        left, right = np.where(np.triu(similarity >= threshold, k=1))
        for a, b in zip(left.tolist(), right.tolist()):
            union(a, b)
        sizes: dict[int, int] = defaultdict(int)
        for item in range(count):
            sizes[find(item)] += 1
        values = sorted(sizes.values(), reverse=True)
        results.append({
            "threshold": threshold,
            "connected_components": len(values),
            "components_ge_2": sum(size >= 2 for size in values),
            "largest_component": values[0] if values else 0,
            "singleton_components": sum(size == 1 for size in values),
            "note": "Диагностические связные компоненты; это не итоговая кластеризация",
        })
    return results


def run_analysis(
    run_dir: Path,
    overwrite: bool = False,
    max_pairs: int = 1_000_000,
    contact_sheet_limit: int = 1000,
    thresholds: Sequence[float] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    rows, vectors = load_face_run(run_dir, require_images=False)
    output_dir = run_dir / "analysis"
    prepare_output_dir(output_dir, overwrite)
    scores = quality_scores(rows)

    pair_scores, pair_mode = sampled_pair_similarities(vectors, max_pairs=max_pairs)
    hist_counts, hist_edges = np.histogram(pair_scores, bins=np.linspace(-1.0, 1.0, 81))
    histogram_rows = [
        {"bin_left": round(float(hist_edges[i]), 4), "bin_right": round(float(hist_edges[i + 1]), 4),
         "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]
    write_csv(output_dir / "similarity_histogram.csv", histogram_rows)

    nearest_index, nearest_score = nearest_neighbors(vectors)
    nearest_rows = []
    for index, row in enumerate(rows):
        neighbor = rows[int(nearest_index[index])]
        nearest_rows.append({
            "face_id": row["face_id"], "neighbor_face_id": neighbor["face_id"],
            "similarity": round(float(nearest_score[index]), 7),
            "same_video": int(row["video_name"] == neighbor["video_name"]),
            "frame_gap": abs(as_int(row, "frame_number") - as_int(neighbor, "frame_number")),
        })
    write_csv(output_dir / "nearest_neighbors.csv", nearest_rows)

    by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[(row["video_name"], as_int(row, "frame_number"))].append(index)
    same_frame_rows = []
    for (video, frame), indices in sorted(by_frame.items()):
        for left, right in itertools.combinations(indices, 2):
            same_frame_rows.append({
                "video_name": video, "frame_number": frame,
                "face_id_a": rows[left]["face_id"], "face_id_b": rows[right]["face_id"],
                "similarity": round(float(vectors[left] @ vectors[right]), 7),
            })
    write_csv(output_dir / "same_frame_pairs.csv", same_frame_rows,
              ["video_name", "frame_number", "face_id_a", "face_id_b", "similarity"])

    suspicious_rows = []
    quality_thresholds = QualityThresholds()
    for row, score in zip(rows, scores):
        reasons = suspicious_reasons(row, quality_thresholds)
        if reasons:
            suspicious_rows.append({
                "face_id": row["face_id"], "source_frame": row["source_frame"],
                "quality_score": round(float(score), 6), "reasons": ";".join(reasons),
            })
    write_csv(output_dir / "suspicious_faces.csv", suspicious_rows,
              ["face_id", "source_frame", "quality_score", "reasons"])

    if len(rows) <= 4000:
        similarity = vectors @ vectors.T
        sweep = component_counts(similarity, thresholds)
        threshold_mode = "exact"
    else:
        sweep = [{"threshold": value, "connected_components": "skipped",
                  "note": "N > 4000; используйте кластеризацию дорожек"} for value in thresholds]
        threshold_mode = "skipped"
    write_csv(output_dir / "threshold_sweep.csv", sweep)

    temporal_dir = output_dir / "temporal_contact_sheets"
    temporal_dir.mkdir()
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["video_name"], as_int(rows[i], "frame_number"), as_int(rows[i], "face_index")))
    order = order[:contact_sheet_limit]
    page_size = 80
    for page_index in range(0, len(order), page_size):
        indices = order[page_index:page_index + page_size]
        items = [(rows[i], f"f={rows[i]['frame_number']} q={scores[i]:.2f}") for i in indices]
        make_contact_sheet(items, run_dir, temporal_dir / f"page_{page_index // page_size + 1:03d}.jpg")

    same_values = np.asarray([float(row["similarity"]) for row in same_frame_rows], dtype=np.float32)
    summary = {
        "faces": len(rows),
        "embedding_dimension": int(vectors.shape[1]),
        "pair_statistics_mode": pair_mode,
        "sampled_pairs": int(pair_scores.size),
        "pair_similarity": {
            "min": round(float(pair_scores.min()), 6),
            "p50": round(float(np.quantile(pair_scores, 0.50)), 6),
            "p95": round(float(np.quantile(pair_scores, 0.95)), 6),
            "p99": round(float(np.quantile(pair_scores, 0.99)), 6),
            "max": round(float(pair_scores.max()), 6),
        },
        "same_frame_pairs": len(same_frame_rows),
        "same_frame_max_similarity": round(float(same_values.max()), 6) if same_values.size else None,
        "suspicious_faces": len(suspicious_rows),
        "threshold_sweep_mode": threshold_mode,
        "contact_sheet_faces": len(order),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
