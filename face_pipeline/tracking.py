from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io import as_float, as_int, bbox, iou, load_face_run, prepare_output_dir, save_npy, write_csv, write_json
from .quality import QualityThresholds, quality_scores, suspicious_reasons


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind: list[int] = []
    col_ind: list[int] = []
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            row = p[j] - 1
            col = j - 1
            row_ind.append(row)
            col_ind.append(col)

    row_arr = np.asarray(row_ind, dtype=int)
    col_arr = np.asarray(col_ind, dtype=int)
    if transposed:
        return col_arr, row_arr
    return row_arr, col_arr


@dataclass
class TrackState:
    track_id: str
    video_name: str
    members: list[int] = field(default_factory=list)
    last_frame: int = -1
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    weighted_sum: np.ndarray | None = None
    total_weight: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        assert self.weighted_sum is not None
        value = self.weighted_sum / np.linalg.norm(self.weighted_sum)
        return value.astype(np.float32)

    def add(
        self,
        index: int,
        frame: int,
        box: tuple[float, float, float, float],
        vector: np.ndarray,
        quality: float,
        quality_weight_offset: float = 0.5,
    ) -> None:
        weight = quality_weight_offset + float(quality)
        if self.weighted_sum is None:
            self.weighted_sum = np.zeros_like(vector, dtype=np.float64)
        self.weighted_sum += vector * weight
        self.total_weight += weight
        self.members.append(index)
        self.last_frame = frame
        self.last_bbox = box


def run_tracking(
    run_dir: Path,
    overwrite: bool = False,
    max_frame_gap: int = 8,
    min_similarity: float = 0.38,
    appearance_weight: float = 0.85,
    quality_weight_offset: float = 0.5,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    rows, vectors = load_face_run(run_dir)
    output_dir = run_dir / "tracking"
    prepare_output_dir(output_dir, overwrite)
    overrides = np.asarray(
        [as_float(row, "quality_score_override", np.nan) for row in rows],
        dtype=np.float32,
    )
    qualities = overrides if np.isfinite(overrides).all() else quality_scores(rows)
    reasons = [suspicious_reasons(row, QualityThresholds()) for row in rows]

    by_video_frame: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_video_frame[row["video_name"]][as_int(row, "frame_number")].append(index)

    tracks: list[TrackState] = []
    face_assignment: dict[int, tuple[str, float, float]] = {}
    for video_name in sorted(by_video_frame):
        active: list[TrackState] = []
        for frame, detections in sorted(by_video_frame[video_name].items()):
            active = [track for track in active if frame - track.last_frame <= max_frame_gap]
            assigned_detections: set[int] = set()
            assigned_tracks: set[int] = set()
            if active and detections:
                score_matrix = np.full((len(active), len(detections)), -1e6, dtype=np.float64)
                similarity_matrix = np.full_like(score_matrix, -1.0)
                iou_matrix = np.zeros_like(score_matrix)
                for track_index, track in enumerate(active):
                    gap = frame - track.last_frame
                    for detection_index, row_index in enumerate(detections):
                        similarity = float(track.centroid @ vectors[row_index])
                        overlap = iou(track.last_bbox, bbox(rows[row_index]))
                        similarity_matrix[track_index, detection_index] = similarity
                        iou_matrix[track_index, detection_index] = overlap
                        if similarity >= min_similarity:
                            score_matrix[track_index, detection_index] = (
                                appearance_weight * similarity
                                + (1.0 - appearance_weight) * overlap
                                - 0.01 * max(0, gap - 1)
                            )
                track_indices, detection_indices = linear_sum_assignment(-score_matrix)
                for track_index, detection_index in zip(track_indices.tolist(), detection_indices.tolist()):
                    if score_matrix[track_index, detection_index] <= -1e5:
                        continue
                    track = active[track_index]
                    row_index = detections[detection_index]
                    similarity = float(similarity_matrix[track_index, detection_index])
                    overlap = float(iou_matrix[track_index, detection_index])
                    track.add(
                        row_index,
                        frame,
                        bbox(rows[row_index]),
                        vectors[row_index],
                        float(qualities[row_index]),
                        quality_weight_offset,
                    )
                    face_assignment[row_index] = (track.track_id, similarity, overlap)
                    assigned_tracks.add(track_index)
                    assigned_detections.add(detection_index)

            for detection_index, row_index in enumerate(detections):
                if detection_index in assigned_detections:
                    continue
                track = TrackState(track_id=f"track_{len(tracks):05d}", video_name=video_name)
                track.add(
                    row_index,
                    frame,
                    bbox(rows[row_index]),
                    vectors[row_index],
                    float(qualities[row_index]),
                    quality_weight_offset,
                )
                tracks.append(track)
                active.append(track)
                face_assignment[row_index] = (track.track_id, 1.0, 1.0)

    track_rows: list[dict[str, Any]] = []
    track_embeddings: list[np.ndarray] = []
    for track in tracks:
        center = track.centroid
        similarities = vectors[track.members] @ center
        member_qualities = qualities[track.members]
        best_local = int(np.argmax(member_qualities + 0.1 * similarities))
        best_index = track.members[best_local]
        track_rows.append({
            "track_id": track.track_id,
            "video_name": track.video_name,
            "start_frame": min(as_int(rows[i], "frame_number") for i in track.members),
            "end_frame": max(as_int(rows[i], "frame_number") for i in track.members),
            "num_faces": len(track.members),
            "mean_quality": round(float(np.mean(member_qualities)), 6),
            "mean_intra_similarity": round(float(np.mean(similarities)), 6),
            "min_intra_similarity": round(float(np.min(similarities)), 6),
            "best_face_id": rows[best_index]["face_id"],
            "track_embedding_index": len(track_embeddings),
        })
        track_embeddings.append(center)

    tracked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        track_id, similarity, overlap = face_assignment[index]
        tracked_rows.append({
            **row,
            "quality_score": round(float(qualities[index]), 6),
            "suspicious_reasons": ";".join(reasons[index]),
            "track_id": track_id,
            "track_similarity": round(similarity, 6),
            "track_iou": round(overlap, 6),
        })

    write_csv(output_dir / "faces_tracked.csv", tracked_rows)
    write_csv(output_dir / "tracks.csv", track_rows)
    matrix = np.stack(track_embeddings).astype(np.float32)
    save_npy(output_dir / "track_embeddings.npy", matrix)
    summary = {
        "faces": len(rows), "tracks": len(tracks),
        "mean_track_length": round(float(np.mean([len(track.members) for track in tracks])), 4),
        "max_track_length": max(len(track.members) for track in tracks),
        "max_frame_gap": max_frame_gap, "min_similarity": min_similarity,
        "appearance_weight": appearance_weight,
        "quality_weight_offset": quality_weight_offset,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
