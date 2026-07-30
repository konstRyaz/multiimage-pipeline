from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .contact_sheets import make_contact_sheet
from .io import as_float, as_int, normalize_rows, prepare_output_dir, read_csv, save_npy, write_csv, write_json


def overlap(a: dict[str, str], b: dict[str, str]) -> bool:
    if a["video_name"] != b["video_name"]:
        return False
    return max(as_int(a, "start_frame"), as_int(b, "start_frame")) <= min(
        as_int(a, "end_frame"), as_int(b, "end_frame")
    )


def automatic_groups(track_rows: list[dict[str, str]], vectors: np.ndarray, threshold: float) -> list[set[int]]:
    similarities = vectors @ vectors.T
    groups: dict[int, set[int]] = {index: {index} for index in range(len(track_rows))}
    owner = list(range(len(track_rows)))

    def root(item: int) -> int:
        while owner[item] != item:
            owner[item] = owner[owner[item]]
            item = owner[item]
        return item

    candidate_left, candidate_right = np.where(np.triu(similarities >= threshold, k=1))
    candidate_scores = similarities[candidate_left, candidate_right]
    order = np.argsort(candidate_scores)[::-1]
    for position in order.tolist():
        left = int(candidate_left[position])
        right = int(candidate_right[position])
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            continue
        left_group, right_group = groups[left_root], groups[right_root]
        if any(overlap(track_rows[a], track_rows[b]) for a in left_group for b in right_group):
            continue
        if min(float(similarities[a, b]) for a in left_group for b in right_group) < threshold:
            continue
        if len(left_group) < len(right_group):
            left_root, right_root = right_root, left_root
            left_group, right_group = right_group, left_group
        owner[right_root] = left_root
        left_group.update(right_group)
        groups[left_root] = left_group
        del groups[right_root]
    return list(groups.values())


def apply_manual_labels(
    groups: list[set[int]], track_rows: list[dict[str, str]], corrections_path: Path | None
) -> list[tuple[str | None, set[int]]]:
    if corrections_path is None:
        return [(None, group) for group in groups]
    corrections = read_csv(corrections_path)
    if corrections and not {"track_id", "identity_label"}.issubset(corrections[0]):
        raise ValueError("В corrections.csv нужны поля track_id и identity_label")
    index_by_id = {row["track_id"]: index for index, row in enumerate(track_rows)}
    labels: dict[int, str] = {}
    for row in corrections:
        track_id = row["track_id"].strip()
        label = row["identity_label"].strip()
        if not label:
            continue
        if track_id not in index_by_id:
            raise ValueError(f"Неизвестная дорожка в corrections.csv: {track_id}")
        labels[index_by_id[track_id]] = label

    merged: dict[str, set[int]] = {}
    untouched: list[tuple[str | None, set[int]]] = []
    for group in groups:
        group_labels = {labels[index] for index in group if index in labels}
        if len(group_labels) > 1:
            raise ValueError(f"Один автоматический кластер получил разные ручные метки: {sorted(group_labels)}")
        if not group_labels:
            untouched.append((None, group))
            continue
        label = next(iter(group_labels))
        merged.setdefault(label, set()).update(group)
    for label, group in merged.items():
        if any(overlap(track_rows[a], track_rows[b]) for a in group for b in group if a < b):
            raise ValueError(f"Ручная метка {label!r} объединяет пересекающиеся дорожки")
        untouched.append((label, group))
    return untouched


def slug(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", value).strip("_")
    return result[:60] or "person"


def run_clustering(
    run_dir: Path,
    overwrite: bool = False,
    threshold: float = 0.45,
    min_cluster_faces: int = 3,
    corrections_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    tracking_dir = run_dir / "tracking"
    track_rows = read_csv(tracking_dir / "tracks.csv")
    face_rows = read_csv(tracking_dir / "faces_tracked.csv")
    track_vectors = normalize_rows(np.load(tracking_dir / "track_embeddings.npy", allow_pickle=False))
    if len(track_rows) != len(track_vectors):
        raise ValueError("Число дорожек не совпадает с числом эмбеддингов дорожек")
    indices = [as_int(row, "track_embedding_index", -1) for row in track_rows]
    if sorted(indices) != list(range(len(track_rows))):
        raise ValueError("Некорректные track_embedding_index")
    track_vectors = track_vectors[indices]

    groups = automatic_groups(track_rows, track_vectors, threshold)
    labeled_groups = apply_manual_labels(groups, track_rows, corrections_path)
    labeled_groups.sort(key=lambda item: sum(as_int(track_rows[i], "num_faces") for i in item[1]), reverse=True)

    output_dir = run_dir / "clustering"
    prepare_output_dir(output_dir, overwrite)
    track_to_cluster: dict[str, str] = {}
    cluster_rows: list[dict[str, Any]] = []
    cluster_vectors: list[np.ndarray] = []
    cluster_members: dict[str, set[int]] = {}
    known_number = 0
    unknown_number = 0
    for manual_label, group in labeled_groups:
        num_faces = sum(as_int(track_rows[i], "num_faces") for i in group)
        is_unknown = manual_label is None and num_faces < min_cluster_faces
        if manual_label is not None:
            cluster_id = f"manual_{slug(manual_label)}"
        elif is_unknown:
            cluster_id = f"unknown_{unknown_number:03d}"
            unknown_number += 1
        else:
            cluster_id = f"person_{known_number:03d}"
            known_number += 1
        weights = np.asarray([max(1, as_int(track_rows[i], "num_faces")) for i in group], dtype=np.float32)
        center = np.average(track_vectors[sorted(group)], axis=0, weights=weights)
        center = (center / np.linalg.norm(center)).astype(np.float32)
        similarities = track_vectors[sorted(group)] @ center
        best_track = sorted(group)[int(np.argmax(similarities))]
        cluster_rows.append({
            "cluster_id": cluster_id,
            "identity_label": manual_label or "",
            "is_unknown": int(is_unknown),
            "num_tracks": len(group),
            "num_faces": num_faces,
            "mean_track_similarity": round(float(np.mean(similarities)), 6),
            "min_track_similarity": round(float(np.min(similarities)), 6),
            "best_face_id": track_rows[best_track]["best_face_id"],
            "cluster_embedding_index": len(cluster_vectors),
        })
        cluster_vectors.append(center)
        cluster_members[cluster_id] = set(group)
        for index in group:
            track_to_cluster[track_rows[index]["track_id"]] = cluster_id

    cluster_matrix = np.stack(cluster_vectors).astype(np.float32)
    cluster_index = {row["cluster_id"]: i for i, row in enumerate(cluster_rows)}
    face_vectors = normalize_rows(np.load(run_dir / "embeddings.npy", allow_pickle=False))
    clustered_faces: list[dict[str, Any]] = []
    for row in face_rows:
        cluster_id = track_to_cluster[row["track_id"]]
        vector = face_vectors[as_int(row, "embedding_index")]
        own_index = cluster_index[cluster_id]
        own_similarity = float(vector @ cluster_matrix[own_index])
        alternatives = vector @ cluster_matrix.T
        alternatives[own_index] = -np.inf
        competitor = float(np.max(alternatives)) if len(cluster_matrix) > 1 else -1.0
        clustered_faces.append({
            **row,
            "cluster_id": cluster_id,
            "similarity_to_cluster": round(own_similarity, 6),
            "nearest_other_similarity": round(competitor, 6),
            "cluster_margin": round(own_similarity - competitor, 6),
        })

    write_csv(output_dir / "faces_clustered.csv", clustered_faces)
    write_csv(output_dir / "clusters.csv", cluster_rows)
    save_npy(output_dir / "cluster_embeddings.npy", cluster_matrix)
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir()
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clustered_faces:
        by_cluster[row["cluster_id"]].append(row)
    for cluster_id, members in by_cluster.items():
        members.sort(key=lambda row: as_float(row, "quality_score", 0.0), reverse=True)
        items = [
            (row, f"f={row['frame_number']} s={as_float(row, 'similarity_to_cluster', 0):.2f}")
            for row in members[:80]
        ]
        make_contact_sheet(items, run_dir, sheets_dir / f"{cluster_id}.jpg")

    summary = {
        "tracks": len(track_rows), "clusters": len(cluster_rows),
        "known_clusters": known_number, "unknown_clusters": unknown_number,
        "threshold": threshold, "min_cluster_faces": min_cluster_faces,
        "manual_corrections": str(corrections_path) if corrections_path else None,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
