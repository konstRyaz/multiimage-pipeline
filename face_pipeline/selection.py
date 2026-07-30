from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .contact_sheets import make_contact_sheet
from .io import as_float, as_int, prepare_output_dir, read_csv, resolve_path, write_csv, write_json


def image_hash(path: Path, size: int = 8) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.float32)
    return (values >= values.mean()).reshape(-1)


def crop_source(row: dict[str, str], run_dir: Path, output_path: Path, margin: float) -> bool:
    source_path = resolve_path(row.get("source_frame", ""), run_dir)
    if source_path is None:
        return False
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        x1, y1 = as_float(row, "bbox_x1", 0), as_float(row, "bbox_y1", 0)
        x2, y2 = as_float(row, "bbox_x2", 0), as_float(row, "bbox_y2", 0)
        width, height = x2 - x1, y2 - y1
        left = max(0, int(round(x1 - width * margin)))
        top = max(0, int(round(y1 - height * margin)))
        right = min(source.width, int(round(x2 + width * margin)))
        bottom = min(source.height, int(round(y2 + height * margin)))
        if right <= left or bottom <= top:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source.crop((left, top, right, bottom)).save(output_path, quality=95)
    return True


def greedy_select(
    rows: list[dict[str, str]], run_dir: Path, maximum: int,
    duplicate_hamming: int, min_frame_gap: int,
) -> list[tuple[dict[str, str], np.ndarray]]:
    selected, _ = greedy_select_detailed(
        rows,
        run_dir,
        maximum,
        duplicate_hamming,
        min_frame_gap,
    )
    return selected


def greedy_select_detailed(
    rows: list[dict[str, str]],
    run_dir: Path,
    maximum: int,
    duplicate_hamming: int,
    min_frame_gap: int,
    hash_size: int = 8,
    diversity_weights: dict[str, float] | None = None,
    pose_scale: float = 60.0,
    time_scale: float = 100.0,
    cross_video_time_distance: float = 100.0,
    minimum_pose_distance: float = 0.0,
) -> tuple[list[tuple[dict[str, str], np.ndarray]], list[dict[str, Any]]]:
    weights = diversity_weights or {
        "quality": 0.65, "hash": 0.20, "pose": 0.10, "time": 0.05,
    }
    candidates: list[tuple[dict[str, str], np.ndarray]] = []
    missing_images: set[str] = set()
    for row in rows:
        path = resolve_path(row.get("aligned_path", ""), run_dir)
        if path is not None:
            candidates.append((row, image_hash(path, hash_size)))
        else:
            missing_images.add(row["face_id"])
    selected: list[tuple[dict[str, str], np.ndarray]] = []
    remaining = list(candidates)
    selected_values: dict[str, float] = {}
    while remaining and len(selected) < maximum:
        best_position = -1
        best_value = -1e9
        for position, (row, bits) in enumerate(remaining):
            quality = as_float(row, "quality_score", 0.0)
            if not selected:
                value = quality
            else:
                hash_distances = [int(np.count_nonzero(bits != chosen_bits)) for _, chosen_bits in selected]
                if min(hash_distances) <= duplicate_hamming:
                    continue
                too_close = any(
                    row["video_name"] == chosen["video_name"]
                    and abs(as_int(row, "frame_number") - as_int(chosen, "frame_number")) < min_frame_gap
                    for chosen, _ in selected
                )
                if too_close:
                    continue
                pose_distance = min(
                    abs(as_float(row, "yaw", 0.0) - as_float(chosen, "yaw", 0.0))
                    for chosen, _ in selected
                )
                if pose_distance < minimum_pose_distance:
                    continue
                time_distance = min(
                    abs(as_int(row, "frame_number") - as_int(chosen, "frame_number"))
                    if row["video_name"] == chosen["video_name"] else cross_video_time_distance
                    for chosen, _ in selected
                )
                value = (
                    float(weights["quality"]) * quality
                    + float(weights["hash"]) * min(hash_distances) / float(hash_size ** 2)
                    + float(weights["pose"]) * min(pose_distance / pose_scale, 1.0)
                    + float(weights["time"]) * min(time_distance / time_scale, 1.0)
                )
            if value > best_value:
                best_value = value
                best_position = position
        if best_position < 0:
            break
        selected_row = remaining.pop(best_position)
        selected.append(selected_row)
        selected_values[selected_row[0]["face_id"]] = best_value

    selected_ids = {row["face_id"] for row, _ in selected}
    details: list[dict[str, Any]] = []
    for row in rows:
        face_id = row["face_id"]
        reasons: list[str] = []
        if face_id in missing_images:
            reasons.append("missing_image")
        elif face_id not in selected_ids:
            bits = next(bits for candidate, bits in candidates if candidate["face_id"] == face_id)
            hash_distances = [
                int(np.count_nonzero(bits != chosen_bits)) for _, chosen_bits in selected
            ]
            if hash_distances and min(hash_distances) <= duplicate_hamming:
                reasons.append("duplicate")
            too_close = any(
                row["video_name"] == chosen["video_name"]
                and abs(as_int(row, "frame_number") - as_int(chosen, "frame_number")) < min_frame_gap
                for chosen, _ in selected
            )
            if too_close:
                reasons.append("temporal_proximity")
            if selected and minimum_pose_distance > 0:
                pose_distance = min(
                    abs(as_float(row, "yaw", 0.0) - as_float(chosen, "yaw", 0.0))
                    for chosen, _ in selected
                )
                if pose_distance < minimum_pose_distance:
                    reasons.append("pose_diversity")
            if not reasons:
                reasons.append("capacity_limit" if len(selected) >= maximum else "lower_ranking")
        details.append({
            "face_id": face_id,
            "selected": face_id in selected_ids,
            "selection_decision": "selected" if face_id in selected_ids else "not_selected",
            "not_selected_reasons": reasons,
            "selection_value": selected_values.get(face_id),
            "quality_score": as_float(row, "quality_score", 0.0),
        })
    return selected, details


def run_selection(
    run_dir: Path,
    overwrite: bool = False,
    max_per_identity: int = 20,
    duplicate_hamming: int = 4,
    min_frame_gap: int = 8,
    source_margin: float = 0.35,
    include_unknown: bool = False,
    hash_size: int = 8,
    diversity_weights: dict[str, float] | None = None,
    pose_scale: float = 60.0,
    time_scale: float = 100.0,
    cross_video_time_distance: float = 100.0,
    minimum_pose_distance: float = 0.0,
    materialize_images: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    faces = read_csv(run_dir / "clustering" / "faces_clustered.csv")
    clusters = read_csv(run_dir / "clustering" / "clusters.csv")
    unknown = {row["cluster_id"] for row in clusters if as_int(row, "is_unknown") == 1}
    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in faces:
        if include_unknown or row["cluster_id"] not in unknown:
            by_cluster[row["cluster_id"]].append(row)

    output_dir = run_dir / "selected"
    prepare_output_dir(output_dir, overwrite)
    selection_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items()):
        selected, details = greedy_select_detailed(
            members,
            run_dir,
            max_per_identity,
            duplicate_hamming,
            min_frame_gap,
            hash_size,
            diversity_weights,
            pose_scale,
            time_scale,
            cross_video_time_distance,
            minimum_pose_distance,
        )
        decision_rows.extend({"cluster_id": cluster_id, **row} for row in details)
        identity_dir = output_dir / cluster_id
        aligned_dir = identity_dir / "aligned"
        source_dir = identity_dir / "source"
        if materialize_images:
            aligned_dir.mkdir(parents=True)
            source_dir.mkdir()
        for rank, (row, _) in enumerate(selected, start=1):
            aligned_source = resolve_path(row["aligned_path"], run_dir)
            assert aligned_source is not None
            aligned_output = aligned_dir / f"{rank:03d}_{row['face_id']}.jpg"
            source_output = source_dir / f"{rank:03d}_{row['face_id']}.jpg"
            if materialize_images:
                shutil.copy2(aligned_source, aligned_output)
                source_saved = crop_source(row, run_dir, source_output, source_margin)
            else:
                source_saved = False
            selection_rows.append({
                "cluster_id": cluster_id, "rank": rank, "face_id": row["face_id"],
                "video_name": row["video_name"], "frame_number": row["frame_number"],
                "quality_score": row.get("quality_score", ""),
                "similarity_to_cluster": row.get("similarity_to_cluster", ""),
                "aligned_output": str(aligned_output.relative_to(run_dir)) if materialize_images else "",
                "source_output": str(source_output.relative_to(run_dir)) if source_saved else "",
            })
        if materialize_images:
            make_contact_sheet(
                [(row, f"#{rank} f={row['frame_number']} q={as_float(row, 'quality_score', 0):.2f}")
                 for rank, (row, _) in enumerate(selected, start=1)],
                run_dir, identity_dir / "contact_sheet.jpg",
            )
    write_csv(output_dir / "selection.csv", selection_rows,
              ["cluster_id", "rank", "face_id", "video_name", "frame_number",
               "quality_score", "similarity_to_cluster", "aligned_output", "source_output"])
    write_csv(
        output_dir / "selection_decisions.csv",
        [
            {**row, "not_selected_reasons": ";".join(row["not_selected_reasons"])}
            for row in decision_rows
        ],
        [
            "cluster_id", "face_id", "selected", "selection_decision",
            "not_selected_reasons", "selection_value", "quality_score",
        ],
    )
    reason_counts: dict[str, int] = defaultdict(int)
    for row in decision_rows:
        for reason in row["not_selected_reasons"]:
            reason_counts[reason] += 1
    summary = {
        "identities": len(by_cluster), "selected_faces": len(selection_rows),
        "max_per_identity": max_per_identity, "duplicate_hamming": duplicate_hamming,
        "perceptual_hash_size": hash_size,
        "min_frame_gap": min_frame_gap, "include_unknown": include_unknown,
        "minimum_pose_distance": minimum_pose_distance,
        "not_selected_reason_counts": dict(sorted(reason_counts.items())),
        "materialize_images": materialize_images,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
