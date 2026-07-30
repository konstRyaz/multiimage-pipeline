from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .clustering import run_clustering
from .experiment_metrics import aggregate_metrics, load_labels
from .io import as_int, read_csv, resolve_path, save_npy, write_csv, write_json
from .policy import apply_policy_rows, soft_ratings
from .quality_features import FEATURE_SCHEMA_VERSION, load_or_build_feature_cache
from .selection import run_selection
from .tracking import run_tracking


EXPERIMENT_RESULT_VERSION = "experiment_result_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{_safe_name(name)}_{uuid.uuid4().hex[:8]}"


def _prepare_stage(
    stage_dir: Path,
    source_run: Path,
    source_rows: list[dict[str, str]],
    source_embeddings: np.ndarray,
    accepted_ids: set[str],
    rating_by_face: dict[str, float],
) -> None:
    stage_dir.mkdir(parents=True)
    staged_rows: list[dict[str, Any]] = []
    staged_vectors: list[np.ndarray] = []
    for row in source_rows:
        if row["face_id"] not in accepted_ids:
            continue
        source_index = as_int(row, "embedding_index", -1)
        staged = dict(row)
        staged["source_embedding_index"] = source_index
        staged["embedding_index"] = len(staged_vectors)
        staged["quality_score_override"] = rating_by_face[row["face_id"]]
        aligned = resolve_path(row.get("aligned_path", ""), source_run)
        source = resolve_path(row.get("source_frame", ""), source_run)
        if aligned is not None:
            staged["aligned_path"] = str(aligned)
        if source is not None:
            staged["source_frame"] = str(source)
        staged_rows.append(staged)
        staged_vectors.append(source_embeddings[source_index])
    write_csv(stage_dir / "faces.csv", staged_rows)
    matrix = (
        np.stack(staged_vectors).astype(np.float32)
        if staged_vectors
        else np.empty((0, source_embeddings.shape[1]), dtype=np.float32)
    )
    save_npy(stage_dir / "embeddings.npy", matrix)
    source_config = source_run / "run_config.json"
    if source_config.is_file():
        shutil.copy2(source_config, stage_dir / "source_run_config.json")


def _features_by_face(features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["face_id"]: row for row in features}


def _empty_stage_outputs(stage_dir: Path) -> None:
    for name in ("tracking", "clustering", "selected"):
        (stage_dir / name).mkdir(parents=True, exist_ok=True)
    write_csv(stage_dir / "tracking" / "faces_tracked.csv", [], ["face_id"])
    write_csv(stage_dir / "tracking" / "tracks.csv", [], ["track_id"])
    save_npy(stage_dir / "tracking" / "track_embeddings.npy", np.empty((0, 0), dtype=np.float32))
    write_json(stage_dir / "tracking" / "summary.json", {"faces": 0, "tracks": 0})
    write_csv(stage_dir / "clustering" / "faces_clustered.csv", [], ["face_id", "cluster_id"])
    write_csv(stage_dir / "clustering" / "clusters.csv", [], ["cluster_id"])
    save_npy(stage_dir / "clustering" / "cluster_embeddings.npy", np.empty((0, 0), dtype=np.float32))
    write_json(stage_dir / "clustering" / "summary.json", {"tracks": 0, "clusters": 0})
    write_csv(stage_dir / "selected" / "selection.csv", [], ["face_id", "cluster_id"])
    write_csv(
        stage_dir / "selected" / "selection_decisions.csv",
        [],
        ["cluster_id", "face_id", "selected", "selection_decision", "not_selected_reasons"],
    )
    write_json(stage_dir / "selected" / "summary.json", {"identities": 0, "selected_faces": 0})


def _ranking_results(
    accepted_policy_rows: list[dict[str, Any]],
    rating_rows: list[dict[str, float]],
    clustered_rows: list[dict[str, str]],
    selection_rows: list[dict[str, str]],
    decision_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    cluster_by_face = {row["face_id"]: row.get("cluster_id", "") for row in clustered_rows}
    selected = {row["face_id"] for row in selection_rows}
    decisions = {row["face_id"]: row for row in decision_rows}
    output: list[dict[str, Any]] = []
    rating_by_face = {row["face_id"]: rating for row, rating in zip(accepted_policy_rows, rating_rows)}
    for policy in accepted_policy_rows:
        face_id = policy["face_id"]
        rating = rating_by_face.get(face_id)
        if face_id in selected:
            selection_decision = "selected"
            reasons = []
        elif face_id in decisions:
            selection_decision = "not_selected"
            value = decisions[face_id].get("not_selected_reasons", "")
            reasons = [item for item in value.split(";") if item]
        else:
            selection_decision = "not_selected"
            reasons = ["unknown_identity_excluded" if cluster_by_face.get(face_id, "").startswith("unknown_")
                       else "not_selected_by_pipeline"]
        output.append({
            "face_id": face_id,
            "embedding_index": policy["embedding_index"],
            "feature_schema_version": policy["feature_schema_version"],
            "policy_version": policy["policy_version"],
            "ranking_version": "",
            "accepted": True,
            "reject_reasons": policy["reject_reasons"],
            "shadow_reasons": policy["shadow_reasons"],
            **(rating or {}),
            "cluster_id": cluster_by_face.get(face_id),
            "selection_decision": selection_decision,
            "not_selected_reasons": reasons,
        })
    return output


def run_experiment(
    config: dict[str, Any],
    repo_root: Path,
    force_features: bool = False,
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    run_id = _run_id(config["experiment_name"])
    source_run = Path(config["paths"]["run_dir"]).resolve()
    experiments_dir = Path(config["paths"]["experiments_dir"]).resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = experiments_dir / f".{run_id}.running"
    final_dir = experiments_dir / run_id
    temp_dir.mkdir()
    metadata: dict[str, Any] = {
        "result_schema_version": EXPERIMENT_RESULT_VERSION,
        "run_id": run_id,
        "experiment_name": config["experiment_name"],
        "status": "running",
        "started_at": _utc_now(),
        "source_git_commit": _git_commit(repo_root),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "policy_version": config["policy"]["version"],
        "ranking_version": config["ranking"]["version"],
        "source_run_dir": str(source_run),
        "warnings": [],
        "errors": [],
    }
    write_json(temp_dir / "run_metadata.json", metadata)
    write_json(temp_dir / "resolved_config.json", config)

    try:
        timings: dict[str, float] = {}
        moment = time.perf_counter()
        features, cache_info = load_or_build_feature_cache(
            source_run,
            config["feature_extraction"],
            force_features,
        )
        timings["feature_extraction"] = time.perf_counter() - moment
        if cache_info.get("warning"):
            metadata["warnings"].append(cache_info["warning"])

        moment = time.perf_counter()
        policy_rows = apply_policy_rows(features, config["policy"])
        feature_by_face = _features_by_face(features)
        accepted_policy_rows = [row for row in policy_rows if row["accepted"]]
        accepted_features = [feature_by_face[row["face_id"]] for row in accepted_policy_rows]
        rating_rows, rating_context = soft_ratings(accepted_features, config["ranking"])
        timings["policy"] = time.perf_counter() - moment
        _write_jsonl(temp_dir / "policy_results.jsonl", policy_rows)
        write_csv(
            temp_dir / "policy_results.csv",
            [
                {
                    **row,
                    "reject_reasons": ";".join(row["reject_reasons"]),
                    "shadow_reasons": ";".join(row["shadow_reasons"]),
                }
                for row in policy_rows
            ],
        )

        source_rows = read_csv(source_run / "faces.csv")
        source_embeddings = np.load(source_run / "embeddings.npy", allow_pickle=False)
        accepted_ids = {row["face_id"] for row in accepted_policy_rows}
        rating_by_face = {
            row["face_id"]: rating["soft_rating"]
            for row, rating in zip(accepted_policy_rows, rating_rows)
        }
        stage_dir = temp_dir / "pipeline"
        _prepare_stage(
            stage_dir,
            source_run,
            source_rows,
            source_embeddings,
            accepted_ids,
            rating_by_face,
        )

        tracking_summary: dict[str, Any]
        clustering_summary: dict[str, Any]
        selection_summary: dict[str, Any]
        if accepted_ids:
            tracking_config = config["tracking"]
            moment = time.perf_counter()
            tracking_summary = run_tracking(
                stage_dir,
                overwrite=False,
                max_frame_gap=int(tracking_config["max_frame_gap"]),
                min_similarity=float(tracking_config["min_similarity"]),
                appearance_weight=float(tracking_config["appearance_weight"]),
                quality_weight_offset=float(tracking_config["quality_weight_offset"]),
            )
            timings["tracking"] = time.perf_counter() - moment

            clustering_config = config["clustering"]
            moment = time.perf_counter()
            clustering_summary = run_clustering(
                stage_dir,
                overwrite=False,
                threshold=float(clustering_config["threshold"]),
                min_cluster_faces=int(clustering_config["min_cluster_faces"]),
                corrections_path=(
                    Path(clustering_config["corrections"])
                    if clustering_config.get("corrections")
                    else None
                ),
            )
            timings["clustering"] = time.perf_counter() - moment

            selection_config = config["selection"]
            moment = time.perf_counter()
            selection_summary = run_selection(
                stage_dir,
                overwrite=False,
                max_per_identity=int(selection_config["max_per_identity"]),
                duplicate_hamming=int(selection_config["duplicate_hamming_threshold"]),
                min_frame_gap=int(selection_config["min_frame_gap"]),
                source_margin=float(selection_config["source_margin"]),
                include_unknown=bool(selection_config["include_unknown"]),
                hash_size=int(selection_config["perceptual_hash_size"]),
                diversity_weights={
                    key: float(value)
                    for key, value in selection_config["diversity_weights"].items()
                },
                pose_scale=float(selection_config["pose_scale"]),
                time_scale=float(selection_config["time_scale"]),
                cross_video_time_distance=float(selection_config["cross_video_time_distance"]),
                minimum_pose_distance=float(selection_config["minimum_pose_distance"]),
                materialize_images=bool(selection_config["materialize_images"]),
            )
            timings["selection"] = time.perf_counter() - moment
        else:
            _empty_stage_outputs(stage_dir)
            tracking_summary = {"faces": 0, "tracks": 0}
            clustering_summary = {"tracks": 0, "clusters": 0}
            selection_summary = {"identities": 0, "selected_faces": 0}
            timings.update({"tracking": 0.0, "clustering": 0.0, "selection": 0.0})

        clustered_rows = read_csv(stage_dir / "clustering" / "faces_clustered.csv")
        selection_rows = read_csv(stage_dir / "selected" / "selection.csv")
        decision_rows = read_csv(stage_dir / "selected" / "selection_decisions.csv")
        ranking_results = _ranking_results(
            accepted_policy_rows, rating_rows, clustered_rows, selection_rows, decision_rows
        )
        for row in ranking_results:
            row["ranking_version"] = config["ranking"]["version"]
        _write_jsonl(temp_dir / "ranking_results.jsonl", ranking_results)

        labels_config = config["labels"]
        labels, identities, label_warnings = load_labels(
            Path(config["paths"]["labels"]) if config["paths"].get("labels") else None,
            labels_config["face_id_column"],
            labels_config["positive_column"],
            list(labels_config["positive_values"]),
            labels_config.get("identity_column"),
        )
        metadata["warnings"].extend(label_warnings)
        if not identities:
            metadata["warnings"].append(
                "Нет внешних меток личности: статистика до фильтрации ограничена "
                "личностями, сохранившимися после фильтрации."
            )
        timings["total"] = time.perf_counter() - started_wall
        metrics = aggregate_metrics(
            policy_rows,
            clustered_rows,
            selection_rows,
            ranking_results,
            timings,
            cache_info,
            labels,
            identities,
        )
        write_json(temp_dir / "metrics.json", metrics)
        write_json(temp_dir / "stage_summaries.json", {
            "tracking": tracking_summary,
            "clustering": clustering_summary,
            "selection": selection_summary,
            "rating_context": rating_context,
        })
        write_json(temp_dir / "cache_info.json", cache_info)
        metadata["status"] = "complete"
        metadata["completed_at"] = _utc_now()
        write_json(temp_dir / "run_metadata.json", metadata)
        temp_dir.replace(final_dir)
        return {
            "run_id": run_id,
            "run_dir": str(final_dir),
            "metadata": metadata,
            "metrics": metrics,
        }
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["completed_at"] = _utc_now()
        metadata["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        write_json(temp_dir / "run_metadata.json", metadata)
        write_json(temp_dir / "error.json", {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        failed_dir = experiments_dir / f"{run_id}.failed"
        temp_dir.replace(failed_dir)
        raise


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_numbers(item, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def compare_experiments(
    configs: Sequence[dict[str, Any]],
    repo_root: Path,
    force_features: bool = False,
) -> dict[str, Any]:
    if len(configs) < 2:
        raise ValueError("Для сравнения нужны как минимум две конфигурации")
    results = [
        run_experiment(config, repo_root, force_features=force_features and index == 0)
        for index, config in enumerate(configs)
    ]
    baseline = results[0]
    base_numbers = _flatten_numbers(baseline["metrics"])
    deltas: dict[str, dict[str, float]] = {}
    for result in results[1:]:
        current = _flatten_numbers(result["metrics"])
        deltas[result["metadata"]["experiment_name"]] = {
            key: round(current[key] - base_numbers[key], 8)
            for key in sorted(set(base_numbers) & set(current))
        }
    comparison_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_comparison_"
        + uuid.uuid4().hex[:8]
    )
    experiments_dir = Path(configs[0]["paths"]["experiments_dir"])
    comparison_dir = experiments_dir / "comparisons" / comparison_id
    comparison_dir.mkdir(parents=True)
    comparison = {
        "comparison_schema_version": "experiment_comparison_v1",
        "comparison_id": comparison_id,
        "created_at": _utc_now(),
        "baseline": baseline["metadata"]["experiment_name"],
        "runs": [
            {
                "experiment_name": result["metadata"]["experiment_name"],
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
            }
            for result in results
        ],
        "absolute": {
            result["metadata"]["experiment_name"]: result["metrics"] for result in results
        },
        "deltas_from_baseline": deltas,
    }
    write_json(comparison_dir / "comparison.json", comparison)
    return {**comparison, "comparison_dir": str(comparison_dir)}
