from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .baseline_data import (
    BaselineDataError,
    CelebARecord,
    WiderImage,
    load_celeba_records,
    parse_wider_annotations,
    parse_xqlfw_pairs,
    validate_wider_images,
)
from .baseline_metrics import (
    match_main_face,
    metrics_all_and_multi,
    precision_recall_f1,
    scalable_threshold_cluster_sweep,
    scalable_threshold_clusters,
    verification_metrics,
    xqlfw_cross_validation,
)
from .baseline_wider import detector_curve, load_wider_subset, select_detector_threshold, wider_compatible_ap
from .io import write_csv, write_json


CONFIG_SCHEMA = "baseline_evaluation_config_v1"
FROZEN_SCHEMA = "baseline_frozen_parameters_v1"
RESULT_SCHEMA = "baseline_evaluation_result_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "неизвестно"
    value = max(0, int(round(seconds)))
    days, value = divmod(value, 86_400)
    hours, value = divmod(value, 3_600)
    minutes, seconds = divmod(value, 60)
    if days:
        return f"{days} д {hours:02d} ч {minutes:02d} мин"
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин {seconds:02d} с"
    return f"{seconds} с"


class ProgressReporter:
    """Живой прогресс стадии с атомарным снимком и дописываемой историей."""

    def __init__(self, run_dir: Path, stage: str, interval_seconds: float = 30.0):
        self.run_dir = run_dir.resolve()
        self.stage = stage
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.progress_dir = self.run_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.stage_started = time.monotonic()
        self.phase_started = self.stage_started
        self.last_written = 0.0
        self.phase = "инициализация"
        self.total = 0
        self._write_event({"event": "stage_started", "stage": stage, "timestamp": utc_now()})

    def _write_event(self, event: dict[str, Any], metrics: bool = False) -> None:
        path = self.progress_dir / ("intermediate_metrics.jsonl" if metrics else "events.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()

    def begin_phase(self, phase: str, total: int) -> None:
        self.phase = phase
        self.total = int(total)
        self.phase_started = time.monotonic()
        self.last_written = 0.0
        self._write_event({"event": "phase_started", "stage": self.stage, "phase": phase,
                           "total": self.total, "timestamp": utc_now()})
        self.advance(0, {"items": self.total, "cache_hits": 0, "computed": 0, "errors": 0}, force=True)

    def advance(
        self,
        processed: int,
        stats: dict[str, int],
        technical_metrics: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_written < self.interval_seconds:
            return
        elapsed = max(0.0, now - self.phase_started)
        rate = processed / elapsed if processed and elapsed > 0 else 0.0
        remaining = max(0, self.total - processed)
        eta = remaining / rate if rate > 0 else None
        snapshot = {
            "status": "running",
            "timestamp": utc_now(),
            "stage": self.stage,
            "phase": self.phase,
            "processed": int(processed),
            "total": self.total,
            "percent": 100.0 * processed / self.total if self.total else 100.0,
            "phase_elapsed_seconds": round(elapsed, 3),
            "stage_elapsed_seconds": round(now - self.stage_started, 3),
            "rate_items_per_second": rate,
            "eta_seconds": eta,
            "eta_scope": "current_phase",
            "estimated_completion": (
                datetime.fromtimestamp(time.time() + eta, timezone.utc).isoformat() if eta is not None else None
            ),
            "cache": {
                "hits": int(stats.get("cache_hits", 0)),
                "computed": int(stats.get("computed", 0)),
                "errors": int(stats.get("errors", 0)),
            },
            "technical_metrics": technical_metrics or {},
        }
        write_json(self.progress_dir / "current.json", snapshot)
        self._write_event({"event": "progress", **snapshot})
        compact_metrics = " ".join(f"{key}={value}" for key, value in (technical_metrics or {}).items())
        print(
            f"[{snapshot['timestamp']}] {self.stage}/{self.phase}: "
            f"{processed}/{self.total} ({snapshot['percent']:.1f}%), "
            f"{rate:.2f} эл/с, осталось ~{_format_duration(eta)}, "
            f"вычислено={stats.get('computed', 0)}, кэш={stats.get('cache_hits', 0)}, "
            f"ошибок={stats.get('errors', 0)}"
            + (f", {compact_metrics}" if compact_metrics else ""),
            flush=True,
        )
        self.last_written = now

    def metric(self, component: str, values: dict[str, Any]) -> None:
        event = {"event": "intermediate_metrics", "timestamp": utc_now(), "stage": self.stage,
                 "component": component, "metrics": values}
        self._write_event(event, metrics=True)
        self._write_event(event)
        print(f"[{event['timestamp']}] ПРОМЕЖУТОЧНЫЕ МЕТРИКИ {component}: "
              f"{json.dumps(values, ensure_ascii=False)}", flush=True)

    def complete(self, elapsed_seconds: float) -> None:
        snapshot = {"status": "complete", "timestamp": utc_now(), "stage": self.stage,
                    "elapsed_seconds": round(float(elapsed_seconds), 3)}
        write_json(self.progress_dir / "current.json", snapshot)
        self._write_event({"event": "stage_complete", **snapshot})
        print(f"[{snapshot['timestamp']}] Стадия {self.stage} завершена за "
              f"{_format_duration(float(elapsed_seconds))}", flush=True)


def record_stage_failure(run_dir: Path, stage: str, error: BaseException) -> None:
    progress_dir = run_dir.resolve() / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {"status": "failed", "timestamp": utc_now(), "stage": stage,
                "error_type": type(error).__name__, "error": str(error)}
    write_json(progress_dir / "current.json", snapshot)
    with (progress_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "stage_failed", **snapshot}, ensure_ascii=False,
                                separators=(",", ":")) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineDataError(f"Не удалось прочитать конфигурацию {path}: {exc}") from exc
    if config.get("config_schema_version") != CONFIG_SCHEMA:
        raise BaselineDataError(f"Ожидалась схема конфигурации {CONFIG_SCHEMA}")
    required = {"experiment_name", "seed", "paths", "model", "processing", "wider", "xqlfw", "celeba"}
    missing = sorted(required - set(config))
    if missing:
        raise BaselineDataError("В конфигурации отсутствуют поля: " + ", ".join(missing))
    nested = {
        "paths": {"data_root", "wider_train_images", "wider_train_annotations", "wider_val_images", "wider_val_annotations", "wider_eval_tools", "xqlfw_images", "xqlfw_pairs", "celeba_images", "celeba_annotations"},
        "model": {"name", "provider", "device", "detection_size", "minimum_detection_score"},
        "processing": {"batch_size", "minimum_free_gib", "limit"},
        "wider": {"iou_threshold", "confidence_thresholds", "high_precision_target"},
        "xqlfw": {"far_values"},
        "celeba": {"expected_records", "main_face_min_iou", "threshold_candidates", "cluster_block_size", "ann_neighbors"},
    }
    nested_missing = [f"{section}.{key}" for section, fields in nested.items() for key in sorted(fields - set(config[section]))]
    if nested_missing:
        raise BaselineDataError("В конфигурации отсутствуют обязательные поля: " + ", ".join(nested_missing))
    if int(config["processing"]["batch_size"]) < 1 or int(config["celeba"]["cluster_block_size"]) < 1:
        raise BaselineDataError("Размеры пакета и блока должны быть положительными")
    if not config["wider"]["confidence_thresholds"] or not config["celeba"]["threshold_candidates"]:
        raise BaselineDataError("Списки кандидатов порогов не должны быть пустыми")
    return config


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def experiment_fingerprint(config: dict[str, Any]) -> str:
    relevant = {key: config[key] for key in ("config_schema_version", "experiment_name", "seed", "model", "processing", "wider", "xqlfw", "celeba")}
    return hashlib.sha256(_canonical(relevant).encode("utf-8")).hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve(config: dict[str, Any], key: str) -> Path:
    root = Path(config["paths"]["data_root"]).expanduser()
    value = Path(config["paths"][key]).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def _limit(config: dict[str, Any], values: Sequence[Any]) -> Sequence[Any]:
    limit = config["processing"].get("limit")
    return values[: int(limit)] if limit is not None else values


def _stage_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "stages" / f"{stage}.json"


def _require_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    path = _stage_path(run_dir, stage)
    if not path.is_file():
        raise BaselineDataError(f"Сначала выполните стадию {stage}: нет {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise BaselineDataError(f"Стадия {stage} не завершена успешно")
    return value


def _ensure_run_compatible(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("experiment_fingerprint") != experiment_fingerprint(config):
        raise BaselineDataError("Конфигурация несовместима с уже подготовленным каталогом запуска")


def _save_stage(run_dir: Path, stage: str, started: float, outputs: dict[str, Any], warnings: Sequence[str] = ()) -> dict[str, Any]:
    result = {"stage": stage, "status": "complete", "completed_at": utc_now(),
              "elapsed_seconds": round(time.monotonic() - started, 3), "outputs": outputs, "warnings": list(warnings)}
    write_json(_stage_path(run_dir, stage), result)
    return result


def preflight(config: dict[str, Any], run_dir: Path, check_model: bool = True) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise BaselineDataError(f"Корень данных не найден: {data_root}")
    run_parent = run_dir.expanduser().resolve().parent
    run_parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(run_parent).free
    minimum = int(float(config["processing"].get("minimum_free_gib", 5)) * 1024 ** 3)
    if free < minimum:
        raise BaselineDataError(f"Недостаточно места: свободно {free / 1024**3:.1f} ГиБ, требуется {minimum / 1024**3:.1f} ГиБ")
    model = {"checked": False}
    if check_model:
        try:
            import cv2  # noqa: F401
            import faiss  # noqa: F401
            import insightface  # noqa: F401
            import onnxruntime as ort
        except ImportError as exc:
            raise BaselineDataError("Для запуска модели установите requirements-baseline.txt") from exc
        available = ort.get_available_providers()
        provider = config["model"]["provider"]
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise BaselineDataError(f"CUDAExecutionProvider недоступен: {available}")
        app = InsightFaceRuntime(config["model"])._app("face")
        missing = {"detection", "recognition"} - set(app.models)
        if missing:
            raise BaselineDataError("В наборе InsightFace отсутствуют компоненты: " + ", ".join(sorted(missing)))
        model = {"checked": True, "onnxruntime_providers": available,
                 "components": sorted(app.models), "model_name": config["model"]["name"]}
    return {"data_root": str(data_root), "run_dir": str(run_dir.resolve()), "free_gib": free / 1024 ** 3,
            "device": config["model"]["device"], "model": model}


class DiskCache:
    def __init__(self, root: Path, fingerprint: str):
        self.root = root
        self.fingerprint = fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, source: Path) -> Path:
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest}.npz"

    def get_or_compute(self, source: Path, compute: Callable[[], dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], bool]:
        source = source.resolve()
        stat = source.stat()
        target = self._path(source)
        if target.is_file():
            try:
                with np.load(target, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved["metadata"].item()))
                    if metadata == {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "fingerprint": self.fingerprint}:
                        return {key: saved[key].copy() for key in saved.files if key != "metadata"}, True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        value = compute()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npz.tmp")
        metadata = {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "fingerprint": self.fingerprint}
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(_canonical(metadata)), **value)
        temporary.replace(target)
        return value, False


class InsightFaceRuntime:
    def __init__(self, model_config: dict[str, Any]):
        self.config = model_config
        self.apps: dict[str, Any] = {}

    def _app(self, task: str) -> Any:
        if task in self.apps:
            return self.apps[task]
        try:
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Для извлечения признаков установите requirements-baseline.txt") from exc
        requested = self.config["provider"]
        device = int(self.config["device"])
        available = set(ort.get_available_providers())
        if requested == "cpu" or device < 0:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = device
        elif requested == "cuda":
            raise RuntimeError(f"CUDAExecutionProvider недоступен: {sorted(available)}")
        else:
            providers, ctx_id = ["CPUExecutionProvider"], -1
        modules = ["detection"] if task == "detection" else ["detection", "recognition"]
        app = FaceAnalysis(name=self.config["name"], allowed_modules=modules, providers=providers)
        app.prepare(ctx_id=ctx_id, det_thresh=float(self.config["minimum_detection_score"]),
                    det_size=(int(self.config["detection_size"]), int(self.config["detection_size"])))
        self.apps[task] = app
        return app

    def infer(self, image_path: Path, task: str) -> dict[str, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Для чтения изображений нужен OpenCV") from exc
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Не удалось прочитать изображение: {image_path}")
        faces = self._app(task).get(image)
        boxes = np.asarray([[*map(float, face.bbox), float(face.det_score)] for face in faces], dtype=np.float32).reshape(-1, 5)
        if task == "detection":
            embeddings = np.empty((len(boxes), 0), dtype=np.float32)
        else:
            rows = []
            valid_boxes = []
            for box, face in zip(boxes, faces):
                vector = np.asarray(getattr(face, "normed_embedding", None), dtype=np.float32)
                if vector.ndim == 1 and vector.size and np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-12:
                    rows.append(vector / np.linalg.norm(vector))
                    valid_boxes.append(box)
            boxes = np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 5)
            embeddings = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1) if rows else np.empty((0, 512), dtype=np.float32)
        return {"boxes": boxes, "embeddings": embeddings}


def _process_cached(
    paths: Sequence[Path],
    cache: DiskCache,
    runtime: InsightFaceRuntime,
    task: str,
    batch_size: int,
    progress: ProgressReporter | None = None,
    phase: str = "обработка",
    on_item: Callable[[int, dict[str, np.ndarray] | None], None] | None = None,
    technical_metrics: Callable[[], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, np.ndarray] | None], dict[str, int], list[str]]:
    output: list[dict[str, np.ndarray] | None] = []
    stats = {"items": len(paths), "cache_hits": 0, "computed": 0, "errors": 0}
    skipped = []
    if progress is not None:
        progress.begin_phase(phase, len(paths))
    for start in range(0, len(paths), batch_size):
        for path in paths[start:start + batch_size]:
            try:
                value, hit = cache.get_or_compute(path, lambda path=path: runtime.infer(path, task))
                stats["cache_hits" if hit else "computed"] += 1
                output.append(value)
            except (OSError, RuntimeError, ValueError) as exc:
                stats["errors"] += 1
                skipped.append(f"{path}: {exc}")
                output.append(None)
            if on_item is not None:
                on_item(len(output) - 1, output[-1])
            if progress is not None:
                progress.advance(len(output), stats, technical_metrics() if technical_metrics is not None else None)
    if progress is not None:
        progress.advance(len(output), stats, technical_metrics() if technical_metrics is not None else None, force=True)
    return output, stats, skipped


def _wider_data(config: dict[str, Any], split: str) -> tuple[list[WiderImage], Path]:
    annotation = _resolve(config, f"wider_{split}_annotations")
    images = _resolve(config, f"wider_{split}_images")
    records = parse_wider_annotations(annotation)
    expected = config["wider"].get(f"expected_{split}_images")
    if expected is not None and len(records) != int(expected):
        raise BaselineDataError(f"WIDER FACE {split}: ожидалось {expected} записей, найдено {len(records)}")
    validate_wider_images(records, images)
    return records, images


def _celeba_data(config: dict[str, Any]) -> tuple[list[CelebARecord], Path]:
    images = _resolve(config, "celeba_images")
    records = load_celeba_records(_resolve(config, "celeba_annotations"), images, int(config["celeba"].get("expected_records", 202599)))
    return records, images


def prepare(
    config: dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(run_dir, "prepare", progress_interval_seconds)
    preflight_result = preflight(config, run_dir, check_model=False)
    wider_train, _ = _wider_data(config, "train")
    wider_val, _ = _wider_data(config, "val")
    pairs = parse_xqlfw_pairs(_resolve(config, "xqlfw_pairs"), _resolve(config, "xqlfw_images"))
    celeba, _ = _celeba_data(config)
    partitions = {name: [item for item in celeba if item.partition == name] for name in ("train", "val", "test")}
    summary = {"wider": {"train_images": len(wider_train), "val_images": len(wider_val), "test_used": False},
               "xqlfw": {"pairs": len(pairs), "folds": {str(fold): sum(pair.fold == fold for pair in pairs) for fold in range(10)}},
               "celeba": {name: {"images": len(items), "identities": len({item.identity for item in items})} for name, items in partitions.items()}}
    versions = {}
    for package in ("numpy", "scipy", "opencv-python", "insightface", "onnxruntime-gpu", "faiss-cpu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {"result_schema_version": RESULT_SCHEMA, "experiment_name": config["experiment_name"],
                "experiment_fingerprint": experiment_fingerprint(config), "source_git_commit": _git_commit(repo_root),
                "seed": int(config["seed"]), "created_at": utc_now(), "host": platform.platform(), "python": platform.python_version(),
                "model": config["model"], "processing": config["processing"], "package_versions": versions,
                "preprocessing": {"color_input": "OpenCV BGR", "embedding_normalization": "L2", "celeba_main_face_rule": "maximum IoU"},
                "preflight": preflight_result}
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "dataset_summary.json", summary)
    write_json(run_dir / "run_metadata.json", metadata)
    progress.metric("dataset_summary", summary)
    result = _save_stage(run_dir, "prepare", started, {"dataset_summary": summary})
    progress.complete(float(result["elapsed_seconds"]))
    return result


def _wider_predictions(
    config: dict[str, Any],
    run_dir: Path,
    split: str,
    runtime: InsightFaceRuntime,
    progress: ProgressReporter | None = None,
) -> tuple[list[WiderImage], list[np.ndarray], dict[str, int], list[str]]:
    records, images_dir = _wider_data(config, split)
    records = list(_limit(config, records))
    cache = DiskCache(run_dir / "cache" / "wider" / split, experiment_fingerprint(config) + ":wider:detection")
    observed = {"processed": 0, "detections": 0, "images_with_detections": 0}
    def on_item(_index: int, value: dict[str, np.ndarray] | None) -> None:
        observed["processed"] += 1
        if value is not None:
            detections = len(value["boxes"])
            observed["detections"] += detections
            observed["images_with_detections"] += int(detections > 0)
    def metrics() -> dict[str, Any]:
        processed = observed["processed"]
        return {"детекций": observed["detections"],
                "изображений_с_лицом": observed["images_with_detections"],
                "доля_с_лицом": round(observed["images_with_detections"] / processed, 6) if processed else 0.0}
    values, stats, skipped = _process_cached(
        [images_dir / item.relative_path for item in records], cache, runtime, "detection",
        int(config["processing"]["batch_size"]), progress, f"wider_{split}", on_item, metrics,
    )
    predictions = [value["boxes"] if value is not None else np.empty((0, 5), dtype=np.float32) for value in values]
    return records, predictions, stats, skipped


def _celeba_embeddings(
    config: dict[str, Any],
    run_dir: Path,
    partitions: set[str],
    runtime: InsightFaceRuntime,
    progress: ProgressReporter | None = None,
) -> tuple[np.ndarray, list[int], list[str], dict[str, Any], list[str]]:
    records, images_dir = _celeba_data(config)
    selected = list(_limit(config, [item for item in records if item.partition in partitions]))
    cache = DiskCache(run_dir / "cache" / "celeba", experiment_fingerprint(config) + ":celeba:face")
    observed = {"processed": 0, "main_face_found": 0, "faces": 0}
    def on_item(index: int, value: dict[str, np.ndarray] | None) -> None:
        observed["processed"] += 1
        if value is None:
            return
        observed["faces"] += len(value["boxes"])
        observed["main_face_found"] += int(match_main_face(
            selected[index].bbox, value["boxes"], float(config["celeba"]["main_face_min_iou"])
        ) is not None)
    def metrics() -> dict[str, Any]:
        processed = observed["processed"]
        return {"найдено_главных_лиц": observed["main_face_found"],
                "текущее_покрытие": round(observed["main_face_found"] / processed, 6) if processed else 0.0,
                "всего_детекций": observed["faces"]}
    phase = "celeba_" + "+".join(sorted(partitions))
    values, stats, skipped = _process_cached(
        [images_dir / item.filename for item in selected], cache, runtime, "face",
        int(config["processing"]["batch_size"]), progress, phase, on_item, metrics,
    )
    vectors, labels, names = [], [], []
    missing_main = 0
    for record, value in zip(selected, values):
        if value is None:
            continue
        match = match_main_face(record.bbox, value["boxes"], float(config["celeba"]["main_face_min_iou"]))
        if match is None:
            missing_main += 1
            continue
        vectors.append(value["embeddings"][match])
        labels.append(record.identity)
        names.append(record.filename)
    matrix = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), -1) if vectors else np.empty((0, 512), dtype=np.float32)
    coverage = {**stats, "selected_images": len(selected), "main_face_not_found": missing_main,
                "clustered_images": len(vectors), "detector_coverage": (len(selected) - missing_main) / len(selected) if selected else 0.0,
                "clustering_coverage": len(vectors) / len(selected) if selected else 0.0}
    return matrix, labels, names, coverage, skipped


def _celeba_threshold_results(vectors: np.ndarray, labels: Sequence[int], config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = list(map(float, config["celeba"]["threshold_candidates"]))
    sweep = scalable_threshold_cluster_sweep(vectors, thresholds, int(config["celeba"]["cluster_block_size"]),
                                             int(config["celeba"]["ann_neighbors"]), int(config["seed"]))
    return [{"threshold": threshold, "metrics": metrics_all_and_multi(labels, sweep[threshold].tolist())} for threshold in thresholds]


def _select_celeba(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row["metrics"]["identities_with_at_least_two_images"]["bcubed_f1"]),
                                      float(row["metrics"]["all_identities"]["pair_f1"]), float(row["threshold"])))


def _xqlfw_scores(
    config: dict[str, Any],
    run_dir: Path,
    runtime: InsightFaceRuntime,
    progress: ProgressReporter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    pairs = parse_xqlfw_pairs(_resolve(config, "xqlfw_pairs"), _resolve(config, "xqlfw_images"))
    paths = sorted({path.resolve() for pair in pairs for path in (pair.left, pair.right)}, key=str)
    cache = DiskCache(run_dir / "cache" / "xqlfw", experiment_fingerprint(config) + ":xqlfw:face")
    observed = {"processed": 0, "usable": 0}
    def on_item(_index: int, value: dict[str, np.ndarray] | None) -> None:
        observed["processed"] += 1
        observed["usable"] += int(value is not None and len(value["embeddings"]) > 0)
    def metrics() -> dict[str, Any]:
        processed = observed["processed"]
        return {"пригодных_изображений": observed["usable"],
                "текущее_покрытие": round(observed["usable"] / processed, 6) if processed else 0.0}
    values, stats, skipped = _process_cached(
        paths, cache, runtime, "face", int(config["processing"]["batch_size"]),
        progress, "xqlfw", on_item, metrics,
    )
    embeddings: dict[Path, np.ndarray] = {}
    for path, value in zip(paths, values):
        if value is None or len(value["embeddings"]) == 0:
            continue
        areas = (value["boxes"][:, 2] - value["boxes"][:, 0]) * (value["boxes"][:, 3] - value["boxes"][:, 1])
        embeddings[path] = value["embeddings"][int(np.argmax(areas))]
    missing = [pair for pair in pairs if pair.left.resolve() not in embeddings or pair.right.resolve() not in embeddings]
    if missing:
        raise BaselineDataError(f"XQLFW: для {len(missing)} пар не удалось получить оба эмбеддинга; исправьте пропуски перед протокольной оценкой")
    scores = np.asarray([float(embeddings[pair.left.resolve()] @ embeddings[pair.right.resolve()]) for pair in pairs])
    labels = np.asarray([pair.same for pair in pairs], dtype=bool)
    folds = np.asarray([pair.fold for pair in pairs], dtype=np.int64)
    return scores, labels, folds, {**stats, "usable_images": len(embeddings), "pairs": len(pairs)}, skipped


def calibrate(config: dict[str, Any], run_dir: Path, progress_interval_seconds: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    progress = ProgressReporter(run_dir, "calibrate", progress_interval_seconds)
    _require_stage(run_dir, "prepare")
    _ensure_run_compatible(run_dir, config)
    np.random.seed(int(config["seed"]))
    runtime = InsightFaceRuntime(config["model"])
    wider_records, predictions, wider_cache, warnings = _wider_predictions(config, run_dir, "train", runtime, progress)
    curve = detector_curve(predictions, [np.asarray(item.boxes) for item in wider_records], config["wider"]["confidence_thresholds"], float(config["wider"]["iou_threshold"]))
    operating = select_detector_threshold(curve)
    high_precision = select_detector_threshold(curve, float(config["wider"]["high_precision_target"]))
    if operating is None:
        raise BaselineDataError("WIDER train: невозможно выбрать рабочий порог")
    write_csv(run_dir / "calibration" / "wider_threshold_curve.csv", curve)
    progress.metric("wider_train", {"operating": operating, "high_precision": high_precision,
                                     "coverage": wider_cache})
    vectors, labels, _, celeba_coverage, celeba_warnings = _celeba_embeddings(
        config, run_dir, {"train"}, runtime, progress
    )
    celeba_rows = _celeba_threshold_results(vectors, labels, config)
    write_json(run_dir / "calibration" / "celeba_train_candidates.json", celeba_rows)
    progress.metric("celeba_train", {"best_candidate": _select_celeba(celeba_rows),
                                      "coverage": celeba_coverage})
    scores, pair_labels, folds, xqlfw_cache, xqlfw_warnings = _xqlfw_scores(config, run_dir, runtime, progress)
    np.savez_compressed(run_dir / "calibration" / "xqlfw_scores.npz", scores=scores, labels=pair_labels, folds=folds)
    progress.metric("xqlfw_scores", {"pairs": len(scores), "cache": xqlfw_cache,
                                      "mean_same_similarity": float(np.mean(scores[pair_labels])),
                                      "mean_different_similarity": float(np.mean(scores[~pair_labels]))})
    result = {"wider": {"split": "train", "operating": operating, "high_precision": high_precision, "cache": wider_cache},
              "celeba": {"split": "train", "candidates": celeba_rows, "coverage": celeba_coverage},
              "xqlfw": {"prepared_scores": len(scores), "cache": xqlfw_cache}}
    write_json(run_dir / "calibration" / "summary.json", result)
    stage = _save_stage(run_dir, "calibrate", started, result, [*warnings, *celeba_warnings, *xqlfw_warnings])
    progress.complete(float(stage["elapsed_seconds"]))
    return stage


def validate(config: dict[str, Any], run_dir: Path, progress_interval_seconds: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    progress = ProgressReporter(run_dir, "validate", progress_interval_seconds)
    _require_stage(run_dir, "calibrate")
    _ensure_run_compatible(run_dir, config)
    runtime = InsightFaceRuntime(config["model"])
    vectors, labels, _, coverage, warnings = _celeba_embeddings(config, run_dir, {"val"}, runtime, progress)
    candidates = _celeba_threshold_results(vectors, labels, config)
    selected = _select_celeba(candidates)
    with np.load(run_dir / "calibration" / "xqlfw_scores.npz", allow_pickle=False) as saved:
        xqlfw = xqlfw_cross_validation(saved["scores"], saved["labels"], saved["folds"], config["xqlfw"]["far_values"])
    progress.metric("celeba_val", {"selected_configuration": selected, "coverage": coverage})
    progress.metric("xqlfw_cross_validation", {"accuracy_mean": xqlfw["accuracy_mean"],
                                                "accuracy_std": xqlfw["accuracy_std"],
                                                "roc_auc": xqlfw["roc_auc"], "eer": xqlfw["eer"],
                                                "verification_threshold": xqlfw["verification_threshold"]})
    result = {"celeba": {"split": "val", "candidates": candidates, "selected_configuration": selected, "coverage": coverage},
              "xqlfw": xqlfw}
    write_json(run_dir / "validation" / "summary.json", result)
    stage = _save_stage(run_dir, "validate", started, result, [*warnings, *xqlfw.get("warnings", [])])
    progress.complete(float(stage["elapsed_seconds"]))
    return stage


def freeze(config: dict[str, Any], run_dir: Path, progress_interval_seconds: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    progress = ProgressReporter(run_dir, "freeze", progress_interval_seconds)
    calibration = _require_stage(run_dir, "calibrate")["outputs"]
    validation = _require_stage(run_dir, "validate")["outputs"]
    _ensure_run_compatible(run_dir, config)
    runtime = InsightFaceRuntime(config["model"])
    vectors, labels, _, coverage, warnings = _celeba_embeddings(config, run_dir, {"train", "val"}, runtime, progress)
    recalibrated = _select_celeba(_celeba_threshold_results(vectors, labels, config))
    frozen = {"frozen_schema_version": FROZEN_SCHEMA, "created_at": utc_now(),
              "experiment_name": config["experiment_name"], "experiment_fingerprint": experiment_fingerprint(config),
              "data_protocol": {"wider_calibration": "train", "wider_evaluation": "val", "xqlfw": "10 folds, 9 calibration + 1 evaluation",
                                "celeba_calibration": "train+val after selection on val", "celeba_evaluation": "test"},
              "wider": {"confidence_threshold": calibration["wider"]["operating"]["threshold"],
                        "high_precision_threshold": calibration["wider"]["high_precision"]["threshold"] if calibration["wider"]["high_precision"] else None,
                        "iou_threshold": float(config["wider"]["iou_threshold"])},
              "xqlfw": {"verification_threshold": validation["xqlfw"]["verification_threshold"]},
              "celeba": {"clustering_threshold": recalibrated["threshold"], "main_face_min_iou": float(config["celeba"]["main_face_min_iou"]),
                         "algorithm": "threshold_graph_hnsw", "ann_neighbors": int(config["celeba"]["ann_neighbors"])},
              "model": config["model"], "recalibration_coverage": coverage}
    validate_frozen(frozen, config)
    write_json(run_dir / "frozen_parameters.json", frozen)
    progress.metric("frozen_parameters", {"wider": frozen["wider"], "xqlfw": frozen["xqlfw"],
                                           "celeba": frozen["celeba"], "coverage": coverage})
    stage = _save_stage(run_dir, "freeze", started, {"frozen_parameters": frozen}, warnings)
    progress.complete(float(stage["elapsed_seconds"]))
    return stage


def validate_frozen(frozen: dict[str, Any], config: dict[str, Any]) -> None:
    if frozen.get("frozen_schema_version") != FROZEN_SCHEMA:
        raise BaselineDataError(f"Несовместимая схема замороженных параметров; ожидается {FROZEN_SCHEMA}")
    if frozen.get("experiment_fingerprint") != experiment_fingerprint(config):
        raise BaselineDataError("Замороженные параметры несовместимы с конфигурацией эксперимента")
    required = (("wider", "confidence_threshold"), ("wider", "iou_threshold"),
                ("xqlfw", "verification_threshold"), ("celeba", "clustering_threshold"), ("celeba", "main_face_min_iou"))
    missing = [".".join(path) for path in required if frozen.get(path[0], {}).get(path[1]) is None]
    if missing:
        raise BaselineDataError("В замороженных параметрах отсутствуют обязательные поля: " + ", ".join(missing))


def evaluate(
    config: dict[str, Any],
    run_dir: Path,
    frozen_path: Path,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Итоговая оценка. Эта функция принципиально не вызывает подбор порогов."""
    started = time.monotonic()
    progress = ProgressReporter(run_dir, "evaluate", progress_interval_seconds)
    _require_stage(run_dir, "prepare")
    _ensure_run_compatible(run_dir, config)
    if not frozen_path.is_file():
        raise BaselineDataError(f"Файл замороженных параметров не найден: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_frozen(frozen, config)
    runtime = InsightFaceRuntime(config["model"])
    wider_records, predictions, wider_cache, warnings = _wider_predictions(config, run_dir, "val", runtime, progress)
    threshold = float(frozen["wider"]["confidence_threshold"])
    curve_at_frozen = detector_curve(predictions, [np.asarray(item.boxes) for item in wider_records], [threshold], float(frozen["wider"]["iou_threshold"]))[0]
    eval_dir = _resolve(config, "wider_eval_tools")
    if not (eval_dir / "wider_face_val.mat").is_file():
        matches = list(eval_dir.rglob("wider_face_val.mat")) if eval_dir.is_dir() else []
        if len(matches) != 1:
            raise BaselineDataError(f"Не найден однозначный каталог официальной WIDER-разметки MAT внутри {eval_dir}")
        eval_dir = matches[0].parent
    ap = {}
    for subset in ("easy", "medium", "hard"):
        keep = load_wider_subset(eval_dir, subset)
        ap[subset] = wider_compatible_ap([item.relative_path for item in wider_records], predictions,
                                         [np.asarray(item.boxes) for item in wider_records], keep,
                                         float(frozen["wider"]["iou_threshold"]))
    progress.metric("wider_val", {"ap_easy": ap["easy"], "ap_medium": ap["medium"],
                                   "ap_hard": ap["hard"], "operating_point": curve_at_frozen,
                                   "coverage": wider_cache})
    vectors, labels, _, celeba_coverage, celeba_warnings = _celeba_embeddings(
        config, run_dir, {"test"}, runtime, progress
    )
    predicted = scalable_threshold_clusters(vectors, float(frozen["celeba"]["clustering_threshold"]),
                                            int(config["celeba"]["cluster_block_size"]), int(frozen["celeba"]["ann_neighbors"]), int(config["seed"]))
    celeba_metrics = metrics_all_and_multi(labels, predicted.tolist())
    progress.metric("celeba_test", {"metrics": celeba_metrics, "coverage": celeba_coverage,
                                     "predicted_clusters": int(len(np.unique(predicted)))})
    scores, pair_labels, _, xqlfw_cache, xqlfw_warnings = _xqlfw_scores(config, run_dir, runtime, progress)
    verification_threshold = float(frozen["xqlfw"]["verification_threshold"])
    xqlfw_metrics = {"frozen_threshold": verification_threshold,
                     "classification": precision_recall_f1(pair_labels, scores >= verification_threshold),
                     **verification_metrics(scores, pair_labels, config["xqlfw"]["far_values"])}
    progress.metric("xqlfw_frozen", {"frozen_threshold": verification_threshold,
                                      "classification": xqlfw_metrics["classification"],
                                      "roc_auc": xqlfw_metrics["roc_auc"], "eer": xqlfw_metrics["eer"],
                                      "tar_at_far": xqlfw_metrics["tar_at_far"]})
    result = {"result_schema_version": RESULT_SCHEMA,
              "wider": {"split": "val", "ap_protocol": "strictly compatible with official widerface_evaluate 1000-point protocol",
                        "ap_easy": ap["easy"], "ap_medium": ap["medium"], "ap_hard": ap["hard"],
                        "operating_point": curve_at_frozen, "coverage": {**wider_cache, "processed_without_error": wider_cache["items"] - wider_cache["errors"],
                                                                         "coverage": (wider_cache["items"] - wider_cache["errors"]) / max(wider_cache["items"], 1)}},
              "xqlfw": {**xqlfw_metrics, "cache": xqlfw_cache},
              "celeba": {"split": "test", "metrics": celeba_metrics, "coverage": celeba_coverage,
                         "predicted_clusters": int(len(np.unique(predicted)))}}
    all_warnings = [*warnings, *celeba_warnings, *xqlfw_warnings, *xqlfw_metrics.get("warnings", [])]
    write_json(run_dir / "evaluation" / "metrics.json", result)
    write_json(run_dir / "evaluation" / "warnings.json", all_warnings)
    report = ["# Итоговая оценка baseline", "", f"Создано: {utc_now()}", "",
              f"WIDER FACE val: AP Easy={ap['easy']:.6f}, Medium={ap['medium']:.6f}, Hard={ap['hard']:.6f}",
              f"WIDER F1 при замороженном пороге: {float(curve_at_frozen['f1']):.6f}",
              f"XQLFW ROC AUC={float(xqlfw_metrics['roc_auc']):.6f}, EER={float(xqlfw_metrics['eer']):.6f}",
              f"CelebA test B-cubed F1={float(celeba_metrics['all_identities']['bcubed_f1']):.6f}", "",
              "Подробные значения находятся в metrics.json."]
    (run_dir / "evaluation" / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    stage = _save_stage(run_dir, "evaluate", started, result, all_warnings)
    progress.complete(float(stage["elapsed_seconds"]))
    return stage
