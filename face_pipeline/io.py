from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REQUIRED_FACE_FIELDS = (
    "face_id",
    "source_frame",
    "video_name",
    "frame_number",
    "face_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "face_width",
    "face_height",
    "det_score",
    "blur_score",
    "aligned_path",
    "embedding_index",
)


class DataValidationError(ValueError):
    """Входные данные не соответствуют контракту конвейера."""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataValidationError(f"CSV-файл не найден: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"В CSV нет заголовка: {path}")
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, array)
    temp.replace(path)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise DataValidationError("Найден нулевой эмбеддинг")
    return vectors / norms


def load_face_run(run_dir: Path, require_images: bool = False) -> tuple[list[dict[str, str]], np.ndarray]:
    run_dir = run_dir.resolve()
    rows = read_csv(run_dir / "faces.csv")
    if not rows:
        raise DataValidationError(f"В {run_dir / 'faces.csv'} нет записей")
    missing = [field for field in REQUIRED_FACE_FIELDS if field not in rows[0]]
    if missing:
        raise DataValidationError("В faces.csv отсутствуют поля: " + ", ".join(missing))

    embeddings_path = run_dir / "embeddings.npy"
    if not embeddings_path.is_file():
        raise DataValidationError(f"Массив эмбеддингов не найден: {embeddings_path}")
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.ndim != 2:
        raise DataValidationError(f"Ожидалась матрица эмбеддингов, получена форма {embeddings.shape}")
    if embeddings.shape[0] != len(rows):
        raise DataValidationError(
            f"Число строк CSV ({len(rows)}) не равно числу эмбеддингов ({embeddings.shape[0]})"
        )
    if not np.isfinite(embeddings).all():
        raise DataValidationError("В эмбеддингах есть NaN или бесконечные значения")

    face_ids = [row["face_id"] for row in rows]
    if len(face_ids) != len(set(face_ids)):
        raise DataValidationError("face_id должны быть уникальны")
    indices = [as_int(row, "embedding_index", -1) for row in rows]
    if sorted(indices) != list(range(len(rows))):
        raise DataValidationError("embedding_index должны без повторов покрывать диапазон 0..N-1")
    ordered = np.asarray(embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(ordered, axis=1)
    if np.max(np.abs(norms - 1.0)) > 5e-3:
        raise DataValidationError(
            f"Эмбеддинги не L2-нормализованы: нормы от {norms.min():.6f} до {norms.max():.6f}"
        )

    for row in rows:
        if as_int(row, "frame_number", -1) < 0:
            raise DataValidationError(f"Некорректный frame_number у {row['face_id']}")
        x1, y1 = as_float(row, "bbox_x1"), as_float(row, "bbox_y1")
        x2, y2 = as_float(row, "bbox_x2"), as_float(row, "bbox_y2")
        if not (x2 > x1 and y2 > y1):
            raise DataValidationError(f"Некорректная рамка у {row['face_id']}")
        if require_images and resolve_path(row["aligned_path"], run_dir) is None:
            raise DataValidationError(f"Не найден кроп лица {row['face_id']}: {row['aligned_path']}")
    return rows, ordered


def resolve_path(value: str, run_dir: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [run_dir / path, run_dir.parent / path, Path.cwd() / path]
        # Старый process_faces мог записать путь вида
        # runs/<имя запуска>/aligned_faces/x.jpg. Такой путь должен работать
        # даже при запуске команды не из корня проекта.
        if run_dir.name in path.parts:
            position = path.parts.index(run_dir.name)
            suffix = Path(*path.parts[position + 1:])
            candidates.insert(0, run_dir / suffix)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        as_float(row, "bbox_x1", 0.0),
        as_float(row, "bbox_y1", 0.0),
        as_float(row, "bbox_x2", 0.0),
        as_float(row, "bbox_y2", 0.0),
    )


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Каталог уже существует: {path}. Для замены укажите --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
