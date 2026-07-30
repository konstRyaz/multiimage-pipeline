#!/usr/bin/env python3
"""Детекция, выравнивание и извлечение эмбеддингов лиц через InsightFace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.io import prepare_output_dir, save_npy, write_csv, write_json


FACE_FIELDS = [
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
    "frame_width",
    "frame_height",
    "border_margin",
    "det_score",
    "pitch",
    "yaw",
    "roll",
    "blur_score",
    "aligned_path",
    "embedding_index",
    "cluster_id",
    "rejection_reason",
]


FRAME_RE = re.compile(r"^(?P<video>.+?)_frame_(?P<number>\d+)$")


def parse_frame(path: Path) -> tuple[str, int]:
    match = FRAME_RE.match(path.stem)
    if match:
        return match.group("video"), int(match.group("number"))
    numbers = re.findall(r"\d+", path.stem)
    return path.stem, int(numbers[-1]) if numbers else 0


def safe_id(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", text).strip("_")
    return value[-80:] or "video"


def parse_source(path: Path, frames_dir: Path, input_type: str) -> tuple[str, int]:
    """Вернуть уникальную группу источника и номер кадра.

    В режиме video-frames сохраняется прежняя логика имени ролика и номера
    кадра. В режиме photos каждая исходная фотография становится отдельной
    группой. Хеш относительного пути не даёт столкнуться одинаковым именам
    вроде 2926/photo_1.jpg и 6210/photo_1.jpg.
    """
    if input_type == "video-frames":
        return parse_frame(path)
    relative = path.relative_to(frames_dir)
    relative_without_suffix = relative.with_suffix("").as_posix()
    digest = hashlib.blake2s(
        relative.as_posix().encode("utf-8"), digest_size=5
    ).hexdigest()
    return f"{safe_id(relative_without_suffix)}_{digest}", 0


def choose_providers(ort: Any, device: int, provider: str) -> list[str]:
    available = set(ort.get_available_providers())
    if provider == "cpu" or device < 0:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "cuda":
        raise RuntimeError(
            "Запрошен CUDAExecutionProvider, но он недоступен. "
            f"ONNX Runtime сообщает: {sorted(available)}"
        )
    print("Предупреждение: CUDA недоступна, используется CPU", file=sys.stderr)
    return ["CPUExecutionProvider"]


def pose_values(face: Any) -> tuple[float | str, float | str, float | str]:
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 3:
        return "", "", ""
    return tuple(float(value) for value in pose[:3])  # type: ignore[return-value]


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
    except ImportError as exc:
        raise SystemExit(
            "Для process_faces.py нужны InsightFace, OpenCV и ONNX Runtime. "
            "Установите зависимости из requirements-insightface.txt"
        ) from exc

    frames_dir = args.frames_dir.expanduser().resolve()
    frames = sorted(frames_dir.glob(args.pattern))
    if args.sample_step > 1:
        frames = frames[:: args.sample_step]
    if args.limit is not None:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"Кадры не найдены: {frames_dir / args.pattern}")

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(output_dir, args.overwrite)
    aligned_dir = output_dir / "aligned_faces"
    aligned_dir.mkdir()

    providers = choose_providers(ort, args.device, args.provider)
    app = FaceAnalysis(
        name=args.model_name,
        allowed_modules=["detection", "recognition", "landmark_3d_68"],
        providers=providers,
    )
    app.prepare(ctx_id=args.device if providers[0] == "CUDAExecutionProvider" else -1,
                det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size))
    missing_models = {"detection", "recognition"} - set(app.models)
    if missing_models:
        raise RuntimeError("В наборе модели отсутствуют компоненты: " + ", ".join(sorted(missing_models)))

    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    unreadable = 0
    used_face_ids: set[str] = set()
    started = time.monotonic()
    for position, frame_path in enumerate(frames, start=1):
        image = cv2.imread(str(frame_path))
        video_name, frame_number = parse_source(
            frame_path, frames_dir, args.input_type
        )
        if image is None:
            unreadable += 1
            frame_rows.append({
                "source_frame": str(frame_path), "video_name": video_name,
                "frame_number": frame_number, "status": "unreadable", "num_faces": 0,
            })
            continue
        height, width = image.shape[:2]
        detected = sorted(app.get(image), key=lambda face: float(face.bbox[0]))
        saved_in_frame = 0
        for face_index, face in enumerate(detected):
            embedding = np.asarray(getattr(face, "normed_embedding", None), dtype=np.float32)
            keypoints = getattr(face, "kps", None)
            if embedding.ndim != 1 or embedding.size == 0 or keypoints is None:
                continue
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(embedding).all() or norm <= 1e-12:
                continue
            embedding /= norm
            aligned = face_align.norm_crop(image, landmark=np.asarray(keypoints), image_size=112)
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            x1c, y1c = max(0.0, x1), max(0.0, y1)
            x2c, y2c = min(float(width), x2), min(float(height), y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            face_id = f"{safe_id(video_name)}_frame_{frame_number:06d}_face_{face_index:02d}"
            if face_id in used_face_ids:
                raise RuntimeError(
                    "Повторяющийся face_id. Используйте --input-type photos "
                    f"для независимых фотографий: {face_id}"
                )
            used_face_ids.add(face_id)
            aligned_rel = Path("aligned_faces") / f"{face_id}.jpg"
            if not cv2.imwrite(str(output_dir / aligned_rel), aligned, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise OSError(f"Не удалось записать {output_dir / aligned_rel}")
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            pitch, yaw, roll = pose_values(face)
            border_margin = min(x1c, y1c, width - x2c, height - y2c)
            embedding_index = len(embeddings)
            rows.append({
                "face_id": face_id,
                "source_frame": str(frame_path.resolve()),
                "video_name": video_name,
                "frame_number": frame_number,
                "face_index": face_index,
                "bbox_x1": round(x1c, 3), "bbox_y1": round(y1c, 3),
                "bbox_x2": round(x2c, 3), "bbox_y2": round(y2c, 3),
                "face_width": round(x2c - x1c, 3), "face_height": round(y2c - y1c, 3),
                "frame_width": width, "frame_height": height,
                "border_margin": round(border_margin, 3),
                "det_score": round(float(face.det_score), 6),
                "pitch": pitch, "yaw": yaw, "roll": roll,
                "blur_score": round(blur_score, 6),
                "aligned_path": str(aligned_rel),
                "embedding_index": embedding_index,
                "cluster_id": "", "rejection_reason": "",
            })
            embeddings.append(embedding)
            saved_in_frame += 1
        frame_rows.append({
            "source_frame": str(frame_path.resolve()), "video_name": video_name,
            "frame_number": frame_number, "status": "ok", "num_faces": saved_in_frame,
        })
        if position == 1 or position % args.progress_every == 0 or position == len(frames):
            elapsed = time.monotonic() - started
            print(f"[{position}/{len(frames)}] лиц={len(rows)}, скорость={position / max(elapsed, 1e-9):.2f} кадр/с")

    matrix = np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    write_csv(output_dir / "faces.csv", rows, FACE_FIELDS)
    write_csv(output_dir / "frame_stats.csv", frame_rows,
              ["source_frame", "video_name", "frame_number", "status", "num_faces"])
    save_npy(output_dir / "embeddings.npy", matrix)
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "providers": providers,
        "det_thresh": args.det_thresh,
        "det_size": args.det_size,
        "input_type": args.input_type,
        "frames_dir": str(frames_dir),
        "pattern": args.pattern,
        "processed_frames": len(frames),
        "unreadable_frames": unreadable,
        "detected_faces": len(rows),
        "embedding_shape": list(matrix.shape),
        "elapsed_seconds": round(elapsed, 3),
    }
    write_json(output_dir / "run_config.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подготовить лица и эмбеддинги из каталога кадров")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-type",
        choices=("video-frames", "photos"),
        default="video-frames",
        help=(
            "video-frames разбирает имена последовательных кадров; "
            "photos считает каждое изображение независимым источником"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--device", type=int, default=0, help="Номер GPU; -1 означает CPU")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_step < 1 or args.det_size < 64 or not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("Некорректные числовые параметры")
    summary = run(args)
    print("\nГотово")
    print("Обработано кадров:", summary["processed_frames"])
    print("Нечитаемых кадров:", summary["unreadable_frames"])
    print("Найдено лиц:", summary["detected_faces"])
    print("Форма массива эмбеддингов:", tuple(summary["embedding_shape"]))


if __name__ == "__main__":
    main()
