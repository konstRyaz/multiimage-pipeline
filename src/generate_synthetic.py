#!/usr/bin/env python3
"""Создаёт небольшой воспроизводимый набор без InsightFace и реальных фотографий."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.io import prepare_output_dir, save_npy, write_csv, write_json


PERSON_STYLE = {
    "alice": ((236, 184, 142), (72, 42, 24), (30, 80, 180)),
    "boris": ((181, 126, 91), (25, 22, 20), (30, 145, 75)),
    "clara": ((244, 205, 175), (172, 92, 35), (170, 45, 100)),
    "noise": ((175, 175, 175), (85, 85, 85), (90, 90, 90)),
}


def portrait(person: str, frame: int, size: int = 112) -> Image.Image:
    skin, hair, background = PERSON_STYLE[person]
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    wobble = int(2 * math.sin(frame * 0.37))
    draw.ellipse((18 + wobble, 9, 94 + wobble, 105), fill=skin, outline=(80, 55, 40), width=2)
    if person == "clara":
        draw.arc((15, 2, 98, 66), 185, 355, fill=hair, width=14)
    else:
        draw.pieslice((16 + wobble, 3, 96 + wobble, 65), 180, 360, fill=hair)
    eye_y = 48 + (frame % 3 - 1)
    draw.ellipse((38 + wobble, eye_y, 45 + wobble, eye_y + 5), fill="black")
    draw.ellipse((68 + wobble, eye_y, 75 + wobble, eye_y + 5), fill="black")
    draw.line((56 + wobble, 54, 53 + wobble, 70, 60 + wobble, 70), fill=(105, 70, 55), width=2)
    mouth_curve = frame % 5
    if mouth_curve < 3:
        draw.arc((43 + wobble, 69, 70 + wobble, 91), 5, 175, fill=(120, 30, 35), width=3)
    else:
        draw.arc((43 + wobble, 78, 70 + wobble, 91), 185, 355, fill=(120, 30, 35), width=3)
    return image


def schedule() -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    def add(person: str, start: int, end: int) -> None:
        for frame in range(start, end + 1):
            result.setdefault(frame, []).append(person)
    add("alice", 0, 39)
    add("boris", 15, 28)
    add("alice", 50, 89)
    add("clara", 60, 76)
    add("boris", 100, 119)
    add("alice", 110, 119)
    for frame in (35, 79, 105):
        result.setdefault(frame, []).append("noise")
    return result


def run(output_dir: Path, seed: int, overwrite: bool) -> dict[str, int]:
    output_dir = output_dir.resolve()
    prepare_output_dir(output_dir, overwrite)
    aligned_dir = output_dir / "aligned_faces"
    frames_dir = output_dir / "source_frames"
    aligned_dir.mkdir()
    frames_dir.mkdir()
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(512, 4)))
    centers = {name: basis[:, index].astype(np.float32) for index, name in enumerate(PERSON_STYLE)}

    rows: list[dict[str, object]] = []
    truths: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    for frame, people in sorted(schedule().items()):
        frame_image = Image.new("RGB", (640, 360), (24 + frame % 20, 32, 48))
        frame_draw = ImageDraw.Draw(frame_image)
        frame_draw.text((10, 10), f"synthetic_video  frame {frame:04d}", fill="white")
        pending: list[tuple[str, int, tuple[int, int, int, int], Image.Image]] = []
        for face_index, person in enumerate(people):
            x1 = 65 + face_index * 285 + int(8 * math.sin(frame * 0.19 + face_index))
            y1 = 88 + int(6 * math.cos(frame * 0.13 + face_index))
            side = 145 if person != "noise" else 58
            x2, y2 = x1 + side, y1 + side
            crop = portrait(person, frame)
            if (frame + face_index) % 31 == 0:
                crop = crop.filter(ImageFilter.GaussianBlur(radius=2.4))
            preview = crop.resize((side, side), Image.Resampling.BILINEAR)
            frame_image.paste(preview, (x1, y1))
            frame_draw.rectangle((x1, y1, x2, y2), outline=(245, 210, 30), width=2)
            pending.append((person, face_index, (x1, y1, x2, y2), crop))
        source_rel = Path("source_frames") / f"synthetic_video_frame_{frame:04d}.jpg"
        frame_image.save(output_dir / source_rel, quality=92)

        for person, face_index, (x1, y1, x2, y2), crop in pending:
            face_id = f"synthetic_video_frame_{frame:06d}_face_{face_index:02d}"
            aligned_rel = Path("aligned_faces") / f"{face_id}.jpg"
            crop.save(output_dir / aligned_rel, quality=94)
            if person == "noise":
                vector = rng.normal(size=512).astype(np.float32)
            else:
                noise_scale = 0.025 if frame % 23 else 0.040
                vector = centers[person] + rng.normal(scale=noise_scale, size=512).astype(np.float32)
            vector /= np.linalg.norm(vector)
            embedding_index = len(embeddings)
            embeddings.append(vector)
            blur_score = 12.0 if (frame + face_index) % 31 == 0 else 140.0 + float(frame % 17) * 7
            det_score = 0.58 if person == "noise" else 0.91 + 0.06 * ((frame % 7) / 6)
            yaw = float(28 * math.sin(frame * 0.17 + face_index))
            pitch = float(12 * math.cos(frame * 0.11))
            roll = float(8 * math.sin(frame * 0.07))
            rows.append({
                "face_id": face_id,
                "source_frame": str(source_rel),
                "video_name": "synthetic_video",
                "frame_number": frame,
                "face_index": face_index,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "face_width": x2 - x1, "face_height": y2 - y1,
                "frame_width": 640, "frame_height": 360,
                "border_margin": min(x1, y1, 640 - x2, 360 - y2),
                "det_score": round(det_score, 6),
                "pitch": round(pitch, 4), "yaw": round(yaw, 4), "roll": round(roll, 4),
                "blur_score": blur_score,
                "aligned_path": str(aligned_rel),
                "embedding_index": embedding_index,
                "cluster_id": "", "rejection_reason": "",
            })
            is_acceptable = int(person != "noise" and blur_score >= 20.0)
            truths.append({
                "face_id": face_id,
                "true_person": person,
                "is_noise": int(person == "noise"),
                "is_acceptable": is_acceptable,
            })

    write_csv(output_dir / "faces.csv", rows)
    write_csv(
        output_dir / "ground_truth.csv",
        truths,
        ["face_id", "true_person", "is_noise", "is_acceptable"],
    )
    save_npy(output_dir / "embeddings.npy", np.stack(embeddings).astype(np.float32))
    summary = {"frames_with_faces": len(schedule()), "faces": len(rows), "real_identities": 3, "noise_faces": 3}
    write_json(output_dir / "run_config.json", {"synthetic": True, "seed": seed, **summary})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать синтетический набор для проверки конвейера")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/synthetic_demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run(args.output_dir, args.seed, args.overwrite)
    print("Синтетический набор создан:", args.output_dir)
    print("Кадров с лицами:", summary["frames_with_faces"])
    print("Лиц:", summary["faces"])
    print("Настоящих личностей:", summary["real_identities"])


if __name__ == "__main__":
    main()
