#!/usr/bin/env python3
"""Загрузка и проверка наборов для baseline_v1 без системной установки."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


WIDER_TRAIN_ID = "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M"
WIDER_VAL_ID = "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"
WIDER_SPLIT_URL = (
    "https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"
)
WIDER_EVAL_URL = "https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip"
XQLFW_BASE = "https://github.com/Martlgap/xqlfw/releases/download/1.0"
CELEBA_FOLDER_URL = (
    "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8"
    "?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg"
)


def say(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_http(url: str, target: Path) -> None:
    """Скачать файл с повторным использованием незавершённого .part."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "baseline-datasets/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if offset and error.code == 416:
            partial.replace(target)
            return
        raise RuntimeError(f"не удалось скачать {url}: HTTP {error.code}") from error

    status = getattr(response, "status", None)
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0
    total_header = response.headers.get("Content-Length")
    total = offset + int(total_header) if total_header else None
    with response, partial.open(mode) as stream:
        copied = offset
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            copied += len(block)
            if total:
                print(f"\r  {target.name}: {copied / 1024**2:.1f}/{total / 1024**2:.1f} МиБ", end="")
    if total:
        print()
    partial.replace(target)


def run_gdown(args: list[str]) -> None:
    command = [sys.executable, "-m", "gdown", *args]
    subprocess.run(command, check=True)


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise RuntimeError(f"повреждён файл {bad} внутри {archive}")
        root = destination.resolve()
        for info in zipped.infolist():
            candidate = (destination / info.filename).resolve()
            if root not in candidate.parents and candidate != root:
                raise RuntimeError(f"небезопасный путь в архиве: {info.filename}")
        zipped.extractall(destination)


def image_count(directory: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def download_wider(root: Path) -> None:
    base = root / "widerface"
    downloads = base / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    train_archive = downloads / "WIDER_train.zip"
    val_archive = downloads / "WIDER_val.zip"
    split_archive = downloads / "wider_face_split.zip"
    eval_archive = downloads / "eval_tools.zip"
    say("\n[1/3] WIDER FACE train и validation")
    if not train_archive.exists():
        run_gdown(["--continue", "--id", WIDER_TRAIN_ID, "-O", str(train_archive)])
    if not val_archive.exists():
        run_gdown(["--continue", "--id", WIDER_VAL_ID, "-O", str(val_archive)])
    if not split_archive.exists():
        download_http(WIDER_SPLIT_URL, split_archive)
    if not eval_archive.exists():
        download_http(WIDER_EVAL_URL, eval_archive)
    extract_zip(train_archive, base)
    extract_zip(val_archive, base)
    extract_zip(split_archive, base)
    extract_zip(eval_archive, base / "evaluation")
    write_checksums(downloads)


def download_xqlfw(root: Path) -> None:
    base = root / "xqlfw"
    downloads = base / "downloads"
    archive = downloads / "xqlfw.zip"
    say("\n[2/3] XQLFW, исходные невыравненные изображения")
    if not archive.exists():
        download_http(f"{XQLFW_BASE}/xqlfw.zip", archive)
    for name in ("xqlfw_pairs.txt", "xqlfw_scores.txt"):
        target = base / name
        if not target.exists():
            download_http(f"{XQLFW_BASE}/{name}", target)
    extract_zip(archive, base)
    write_checksums(downloads, extra=[base / "xqlfw_pairs.txt", base / "xqlfw_scores.txt"])


def locate_celeba_images(base: Path) -> Path | None:
    candidates = [path for path in base.rglob("img_celeba") if path.is_dir()]
    return candidates[0] if candidates else None


def extract_celeba_archives(downloads: Path, base: Path) -> None:
    zip_candidates = list(downloads.rglob("img_celeba.zip"))
    if zip_candidates:
        extract_zip(zip_candidates[0], base)
        return
    first_parts = list(downloads.rglob("img_celeba.7z.001"))
    archives = list(downloads.rglob("img_celeba.7z"))
    source = first_parts[0] if first_parts else (archives[0] if archives else None)
    if source is None:
        return
    import multivolumefile
    import py7zr

    if source.name.endswith(".001"):
        stream = multivolumefile.open(str(source)[:-4], mode="rb")
    else:
        stream = source.open("rb")
    with stream, py7zr.SevenZipFile(stream, mode="r") as archive:
        archive.extractall(path=base)


def copy_celeba_annotations(downloads: Path, annotations: Path) -> None:
    names = (
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_celeba.txt",
        "list_eval_partition.txt",
    )
    annotations.mkdir(parents=True, exist_ok=True)
    for name in names:
        matches = list(downloads.rglob(name))
        if matches and not (annotations / name).exists():
            shutil.copy2(matches[0], annotations / name)


def download_celeba(root: Path, accepted: bool, source: Path | None) -> None:
    say("\n[3/3] CelebA In-The-Wild")
    base = root / "celeba"
    downloads = base / "downloads"
    annotations = base / "annotations"
    if not accepted:
        say("  Пропущено: добавьте --accept-celeba после принятия официальных условий.")
        return
    downloads.mkdir(parents=True, exist_ok=True)
    if source:
        if not source.exists():
            raise RuntimeError(f"каталог CelebA не найден: {source}")
        say(f"  Импорт файлов из {source}")
        for item in source.iterdir():
            destination = downloads / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
    elif not any(downloads.iterdir()):
        say("  Загрузка официального каталога Google Drive")
        run_gdown(["--folder", "--remaining-ok", CELEBA_FOLDER_URL, "-O", str(downloads)])
    extract_celeba_archives(downloads, base)
    copy_celeba_annotations(downloads, annotations)


def write_checksums(directory: Path, extra: list[Path] | None = None) -> None:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.endswith(".part")
        and path.name != "SHA256SUMS.local"
    )
    files.extend(extra or [])
    lines = [f"{sha256(path)}  {path.name}" for path in files]
    (directory / "SHA256SUMS.local").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_file(base: Path, name: str) -> Path | None:
    return next((path for path in base.rglob(name) if path.is_file()), None)


def verify_wider(root: Path) -> list[str]:
    base = root / "widerface"
    errors: list[str] = []
    counts = {}
    for split, expected in (("train", 12880), ("val", 3226)):
        images = base / f"WIDER_{split}" / "images"
        counts[split] = image_count(images) if images.exists() else 0
        if counts[split] != expected:
            errors.append(f"WIDER FACE {split}: ожидалось {expected} изображений, найдено {counts[split]}")
        if not find_file(base, f"wider_face_{split}_bbx_gt.txt"):
            errors.append(f"WIDER FACE: нет wider_face_{split}_bbx_gt.txt")
    for name in ("wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"):
        if not find_file(base, name):
            errors.append(f"WIDER FACE: нет {name}")
    if not errors:
        say(f"  OK WIDER FACE: train {counts['train']}, val {counts['val']} изображений")
    return errors


def verify_xqlfw(root: Path) -> list[str]:
    base = root / "xqlfw"
    errors: list[str] = []
    count = image_count(base)
    if count != 13233:
        errors.append(f"XQLFW: ожидалось 13233 изображения, найдено {count}")
    pairs = base / "xqlfw_pairs.txt"
    scores = base / "xqlfw_scores.txt"
    if not pairs.is_file() or not pairs.stat().st_size:
        errors.append("XQLFW: нет xqlfw_pairs.txt")
    elif xqlfw_pair_count(pairs) != 6000:
        errors.append(
            f"XQLFW: ожидалось 6000 пар, найдено {xqlfw_pair_count(pairs)}"
        )
    if not scores.is_file() or not scores.stat().st_size:
        errors.append("XQLFW: нет xqlfw_scores.txt")
    if not errors:
        say(f"  OK XQLFW: {count} изображений, протокол пар и оценки качества")
    return errors


def xqlfw_pair_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as stream:
        lines = [line.split() for line in stream if line.strip()]
    if lines and len(lines[0]) == 2 and all(value.isdigit() for value in lines[0]):
        folds, pairs_per_kind = map(int, lines[0])
        return folds * pairs_per_kind * 2
    return len(lines)


def annotation_rows(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as stream:
        lines = [line for line in stream if line.strip()]
    if lines and lines[0].strip().isdigit():
        return int(lines[0].strip())
    return len(lines)


def verify_celeba(root: Path, required: bool) -> list[str]:
    base = root / "celeba"
    images = locate_celeba_images(base) if base.exists() else None
    if images is None:
        return ["CelebA: нет каталога img_celeba"] if required else []
    errors: list[str] = []
    count = image_count(images)
    if count != 202599:
        errors.append(f"CelebA: ожидалось 202599 изображений, найдено {count}")
    annotations = base / "annotations"
    for name in (
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_celeba.txt",
        "list_eval_partition.txt",
    ):
        path = annotations / name
        if not path.is_file():
            errors.append(f"CelebA: нет annotations/{name}")
        elif annotation_rows(path) != 202599:
            errors.append(f"CelebA: неожиданное число записей в {name}")
    if not errors:
        say(f"  OK CelebA: {count} изображений и пять файлов разметки")
    return errors


def verify(root: Path, require_celeba: bool) -> None:
    say("\nПроверка структуры наборов")
    errors = verify_wider(root) + verify_xqlfw(root) + verify_celeba(root, require_celeba)
    if errors:
        for error in errors:
            print(f"  ОШИБКА: {error}", file=sys.stderr)
        if any("identity_CelebA.txt" in error for error in errors):
            print(
                "  Идентификаторы CelebA выдаются авторами по запросу. "
                "Положите полученный файл в DATA_ROOT/celeba/annotations/.",
                file=sys.stderr,
            )
        raise SystemExit(1)
    say("Все запрошенные наборы готовы.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--require-celeba", action="store_true")
        if command == "download":
            child.add_argument("--accept-celeba", action="store_true")
            child.add_argument("--celeba-source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "download":
        download_wider(root)
        download_xqlfw(root)
        download_celeba(root, args.accept_celeba, args.celeba_source)
        verify(root, require_celeba=args.require_celeba or args.accept_celeba)
    else:
        verify(root, require_celeba=args.require_celeba)


if __name__ == "__main__":
    main()
