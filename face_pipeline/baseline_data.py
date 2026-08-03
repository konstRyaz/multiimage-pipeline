from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class BaselineDataError(ValueError):
    """Ошибка структуры или протокола набора данных."""


@dataclass(frozen=True)
class WiderImage:
    relative_path: str
    boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class XqlfwPair:
    left: Path
    right: Path
    same: bool
    fold: int


@dataclass(frozen=True)
class CelebARecord:
    filename: str
    identity: int
    partition: str
    bbox: tuple[float, float, float, float] | None


def _nonempty(path: Path) -> list[str]:
    if not path.is_file():
        raise BaselineDataError(f"Файл разметки не найден: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def _is_wider_zero_placeholder(raw: str) -> bool:
    values = raw.split()
    return len(values) == 10 and all(value == "0" for value in values)


def parse_wider_annotations(path: Path) -> list[WiderImage]:
    lines = _nonempty(path)
    output: list[WiderImage] = []
    position = 0
    while position < len(lines):
        relative = lines[position]
        position += 1
        if position >= len(lines):
            raise BaselineDataError(f"WIDER FACE: после {relative} нет числа рамок")
        try:
            count = int(lines[position])
        except ValueError as exc:
            raise BaselineDataError(f"WIDER FACE: некорректное число рамок для {relative}") from exc
        position += 1
        if (
            count == 0
            and position < len(lines)
            and _is_wider_zero_placeholder(lines[position])
        ):
            position += 1
        boxes: list[tuple[float, float, float, float]] = []
        if position + count > len(lines):
            raise BaselineDataError(f"WIDER FACE: для {relative} не хватает строк рамок")
        for raw in lines[position:position + count]:
            values = raw.split()
            if len(values) < 4:
                raise BaselineDataError(f"WIDER FACE: некорректная рамка: {raw}")
            x, y, width, height = map(float, values[:4])
            if width > 0 and height > 0:
                boxes.append((x, y, x + width, y + height))
        position += count
        output.append(WiderImage(relative, tuple(boxes)))
    if len({item.relative_path for item in output}) != len(output):
        raise BaselineDataError("WIDER FACE: пути изображений в разметке должны быть уникальны")
    return output


def validate_wider_images(records: Iterable[WiderImage], images_dir: Path) -> None:
    missing = [item.relative_path for item in records if not (images_dir / item.relative_path).is_file()]
    if missing:
        sample = ", ".join(missing[:5])
        raise BaselineDataError(f"WIDER FACE: отсутствуют {len(missing)} изображений, первые: {sample}")


def _xqlfw_image(images_dir: Path, name: str, number: str) -> Path:
    filename = f"{name}_{int(number):04d}.jpg"
    candidates = (images_dir / name / filename, images_dir / filename)
    return next((path for path in candidates if path.is_file()), candidates[0])


def parse_xqlfw_pairs(path: Path, images_dir: Path, require_images: bool = True) -> list[XqlfwPair]:
    lines = _nonempty(path)
    if not lines:
        raise BaselineDataError("XQLFW: файл пар пуст")
    header = lines.pop(0).split()
    if len(header) != 2 or not all(value.isdigit() for value in header):
        raise BaselineDataError("XQLFW: ожидается официальный заголовок '10 300'")
    folds, pairs_per_kind = map(int, header)
    if folds != 10 or pairs_per_kind != 300:
        raise BaselineDataError(f"XQLFW: ожидается 10 блоков по 600 пар, заголовок: {folds} {pairs_per_kind}")
    expected = folds * pairs_per_kind * 2
    if len(lines) != expected:
        raise BaselineDataError(f"XQLFW: ожидалось 6000 записей пар, найдено {len(lines)}")
    output: list[XqlfwPair] = []
    for index, raw in enumerate(lines):
        values = raw.split()
        if len(values) == 3:
            left = _xqlfw_image(images_dir, values[0], values[1])
            right = _xqlfw_image(images_dir, values[0], values[2])
            same = True
        elif len(values) == 4:
            left = _xqlfw_image(images_dir, values[0], values[1])
            right = _xqlfw_image(images_dir, values[2], values[3])
            same = False
        else:
            raise BaselineDataError(f"XQLFW: некорректная строка {index + 2}: {raw}")
        output.append(XqlfwPair(left, right, same, index // 600))
    for fold in range(10):
        block = output[fold * 600:(fold + 1) * 600]
        if len(block) != 600 or sum(pair.same for pair in block) != 300:
            raise BaselineDataError(f"XQLFW: блок {fold} должен содержать 300 положительных и 300 отрицательных пар")
    if require_images:
        missing = sorted({str(path) for pair in output for path in (pair.left, pair.right) if not path.is_file()})
        if missing:
            raise BaselineDataError(f"XQLFW: отсутствуют {len(missing)} изображений, первые: {', '.join(missing[:5])}")
    return output


PARTITION_NAMES = {0: "train", 1: "val", 2: "test"}


def _parse_two_columns(path: Path, expected: int | None = None) -> dict[str, int]:
    lines = _nonempty(path)
    if lines and len(lines[0].split()) == 1:
        declared = int(lines.pop(0))
        if expected is not None and declared != expected:
            raise BaselineDataError(f"{path.name}: объявлено {declared}, ожидалось {expected}")
    result: dict[str, int] = {}
    for line in lines:
        values = line.split()
        if len(values) != 2:
            raise BaselineDataError(f"{path.name}: некорректная строка: {line}")
        if values[0] in result:
            raise BaselineDataError(f"{path.name}: повтор изображения {values[0]}")
        result[values[0]] = int(values[1])
    return result


def _parse_celeba_boxes(
    path: Path,
) -> dict[str, tuple[float, float, float, float] | None]:
    lines = _nonempty(path)
    if lines and len(lines[0].split()) == 1:
        lines.pop(0)
    if lines and lines[0].lower().startswith("image_id"):
        lines.pop(0)
    result: dict[str, tuple[float, float, float, float] | None] = {}
    for line in lines:
        values = line.split()
        if len(values) != 5:
            raise BaselineDataError(f"{path.name}: некорректная строка: {line}")
        x, y, width, height = map(float, values[1:])
        if width == 0 and height == 0:
            result[values[0]] = None
            continue
        if width <= 0 or height <= 0:
            raise BaselineDataError(f"{path.name}: неположительная рамка у {values[0]}")
        result[values[0]] = (x, y, x + width, y + height)
    return result


def load_celeba_records(
    annotations_dir: Path,
    images_dir: Path | None = None,
    expected: int = 202599,
    allowed_partitions: set[str] | None = None,
) -> list[CelebARecord]:
    identities = _parse_two_columns(annotations_dir / "identity_CelebA.txt", expected)
    partitions = _parse_two_columns(annotations_dir / "list_eval_partition.txt", expected)
    boxes = _parse_celeba_boxes(annotations_dir / "list_bbox_celeba.txt")
    names = set(identities)
    if names != set(partitions) or names != set(boxes):
        raise BaselineDataError("CelebA: состав изображений в identity, partition и bbox не совпадает")
    if expected and len(names) != expected:
        raise BaselineDataError(f"CelebA: ожидалось {expected} записей, найдено {len(names)}")
    invalid = sorted({value for value in partitions.values() if value not in PARTITION_NAMES})
    if invalid:
        raise BaselineDataError(f"CelebA: неизвестные номера частей: {invalid}")
    selected_names = sorted(
        name
        for name in names
        if allowed_partitions is None
        or PARTITION_NAMES[partitions[name]] in allowed_partitions
    )
    if images_dir is not None:
        missing = [name for name in selected_names if not (images_dir / name).is_file()]
        if missing:
            raise BaselineDataError(f"CelebA: отсутствуют {len(missing)} изображений, первые: {', '.join(missing[:5])}")
    records = [
        CelebARecord(name, identities[name], PARTITION_NAMES[partitions[name]], boxes[name])
        for name in selected_names
    ]
    validate_identity_disjoint(records)
    return records


def validate_identity_disjoint(records: Iterable[CelebARecord]) -> None:
    by_partition: dict[str, set[int]] = {name: set() for name in PARTITION_NAMES.values()}
    for record in records:
        by_partition[record.partition].add(record.identity)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_partition[left] & by_partition[right]
        if overlap:
            sample = ", ".join(map(str, sorted(overlap)[:10]))
            raise BaselineDataError(f"CelebA: личности пересекаются между {left} и {right}: {sample}")
