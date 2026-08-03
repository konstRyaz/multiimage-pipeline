from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baseline_data import BaselineDataError, CelebARecord, WiderImage, XqlfwPair


MANIFEST_SCHEMA = "celeba_profile_manifest_v1"
MANIFEST_SEED = 20260803
SMOKE_PROFILE_SCHEMA = "technical_smoke_profile_v2"
RESEARCH_PROFILES = ("smoke", "dev-a", "dev-b")
ALL_PROFILES = (*RESEARCH_PROFILES, "full")
PROFILE_LIMITS = {
    "smoke": {"train": 64, "val": 32},
    "dev-a": {"train": 10_000, "val": 3_000},
    "dev-b": {"train": 10_000, "val": 3_000},
}
SMOKE_WIDER_LIMITS = {"train": 64, "val": 32}
SMOKE_XQLFW_PAIRS_PER_CLASS_PER_FOLD = 5
SOURCE_ANNOTATIONS = (
    "identity_CelebA.txt",
    "list_eval_partition.txt",
    "list_bbox_celeba.txt",
)


def profile_name(config: Mapping[str, Any]) -> str:
    value = str(config.get("evaluation_profile", "full"))
    if value not in ALL_PROFILES:
        raise BaselineDataError(
            f"Неизвестный профиль оценки {value!r}; ожидается: "
            + ", ".join(ALL_PROFILES)
        )
    return value


def is_research_profile(config: Mapping[str, Any]) -> bool:
    return profile_name(config) in RESEARCH_PROFILES


def apply_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in ALL_PROFILES:
        raise BaselineDataError(
            f"Неизвестный профиль оценки {profile!r}; ожидается: "
            + ", ".join(ALL_PROFILES)
        )
    if profile == "full":
        config.pop("evaluation_profile", None)
    else:
        config["evaluation_profile"] = profile
    return config


def _smoke_key(*values: object) -> str:
    raw = "\x1f".join(
        (MANIFEST_SCHEMA, str(MANIFEST_SEED), "smoke", *(str(value) for value in values))
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select_wider_records(
    config: Mapping[str, Any], records: Sequence[WiderImage], split: str
) -> list[WiderImage]:
    """Return a deterministic bounded WIDER subset only for the smoke profile."""
    if profile_name(config) != "smoke":
        return list(records)
    limit = SMOKE_WIDER_LIMITS[split]
    return sorted(
        records,
        key=lambda item: (_smoke_key("wider", split, item.relative_path), item.relative_path),
    )[:limit]


def select_xqlfw_pairs(
    config: Mapping[str, Any], pairs: Sequence[XqlfwPair]
) -> list[XqlfwPair]:
    """Keep a balanced deterministic sample from every XQLFW fold for smoke."""
    if profile_name(config) != "smoke":
        return list(pairs)
    selected: list[XqlfwPair] = []
    for fold in range(10):
        for same in (True, False):
            candidates = [
                pair for pair in pairs if pair.fold == fold and pair.same is same
            ]
            candidates.sort(
                key=lambda pair: (
                    _smoke_key(
                        "xqlfw",
                        fold,
                        int(same),
                        pair.left.as_posix(),
                        pair.right.as_posix(),
                    ),
                    pair.left.as_posix(),
                    pair.right.as_posix(),
                )
            )
            required = SMOKE_XQLFW_PAIRS_PER_CLASS_PER_FOLD
            if len(candidates) < required:
                raise BaselineDataError(
                    f"XQLFW smoke: в блоке {fold} недостаточно пар класса "
                    f"same={same}: {len(candidates)} < {required}"
                )
            selected.extend(candidates[:required])
    return selected


def xqlfw_expected_fold_size(config: Mapping[str, Any]) -> int:
    if profile_name(config) == "smoke":
        return 2 * SMOKE_XQLFW_PAIRS_PER_CLASS_PER_FOLD
    return 600


def smoke_profile_schema(config: Mapping[str, Any]) -> str | None:
    return SMOKE_PROFILE_SCHEMA if profile_name(config) == "smoke" else None


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BaselineDataError(f"Не удалось прочитать исходную разметку {path}: {exc}") from exc
    return digest.hexdigest()


def annotation_fingerprint(annotations_dir: Path) -> dict[str, str]:
    return {
        name: _sha256_file(annotations_dir / name)
        for name in SOURCE_ANNOTATIONS
    }


def _identity_key(partition: str, identity: int, profile_space: str) -> str:
    raw = "\x1f".join(
        (
            MANIFEST_SCHEMA,
            str(MANIFEST_SEED),
            partition,
            str(identity),
            profile_space,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _group_by_identity(
    records: Iterable[CelebARecord], partition: str
) -> dict[int, tuple[CelebARecord, ...]]:
    grouped: dict[int, list[CelebARecord]] = defaultdict(list)
    for record in records:
        if record.partition == partition:
            grouped[record.identity].append(record)
    return {
        identity: tuple(sorted(items, key=lambda item: item.filename))
        for identity, items in grouped.items()
    }


def _take_whole_identities(
    groups: Mapping[int, Sequence[CelebARecord]],
    identities: Sequence[int],
    limit: int,
) -> list[CelebARecord]:
    selected: list[CelebARecord] = []
    for identity in identities:
        group = list(groups[identity])
        if len(selected) + len(group) > limit:
            continue
        selected.extend(group)
    return selected


def build_profile_selections(
    records: Sequence[CelebARecord],
) -> dict[tuple[str, str], list[CelebARecord]]:
    output: dict[tuple[str, str], list[CelebARecord]] = {}
    for partition in ("train", "val"):
        groups = _group_by_identity(records, partition)
        all_identities = set(groups)

        dev_a_order = sorted(
            all_identities,
            key=lambda identity: (_identity_key(partition, identity, "dev-a"), identity),
        )
        dev_a = _take_whole_identities(
            groups, dev_a_order, PROFILE_LIMITS["dev-a"][partition]
        )
        dev_a_identities = {item.identity for item in dev_a}

        dev_b_order = sorted(
            all_identities - dev_a_identities,
            key=lambda identity: (_identity_key(partition, identity, "dev-b"), identity),
        )
        dev_b = _take_whole_identities(
            groups, dev_b_order, PROFILE_LIMITS["dev-b"][partition]
        )

        smoke_order = sorted(
            dev_a_identities,
            key=lambda identity: (_identity_key(partition, identity, "smoke"), identity),
        )
        smoke = _take_whole_identities(
            groups, smoke_order, PROFILE_LIMITS["smoke"][partition]
        )

        output[("dev-a", partition)] = dev_a
        output[("dev-b", partition)] = dev_b
        output[("smoke", partition)] = smoke
    return output


def manifest_filename(profile: str, partition: str) -> str:
    return f"celeba_{profile.replace('-', '_')}_{partition}.json"


def _manifest_payload(
    profile: str,
    partition: str,
    records: Sequence[CelebARecord],
    source_annotations: Mapping[str, str],
) -> dict[str, Any]:
    entries = [
        {
            "image": item.filename,
            "official_partition": item.partition,
            "identity": item.identity,
        }
        for item in records
    ]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "profile": profile,
        "official_partition": partition,
        "selection_seed": MANIFEST_SEED,
        "profile_space": profile,
        "actual_image_count": len(entries),
        "actual_identity_count": len({item.identity for item in records}),
        "source_annotations_sha256": dict(source_annotations),
        "records": entries,
    }


def make_manifest(
    profile: str,
    partition: str,
    records: Sequence[CelebARecord],
    source_annotations: Mapping[str, str],
) -> dict[str, Any]:
    payload = _manifest_payload(profile, partition, records, source_annotations)
    return {**payload, "content_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}


def validate_manifest(
    document: Mapping[str, Any], expected: Mapping[str, Any] | None = None
) -> None:
    stored = document.get("content_sha256")
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if not isinstance(stored, str) or stored != actual:
        raise BaselineDataError(
            "Манифест CelebA повреждён: SHA-256 содержимого не совпадает"
        )
    if document.get("schema_version") != MANIFEST_SCHEMA:
        raise BaselineDataError(
            f"Несовместимая схема манифеста CelebA; ожидается {MANIFEST_SCHEMA}"
        )
    records = document.get("records")
    if not isinstance(records, list):
        raise BaselineDataError("Манифест CelebA повреждён: records должен быть списком")
    partition = document.get("official_partition")
    if any(item.get("official_partition") != partition for item in records):
        raise BaselineDataError(
            "Манифест CelebA смешивает официальные части или содержит неверную запись"
        )
    if len(records) != document.get("actual_image_count"):
        raise BaselineDataError("Манифест CelebA содержит неверное число изображений")
    identities = {item.get("identity") for item in records}
    if len(identities) != document.get("actual_identity_count"):
        raise BaselineDataError("Манифест CelebA содержит неверное число личностей")
    if expected is not None and dict(document) != dict(expected):
        raise BaselineDataError(
            "Существующий манифест CelebA несовместим с текущей разметкой или "
            "правилами отбора. Сохраните его для расследования и используйте новый "
            "каталог запуска; автоматическая перегенерация запрещена."
        )


def ensure_research_manifests(
    run_dir: Path,
    records: Sequence[CelebARecord],
    annotations_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    selections = build_profile_selections(records)
    source_hashes = annotation_fingerprint(annotations_dir)
    directory = run_dir / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for key, selected in selections.items():
        profile, partition = key
        expected = make_manifest(profile, partition, selected, source_hashes)
        path = directory / manifest_filename(profile, partition)
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BaselineDataError(
                    f"Манифест CelebA повреждён и не будет перезаписан: {path}: {exc}"
                ) from exc
            validate_manifest(existing, expected)
            output[key] = existing
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        output[key] = expected
    return output


def records_from_manifest(
    manifest: Mapping[str, Any], records: Sequence[CelebARecord]
) -> list[CelebARecord]:
    validate_manifest(manifest)
    by_name = {item.filename: item for item in records}
    selected: list[CelebARecord] = []
    for entry in manifest["records"]:
        name = entry["image"]
        record = by_name.get(name)
        if record is None:
            raise BaselineDataError(
                f"Манифест CelebA ссылается на отсутствующую запись разметки: {name}"
            )
        if (
            record.partition != entry["official_partition"]
            or record.identity != entry["identity"]
        ):
            raise BaselineDataError(
                f"Манифест CelebA несовместим с исходной разметкой: {name}"
            )
        selected.append(record)
    return selected
