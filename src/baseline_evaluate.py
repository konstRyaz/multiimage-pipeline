#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.baseline_data import BaselineDataError
from face_pipeline.celeba_profiles import ALL_PROFILES, RESEARCH_PROFILES, apply_profile
from face_pipeline.baseline_workflow import (
    calibrate,
    evaluate,
    freeze,
    load_config,
    preflight,
    prepare,
    record_stage_failure,
    validate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Воспроизводимая оценка базового конвейера лиц")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "prepare", "calibrate", "validate", "freeze", "evaluate"):
        child = subparsers.add_parser(stage)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument(
            "--profile",
            choices=ALL_PROFILES,
            default="full",
            help="Профиль оценки CelebA; WIDER FACE и XQLFW всегда полные",
        )
        child.add_argument("--limit", type=int, help="Быстрая проверка: не более N элементов каждой части")
        if stage != "preflight":
            child.add_argument("--progress-interval", type=float, default=30.0, metavar="SECONDS",
                               help="Интервал живого прогресса и обновления ETA, по умолчанию 30 секунд")
        if stage == "evaluate":
            child.add_argument("--frozen", type=Path, required=True,
                               help="Обязательный заранее созданный файл замороженных параметров")
        if stage == "preflight":
            child.add_argument("--skip-model", action="store_true")
    status = subparsers.add_parser("status", help="Показать текущий прогресс и последние промежуточные метрики")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--metrics", type=int, default=3, metavar="N",
                        help="Сколько последних записей промежуточных метрик показать")
    return parser


def read_status(run_dir: Path, metric_count: int) -> dict[str, object]:
    progress_dir = run_dir.expanduser().resolve() / "progress"
    current_path = progress_dir / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.is_file() else {
        "status": "not_started", "message": f"Нет файла прогресса: {current_path}"
    }
    metrics = []
    metrics_path = progress_dir / "intermediate_metrics.jsonl"
    if metrics_path.is_file() and metric_count > 0:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(metrics) == metric_count:
                break
        metrics.reverse()
    completed_stages = []
    stages_dir = run_dir.expanduser().resolve() / "stages"
    if stages_dir.is_dir():
        completed_stages = sorted(path.stem for path in stages_dir.glob("*.json")
                                  if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete")
    return {"current": current, "completed_stages": completed_stages, "latest_intermediate_metrics": metrics}


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "status":
        if args.metrics < 0:
            print("ОШИБКА: --metrics не может быть отрицательным", file=sys.stderr)
            raise SystemExit(2)
        print(json.dumps(read_status(args.run_dir, args.metrics), ensure_ascii=False, indent=2))
        return
    run_dir = args.run_dir.expanduser().resolve()
    try:
        config = load_config(args.config.expanduser().resolve())
        apply_profile(config, args.profile)
        if args.limit is not None:
            if args.limit < 1:
                raise BaselineDataError("--limit должен быть положительным")
            if args.profile in RESEARCH_PROFILES:
                raise BaselineDataError(
                    "--limit несовместим с исследовательскими профилями: "
                    "их состав фиксируется манифестами, а WIDER FACE и XQLFW должны быть полными"
                )
            config["processing"]["limit"] = args.limit
        repo_root = Path(__file__).resolve().parents[1]
        if args.stage == "preflight":
            result = preflight(config, run_dir, check_model=not args.skip_model)
        elif args.stage == "prepare":
            result = prepare(config, run_dir, repo_root, args.progress_interval)
        elif args.stage == "calibrate":
            result = calibrate(config, run_dir, args.progress_interval)
        elif args.stage == "validate":
            result = validate(config, run_dir, args.progress_interval)
        elif args.stage == "freeze":
            result = freeze(config, run_dir, args.progress_interval)
        else:
            result = evaluate(config, run_dir, args.frozen.expanduser().resolve(), args.progress_interval)
    except (BaselineDataError, FileNotFoundError, RuntimeError, ValueError) as exc:
        record_stage_failure(run_dir, args.stage, exc)
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt as exc:
        record_stage_failure(run_dir, args.stage, exc)
        print("ОШИБКА: стадия прервана пользователем; сохранённый кэш можно использовать при повторном запуске",
              file=sys.stderr)
        raise SystemExit(130) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
