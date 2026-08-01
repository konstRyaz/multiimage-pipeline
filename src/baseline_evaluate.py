#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.baseline_data import BaselineDataError
from face_pipeline.baseline_workflow import (
    calibrate,
    evaluate,
    freeze,
    load_config,
    preflight,
    prepare,
    validate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Воспроизводимая оценка базового конвейера лиц")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "prepare", "calibrate", "validate", "freeze", "evaluate"):
        child = subparsers.add_parser(stage)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--limit", type=int, help="Быстрая проверка: не более N элементов каждой части")
        if stage == "evaluate":
            child.add_argument("--frozen", type=Path, required=True,
                               help="Обязательный заранее созданный файл замороженных параметров")
        if stage == "preflight":
            child.add_argument("--skip-model", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        if args.limit is not None:
            if args.limit < 1:
                raise BaselineDataError("--limit должен быть положительным")
            config["processing"]["limit"] = args.limit
        run_dir = args.run_dir.expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[1]
        if args.stage == "preflight":
            result = preflight(config, run_dir, check_model=not args.skip_model)
        elif args.stage == "prepare":
            result = prepare(config, run_dir, repo_root)
        elif args.stage == "calibrate":
            result = calibrate(config, run_dir)
        elif args.stage == "validate":
            result = validate(config, run_dir)
        elif args.stage == "freeze":
            result = freeze(config, run_dir)
        else:
            result = evaluate(config, run_dir, args.frozen.expanduser().resolve())
    except (BaselineDataError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
