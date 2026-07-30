#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.experiment_config import load_experiment_config, resolve_paths
from face_pipeline.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запустить воспроизводимый эксперимент фильтрации на тяжёлом кэше"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Каталог с faces.csv и embeddings.npy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiments-dir", type=Path)
    parser.add_argument("--labels", type=Path, help="CSV-разметка пригодности; положительный класс — пригодное лицо")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Пересчитать только дешёвые производные признаки; InsightFace не запускается",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = resolve_paths(
        load_experiment_config(config_path),
        config_path,
        args.run_dir,
        args.experiments_dir,
        args.labels,
    )
    result = run_experiment(
        config,
        repo_root=Path(__file__).resolve().parents[1],
        force_features=args.force_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
