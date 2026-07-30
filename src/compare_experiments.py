#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_pipeline.experiment_config import load_experiment_config, resolve_paths
from face_pipeline.experiments import compare_experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить несколько конфигураций на одном тяжёлом и производном кэше"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--experiments-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    configs = []
    for path in args.configs:
        config_path = path.resolve()
        configs.append(resolve_paths(
            load_experiment_config(config_path),
            config_path,
            args.run_dir,
            args.experiments_dir,
            args.labels,
        ))
    source_dirs = {config["paths"]["run_dir"] for config in configs}
    experiment_dirs = {config["paths"]["experiments_dir"] for config in configs}
    if len(source_dirs) != 1:
        raise SystemExit("Все конфигурации сравнения должны использовать один run_dir")
    if len(experiment_dirs) != 1:
        raise SystemExit("Все конфигурации сравнения должны использовать один experiments_dir")
    result = compare_experiments(
        configs,
        repo_root=Path(__file__).resolve().parents[1],
        force_features=args.force_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
