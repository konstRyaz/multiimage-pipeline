#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$REPO_DIR/configs/baseline_evaluation_v1.json"
RUN_DIR=""
GPU=""
PROGRESS_INTERVAL=30
PYTHON_BIN=${PYTHON_BIN:-python}

usage() {
  echo "Использование: $0 --run-dir PATH --gpu INDEX [--config PATH] [--progress-interval SECONDS]"
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --progress-interval)
      PROGRESS_INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Неизвестный аргумент: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$GPU" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Конфигурация не найдена: $CONFIG" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Интерпретатор не найден: $PYTHON_BIN. Сначала активируйте виртуальное окружение." >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
LOG="$RUN_DIR/full_run.log"
export CUDA_VISIBLE_DEVICES="$GPU"

stage_complete() {
  "$PYTHON_BIN" - "$RUN_DIR/stages/$1.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    complete = path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
except (OSError, json.JSONDecodeError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
}

run_stage() {
  local stage="$1"
  shift
  if stage_complete "$stage"; then
    echo "=== $(date --iso-8601=seconds) SKIP $stage: стадия уже завершена ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== $(date --iso-8601=seconds) START $stage ===" | tee -a "$LOG"
  set +e
  "$PYTHON_BIN" "$REPO_DIR/src/baseline_evaluate.py" "$stage" \
    --config "$CONFIG" --run-dir "$RUN_DIR" \
    --progress-interval "$PROGRESS_INTERVAL" "$@" \
    2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  set -e
  echo "=== $(date --iso-8601=seconds) END $stage status=$status ===" | tee -a "$LOG"
  return "$status"
}

echo "=== $(date --iso-8601=seconds) START preflight GPU=$GPU ===" | tee -a "$LOG"
"$PYTHON_BIN" "$REPO_DIR/src/baseline_evaluate.py" preflight \
  --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

run_stage prepare
run_stage calibrate
run_stage validate
run_stage freeze
run_stage evaluate --frozen "$RUN_DIR/frozen_parameters.json"

echo "=== $(date --iso-8601=seconds) ПОЛНЫЙ ПРОГОН ЗАВЕРШЁН ===" | tee -a "$LOG"
