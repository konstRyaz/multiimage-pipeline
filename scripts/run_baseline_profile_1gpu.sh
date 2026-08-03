#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$REPO_DIR/configs/baseline_evaluation_v1.json"
PROFILE=""
RUN_DIR=""
GPU=""
PROGRESS_INTERVAL=30
PYTHON_BIN=${PYTHON_BIN:-python}

usage() {
  echo "Использование: $0 --profile smoke|dev-a|dev-b|full --run-dir PATH --gpu INDEX [--config PATH] [--progress-interval SECONDS]"
}

while (($#)); do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
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

case "$PROFILE" in
  smoke|dev-a|dev-b|full) ;;
  *)
    echo "Требуется --profile smoke|dev-a|dev-b|full" >&2
    usage >&2
    exit 2
    ;;
esac
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
LOG="$RUN_DIR/${PROFILE}_run.log"
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

ensure_run_compatible() {
  "$PYTHON_BIN" - "$REPO_DIR" "$CONFIG" "$RUN_DIR/run_metadata.json" "$PROFILE" <<'PY'
import json
import sys
from pathlib import Path

repo, config_path, metadata_path, profile = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
from face_pipeline.baseline_workflow import experiment_fingerprint, load_config
from face_pipeline.celeba_profiles import apply_profile

if not metadata_path.is_file():
    raise SystemExit(0)
config = load_config(config_path)
apply_profile(config, str(profile))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("experiment_fingerprint") != experiment_fingerprint(config):
    print(
        "Каталог запуска несовместим с выбранным профилем или конфигурацией; "
        "используйте отдельный новый --run-dir.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
}

run_stage() {
  local stage="$1"
  shift
  if stage_complete "$stage"; then
    echo "=== $(date --iso-8601=seconds) SKIP $stage: стадия уже завершена ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== $(date --iso-8601=seconds) START $stage profile=$PROFILE ===" | tee -a "$LOG"
  set +e
  "$PYTHON_BIN" "$REPO_DIR/src/baseline_evaluate.py" "$stage" \
    --config "$CONFIG" --run-dir "$RUN_DIR" --profile "$PROFILE" \
    --progress-interval "$PROGRESS_INTERVAL" "$@" \
    2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  set -e
  echo "=== $(date --iso-8601=seconds) END $stage status=$status ===" | tee -a "$LOG"
  return "$status"
}

echo "=== $(date --iso-8601=seconds) START preflight profile=$PROFILE physical_GPU=$GPU logical_device=0 ===" | tee -a "$LOG"
"$PYTHON_BIN" "$REPO_DIR/src/baseline_evaluate.py" preflight \
  --config "$CONFIG" --run-dir "$RUN_DIR" --profile "$PROFILE" 2>&1 | tee -a "$LOG"
ensure_run_compatible

run_stage prepare
run_stage calibrate
run_stage validate

if [[ "$PROFILE" == "full" ]]; then
  run_stage freeze
  run_stage evaluate --frozen "$RUN_DIR/frozen_parameters.json"
  echo "=== $(date --iso-8601=seconds) ПОЛНЫЙ ПРОГОН ЗАВЕРШЁН ===" | tee -a "$LOG"
else
  echo "=== $(date --iso-8601=seconds) ИССЛЕДОВАТЕЛЬСКИЙ ПРОГОН ЗАВЕРШЁН ПОСЛЕ validate; CelebA test НЕ ИСПОЛЬЗОВАН ===" | tee -a "$LOG"
fi
