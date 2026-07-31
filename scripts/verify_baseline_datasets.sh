#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 "$(dirname "$0")/baseline_datasets.py" verify --help
fi

: "${DATA_ROOT:?Задайте DATA_ROOT: export DATA_ROOT=/путь/к/datasets}"

PYTHON="$DATA_ROOT/.download_tools/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Ошибка: сначала запустите scripts/download_baseline_datasets.sh." >&2
  exit 2
fi

exec "$PYTHON" "$(dirname "$0")/baseline_datasets.py" \
  verify --data-root "$DATA_ROOT" "$@"
