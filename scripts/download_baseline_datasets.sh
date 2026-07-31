#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Ошибка: установщик нельзя запускать от root." >&2
  exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 "$(dirname "$0")/baseline_datasets.py" --help
fi

: "${DATA_ROOT:?Задайте DATA_ROOT: export DATA_ROOT=/путь/к/datasets}"

command -v python3 >/dev/null 2>&1 || {
  echo "Ошибка: python3 не найден. Установщик не использует sudo." >&2
  exit 2
}

mkdir -p "$DATA_ROOT"
TOOLS_DIR="$DATA_ROOT/.download_tools"

if [[ ! -x "$TOOLS_DIR/bin/python" ]]; then
  if ! python3 -m venv "$TOOLS_DIR"; then
    echo "Ошибка: не удалось создать $TOOLS_DIR." >&2
    echo "Нужен python3 с модулем venv; sudo установщик намеренно не использует." >&2
    exit 2
  fi
fi

"$TOOLS_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "gdown>=5,<6" "py7zr>=0.21,<1"

exec "$TOOLS_DIR/bin/python" "$(dirname "$0")/baseline_datasets.py" \
  download --data-root "$DATA_ROOT" "$@"
