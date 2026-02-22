#!/usr/bin/env bash
set -euo pipefail

CHUNK_MINUTES="${CHUNK_MINUTES:-340}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

printf 'Starting bot chunk for %s minutes\n' "$CHUNK_MINUTES"

if command -v timeout >/dev/null 2>&1; then
  set +e
  timeout "${CHUNK_MINUTES}m" "$PYTHON_BIN" bot.py
  exit_code=$?
  set -e

  if [[ "$exit_code" -eq 124 ]]; then
    echo "Bot chunk reached timeout cleanly."
    exit 0
  fi

  if [[ "$exit_code" -ne 0 ]]; then
    echo "Bot exited with non-zero status: $exit_code"
    exit "$exit_code"
  fi

  exit 0
fi

echo "GNU timeout not found. Running bot without chunk timeout."
"$PYTHON_BIN" bot.py
