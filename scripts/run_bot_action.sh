#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

CHUNK_MINUTES="${CHUNK_MINUTES:-340}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

printf 'Starting bot chunk for %s minutes\n' "$CHUNK_MINUTES"

# Create logs directory
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/bot_$(date -u +%Y%m%d_%H%M%S)_utc.log"

if command -v timeout >/dev/null 2>&1; then
  set +e
  timeout "${CHUNK_MINUTES}m" "$PYTHON_BIN" src/bot.py 2>&1 | tee "$LOG_FILE"
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
"$PYTHON_BIN" src/bot.py 2>&1 | tee "$LOG_FILE"
