#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SESSION_TAG="${SESSION_TAG:-prod}"
DURATION_HOURS="${DURATION_HOURS:-12}"
TIMEOUT_KILL_GRACE_SECONDS="${TIMEOUT_KILL_GRACE_SECONDS:-20}"
PREPARE_MODEL="${PREPARE_MODEL:-false}"
PREPARE_MODEL_STRICT="${PREPARE_MODEL_STRICT:-false}"
MODEL_DAYS="${MODEL_DAYS:-7}"

# Production defaults for periodic GPU retraining while bot is running.
AUTO_MODEL_RETRAIN="${AUTO_MODEL_RETRAIN:-true}"
MODEL_REFRESH_MINUTES="${MODEL_REFRESH_MINUTES:-240}"
MODEL_TRAIN_DAYS="${MODEL_TRAIN_DAYS:-7}"
MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}"
MODEL_GPU_WARMUP="${MODEL_GPU_WARMUP:-true}"
MODEL_FORCE_PTX_JIT="${MODEL_FORCE_PTX_JIT:-true}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required but was not found on PATH." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_utc"
PREFIX="${SESSION_TAG}_${RUN_ID}"
BOT_LOG="$LOG_DIR/${PREFIX}_bot.log"
SUMMARY_LOG="$LOG_DIR/${PREFIX}_summary.json"
STATUS_FILE="$LOG_DIR/${PREFIX}.status"
META_FILE="$LOG_DIR/${PREFIX}.meta.json"

cat > "$META_FILE" <<EOF
{
  "session_tag": "${SESSION_TAG}",
  "run_id": "${RUN_ID}",
  "duration_hours": ${DURATION_HOURS},
  "timeout_kill_grace_seconds": ${TIMEOUT_KILL_GRACE_SECONDS},
  "python_bin": "${PYTHON_BIN}",
  "prepare_model": "${PREPARE_MODEL}",
  "model_days": ${MODEL_DAYS},
  "bot_log": "${BOT_LOG}",
  "summary_log": "${SUMMARY_LOG}",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export AUTO_MODEL_RETRAIN
export MODEL_REFRESH_MINUTES
export MODEL_TRAIN_DAYS
export MODEL_REQUIRE_GPU
export MODEL_GPU_WARMUP
export MODEL_FORCE_PTX_JIT
export MODEL_TRAIN_PYTHON="${MODEL_TRAIN_PYTHON:-$PYTHON_BIN}"

if [[ -n "${UV_RUN:-}" ]]; then
  echo "[session] warning: detected uv run launcher; this can re-resolve packages and break custom TensorFlow wheels" | tee -a "$BOT_LOG"
  echo "[session] recommendation: run directly via 'bash scripts/run_production_session.sh'" | tee -a "$BOT_LOG"
fi

echo "[session] run_id=${RUN_ID}" | tee -a "$BOT_LOG"
echo "[session] bot_log=${BOT_LOG}" | tee -a "$BOT_LOG"
echo "[session] auto_model_retrain=${AUTO_MODEL_RETRAIN}" | tee -a "$BOT_LOG"
echo "[session] model_refresh_minutes=${MODEL_REFRESH_MINUTES}" | tee -a "$BOT_LOG"
echo "[session] model_train_days=${MODEL_TRAIN_DAYS}" | tee -a "$BOT_LOG"
echo "[session] model_train_python=${MODEL_TRAIN_PYTHON}" | tee -a "$BOT_LOG"
echo "[session] model_require_gpu=${MODEL_REQUIRE_GPU}" | tee -a "$BOT_LOG"

if [[ "$PREPARE_MODEL" == "true" ]]; then
  echo "[session] running model warmup + training" | tee -a "$BOT_LOG"
  set +e
  MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "$ROOT_DIR/scripts/train_model_gpu_warmup.sh" --days "$MODEL_DAYS" --quiet >> "$BOT_LOG" 2>&1
  MODEL_PREP_EXIT_CODE=$?
  set -e

  if [[ "$MODEL_PREP_EXIT_CODE" -ne 0 ]]; then
    echo "[session] warning: model prep failed with exit code $MODEL_PREP_EXIT_CODE" | tee -a "$BOT_LOG"
    if [[ "$PREPARE_MODEL_STRICT" == "true" ]]; then
      echo "[session] PREPARE_MODEL_STRICT=true, aborting session" | tee -a "$BOT_LOG"
      exit "$MODEL_PREP_EXIT_CODE"
    fi
    echo "[session] continuing bot run with existing model/runtime" | tee -a "$BOT_LOG"
  fi
fi

set +e
timeout -k "${TIMEOUT_KILL_GRACE_SECONDS}s" "${DURATION_HOURS}h" "$PYTHON_BIN" src/bot.py >> "$BOT_LOG" 2>&1
EXIT_CODE=$?
set -e

echo "$EXIT_CODE" > "$STATUS_FILE"

"$PYTHON_BIN" "$ROOT_DIR/scripts/review_bot_session.py" --log "$BOT_LOG" --exit-code "$EXIT_CODE" > "$SUMMARY_LOG"

echo "[session] status_file=$STATUS_FILE"
echo "[session] summary_log=$SUMMARY_LOG"

if [[ "$EXIT_CODE" -eq 124 ]]; then
  echo "[session] timeout reached cleanly after ${DURATION_HOURS}h"
  exit 0
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "[session] bot exited with non-zero status: $EXIT_CODE" >&2
  exit "$EXIT_CODE"
fi

echo "[session] bot exited before timeout with status 0"
