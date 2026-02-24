#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SESSION_TAG="${SESSION_TAG:-gpuprod}"
DURATION_HOURS="${DURATION_HOURS:-12}"
TIMEOUT_KILL_GRACE_SECONDS="${TIMEOUT_KILL_GRACE_SECONDS:-20}"
PREPARE_MODEL="${PREPARE_MODEL:-true}"
PREPARE_MODEL_STRICT="${PREPARE_MODEL_STRICT:-true}"
MODEL_DAYS="${MODEL_DAYS:-7}"

# Production defaults for periodic GPU retraining while bot is running.
AUTO_MODEL_RETRAIN="${AUTO_MODEL_RETRAIN:-true}"
MODEL_REFRESH_MINUTES="${MODEL_REFRESH_MINUTES:-240}"
MODEL_TRAIN_DAYS="${MODEL_TRAIN_DAYS:-7}"
MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}"
MODEL_GPU_WARMUP="${MODEL_GPU_WARMUP:-true}"
MODEL_FORCE_PTX_JIT="${MODEL_FORCE_PTX_JIT:-true}"
MODEL_LOCAL_TF_WHEEL_PATH="${MODEL_LOCAL_TF_WHEEL_PATH:-$ROOT_DIR/.wheels/tensorflow-2.22.0.dev0+selfbuilt-cp313-cp313-linux_x86_64.whl}"
MODEL_ENSURE_LOCAL_TF_WHEEL="${MODEL_ENSURE_LOCAL_TF_WHEEL:-true}"

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
MODEL_PATH="$ROOT_DIR/src/model.keras"
RUNNER_CHILD_PID=""

handle_interrupt() {
  trap - INT TERM
  echo "[session] interrupt received; stopping production session" | tee -a "$BOT_LOG" >&2
  if [[ -n "$RUNNER_CHILD_PID" ]] && kill -0 "$RUNNER_CHILD_PID" 2>/dev/null; then
    kill -INT -- "-$RUNNER_CHILD_PID" 2>/dev/null || kill -INT "$RUNNER_CHILD_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$RUNNER_CHILD_PID" 2>/dev/null; then
      kill -TERM -- "-$RUNNER_CHILD_PID" 2>/dev/null || kill -TERM "$RUNNER_CHILD_PID" 2>/dev/null || true
      sleep 1
      if kill -0 "$RUNNER_CHILD_PID" 2>/dev/null; then
        kill -KILL -- "-$RUNNER_CHILD_PID" 2>/dev/null || kill -KILL "$RUNNER_CHILD_PID" 2>/dev/null || true
      fi
    fi
  fi
  exit 130
}

trap handle_interrupt INT TERM

cat > "$META_FILE" <<EOF
{
  "session_tag": "${SESSION_TAG}",
  "run_id": "${RUN_ID}",
  "duration_hours": ${DURATION_HOURS},
  "timeout_kill_grace_seconds": ${TIMEOUT_KILL_GRACE_SECONDS},
  "python_bin": "${PYTHON_BIN}",
  "prepare_model": "${PREPARE_MODEL}",
  "prepare_model_strict": "${PREPARE_MODEL_STRICT}",
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
export MODEL_LOCAL_TF_WHEEL_PATH
export MODEL_ENSURE_LOCAL_TF_WHEEL
export MODEL_TRAIN_PYTHON="${MODEL_TRAIN_PYTHON:-$PYTHON_BIN}"

if [[ "$MODEL_ENSURE_LOCAL_TF_WHEEL" == "true" ]] && [[ ! -f "$MODEL_LOCAL_TF_WHEEL_PATH" ]]; then
  echo "[session] required local TensorFlow wheel not found: $MODEL_LOCAL_TF_WHEEL_PATH" | tee -a "$BOT_LOG" >&2
  echo "[session] run scripts/build_tensorflow_sm120.sh to build and stage local wheel" | tee -a "$BOT_LOG" >&2
  exit 1
fi

echo "[session] run_id=${RUN_ID}" | tee -a "$BOT_LOG"
echo "[session] bot_log=${BOT_LOG}" | tee -a "$BOT_LOG"
echo "[session] auto_model_retrain=${AUTO_MODEL_RETRAIN}" | tee -a "$BOT_LOG"
echo "[session] model_refresh_minutes=${MODEL_REFRESH_MINUTES}" | tee -a "$BOT_LOG"
echo "[session] model_train_days=${MODEL_TRAIN_DAYS}" | tee -a "$BOT_LOG"
echo "[session] model_train_python=${MODEL_TRAIN_PYTHON}" | tee -a "$BOT_LOG"
echo "[session] model_require_gpu=${MODEL_REQUIRE_GPU}" | tee -a "$BOT_LOG"
echo "[session] model_local_tf_wheel_path=${MODEL_LOCAL_TF_WHEEL_PATH}" | tee -a "$BOT_LOG"

if [[ "$PREPARE_MODEL" == "true" ]]; then
  MODEL_MTIME_BEFORE=0
  if [[ -f "$MODEL_PATH" ]]; then
    MODEL_MTIME_BEFORE="$(stat -c %Y "$MODEL_PATH" 2>/dev/null || echo 0)"
  fi

  echo "[session] running model warmup + training" | tee -a "$BOT_LOG"
  set +e
  MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}" \
  "$PYTHON_BIN" src/model.py --warmup-only >> "$BOT_LOG" 2>&1
  WARMUP_EXIT_CODE=$?
  if [[ "$WARMUP_EXIT_CODE" -eq 0 ]]; then
    MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}" \
    "$PYTHON_BIN" src/model.py --days "$MODEL_DAYS" --quiet >> "$BOT_LOG" 2>&1
  fi
  MODEL_PREP_EXIT_CODE=$?
  set -e

  if [[ "$MODEL_PREP_EXIT_CODE" -ne 0 ]]; then
    echo "[session] warning: model prep failed with exit code $MODEL_PREP_EXIT_CODE" | tee -a "$BOT_LOG"
    echo "[session] attempting fallback startup retrain with GPU required (warmup disabled)" | tee -a "$BOT_LOG"
    set +e
    MODEL_GPU_WARMUP=false \
    "$PYTHON_BIN" src/model.py --days "$MODEL_DAYS" --quiet >> "$BOT_LOG" 2>&1
    MODEL_PREP_EXIT_CODE=$?
    set -e
    if [[ "$MODEL_PREP_EXIT_CODE" -eq 0 ]]; then
      echo "[session] fallback startup retrain completed" | tee -a "$BOT_LOG"
    fi
  fi

  if [[ "$MODEL_PREP_EXIT_CODE" -eq 0 ]]; then
    if [[ -f "$MODEL_PATH" ]]; then
      MODEL_MTIME_AFTER="$(stat -c %Y "$MODEL_PATH" 2>/dev/null || echo 0)"
      if (( MODEL_MTIME_AFTER <= MODEL_MTIME_BEFORE )); then
        MODEL_PREP_EXIT_CODE=2
        echo "[session] warning: model prep finished but did not produce a newer model file" | tee -a "$BOT_LOG"
      else
        echo "[session] fresh_model_ready=true model_path=${MODEL_PATH}" | tee -a "$BOT_LOG"
        MODEL_MTIME_UTC="$(date -u -d "@${MODEL_MTIME_AFTER}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
        echo "[session] fresh_model_mtime_utc=${MODEL_MTIME_UTC}" | tee -a "$BOT_LOG"
      fi
    else
      MODEL_PREP_EXIT_CODE=3
      echo "[session] warning: model prep finished but model file is missing: ${MODEL_PATH}" | tee -a "$BOT_LOG"
    fi
  fi

  if [[ "$MODEL_PREP_EXIT_CODE" -ne 0 ]]; then
    if [[ "$PREPARE_MODEL_STRICT" == "true" ]]; then
      echo "[session] PREPARE_MODEL_STRICT=true, aborting session" | tee -a "$BOT_LOG"
      exit "$MODEL_PREP_EXIT_CODE"
    fi
    echo "[session] continuing bot run with existing model/runtime" | tee -a "$BOT_LOG"
  fi
fi

set +e
timeout -k "${TIMEOUT_KILL_GRACE_SECONDS}s" "${DURATION_HOURS}h" "$PYTHON_BIN" src/bot.py >> "$BOT_LOG" 2>&1 &
RUNNER_CHILD_PID=$!
wait "$RUNNER_CHILD_PID"
EXIT_CODE=$?
RUNNER_CHILD_PID=""
set -e

echo "$EXIT_CODE" > "$STATUS_FILE"
cat > "$SUMMARY_LOG" <<EOF
{
  "log": "${BOT_LOG}",
  "exit_code": ${EXIT_CODE},
  "timed_out": $([[ "$EXIT_CODE" -eq 124 ]] && echo true || echo false)
}
EOF

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
