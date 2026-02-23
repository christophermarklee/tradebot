#!/usr/bin/env bash
set -euo pipefail

export MODEL_FORCE_PTX_JIT="${MODEL_FORCE_PTX_JIT:-true}"
export MODEL_GPU_WARMUP="${MODEL_GPU_WARMUP:-true}"
export MODEL_REQUIRE_GPU="${MODEL_REQUIRE_GPU:-true}"
export CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-0}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python interpreter not found at $PYTHON_BIN" >&2
	exit 1
fi

echo "[1/2] Warming up TensorFlow GPU kernels (PTX JIT compile/cache)..."
"$PYTHON_BIN" src/model.py --warmup-only

echo "[2/2] Running model training..."
"$PYTHON_BIN" src/model.py "$@"
