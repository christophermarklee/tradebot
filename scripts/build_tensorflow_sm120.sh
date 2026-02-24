#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
TF_SRC_DIR="${TF_SRC_DIR:-$ROOT_DIR/.tensorflow-src}"
TF_REF="${TF_REF:-master}"
JOBS="${JOBS:-$(nproc)}"
CUDA_CC="${CUDA_CC:-12.0}"
HERMETIC_CUDA_VERSION_OVERRIDE="${HERMETIC_CUDA_VERSION_OVERRIDE:-}"
HERMETIC_CUDNN_VERSION_OVERRIDE="${HERMETIC_CUDNN_VERSION_OVERRIDE:-}"
HERMETIC_NCCL_VERSION_OVERRIDE="${HERMETIC_NCCL_VERSION_OVERRIDE:-}"
DO_SYNC="${DO_SYNC:-true}"
DO_BUILD="${DO_BUILD:-true}"
DO_INSTALL="${DO_INSTALL:-true}"
LOCAL_WHEEL_PATH="${LOCAL_WHEEL_PATH:-$ROOT_DIR/.wheels/tensorflow-2.22.0.dev0+selfbuilt-cp313-cp313-linux_x86_64.whl}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

normalize_semver3() {
  local v="$1"
  if [[ "$v" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "${v}.0"
  else
    echo "$v"
  fi
}

echo "[preflight] checking required commands"
need_cmd git
need_cmd curl
need_cmd sed
need_cmd awk

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python not found at $VENV_PYTHON" >&2
  echo "Run: uv venv && uv sync" >&2
  exit 1
fi

if ! command -v bazelisk >/dev/null 2>&1; then
  TOOLS_DIR="$ROOT_DIR/.tools"
  mkdir -p "$TOOLS_DIR"
  if [[ ! -x "$TOOLS_DIR/bazelisk" ]]; then
    echo "[preflight] downloading bazelisk"
    curl -fsSL -o "$TOOLS_DIR/bazelisk" https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x "$TOOLS_DIR/bazelisk"
  fi
  PATH="$TOOLS_DIR:$PATH"
fi

if [[ "$DO_SYNC" == "true" ]]; then
  if [[ -f "$LOCAL_WHEEL_PATH" ]]; then
    echo "[1/6] syncing project dependencies"
    (cd "$ROOT_DIR" && uv sync)
  else
    echo "[1/6] skipping initial uv sync (local TensorFlow wheel not staged yet: $LOCAL_WHEEL_PATH)"
  fi
fi

echo "[2/6] preparing tensorflow source"
if [[ -d "$TF_SRC_DIR/.git" ]]; then
  git -C "$TF_SRC_DIR" fetch --all --tags
  git -C "$TF_SRC_DIR" checkout "$TF_REF"
  git -C "$TF_SRC_DIR" pull --ff-only
else
  git clone https://github.com/tensorflow/tensorflow.git "$TF_SRC_DIR"
  git -C "$TF_SRC_DIR" checkout "$TF_REF"
fi

export PYTHON_BIN_PATH="$VENV_PYTHON"
export PYTHON_LIB_PATH="$($VENV_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
export TF_NEED_CUDA=1
export TF_NEED_ROCM=0
export TF_NEED_TENSORRT=0
export TF_ENABLE_XLA=1
export TF_CUDA_COMPUTE_CAPABILITIES="$CUDA_CC"
export HERMETIC_CUDA_COMPUTE_CAPABILITIES="$CUDA_CC"
export HERMETIC_PYTHON_VERSION="3.13"
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH="$(command -v gcc || true)"
export CLANG_CUDA_COMPILER_PATH="$(command -v clang || true)"
export CC_OPT_FLAGS="-Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

echo "[3/6] running tensorflow configure"
(cd "$TF_SRC_DIR" && \
  yes "" | "$VENV_PYTHON" configure.py || \
  [[ ${PIPESTATUS[0]} -eq 141 && ${PIPESTATUS[1]} -eq 0 ]])

if [[ "$DO_BUILD" == "true" ]]; then
  echo "[4/6] building tensorflow wheel (this may take a long time)"
  HERMETIC_CUDA_VERSION_OVERRIDE="$(normalize_semver3 "$HERMETIC_CUDA_VERSION_OVERRIDE")"
  HERMETIC_CUDNN_VERSION_OVERRIDE="$(normalize_semver3 "$HERMETIC_CUDNN_VERSION_OVERRIDE")"
  HERMETIC_NCCL_VERSION_OVERRIDE="$(normalize_semver3 "$HERMETIC_NCCL_VERSION_OVERRIDE")"

  BUILD_ARGS=(
    --config=opt
    --config=cuda_wheel
    --jobs="$JOBS"
    --repo_env=HERMETIC_PYTHON_VERSION=3.13
    --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="$CUDA_CC"
  )

  if [[ -n "$HERMETIC_CUDA_VERSION_OVERRIDE" ]]; then
    BUILD_ARGS+=(--repo_env=HERMETIC_CUDA_VERSION="$HERMETIC_CUDA_VERSION_OVERRIDE")
  fi
  if [[ -n "$HERMETIC_CUDNN_VERSION_OVERRIDE" ]]; then
    BUILD_ARGS+=(--repo_env=HERMETIC_CUDNN_VERSION="$HERMETIC_CUDNN_VERSION_OVERRIDE")
  fi
  if [[ -n "$HERMETIC_NCCL_VERSION_OVERRIDE" ]]; then
    BUILD_ARGS+=(--repo_env=HERMETIC_NCCL_VERSION="$HERMETIC_NCCL_VERSION_OVERRIDE")
  fi

  (cd "$TF_SRC_DIR" && bazelisk build "${BUILD_ARGS[@]}" //tensorflow/tools/pip_package:wheel)
else
  echo "[4/6] build skipped (DO_BUILD=false)"
fi

WHEEL_PATH=""
if [[ "$DO_BUILD" == "true" || "$DO_INSTALL" == "true" ]]; then
  WHEEL_PATH="$(cd "$TF_SRC_DIR" && ls -1 bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-*.whl 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$WHEEL_PATH" ]]; then
    echo "No TensorFlow wheel found at expected output path" >&2
    exit 1
  fi
fi

if [[ "$DO_INSTALL" == "true" ]]; then
  echo "[5/6] staging locally built wheel"
  mkdir -p "$(dirname "$LOCAL_WHEEL_PATH")"
  cp -f "$TF_SRC_DIR/$WHEEL_PATH" "$LOCAL_WHEEL_PATH"

  echo "[6/6] installing staged local wheel"
  (cd "$ROOT_DIR" && uv pip install --python "$VENV_PYTHON" --force-reinstall "$LOCAL_WHEEL_PATH")
  if [[ "$DO_SYNC" == "true" ]]; then
    (cd "$ROOT_DIR" && uv sync)
  fi
else
  echo "[5/6] staging locally built wheel"
  mkdir -p "$(dirname "$LOCAL_WHEEL_PATH")"
  cp -f "$TF_SRC_DIR/$WHEEL_PATH" "$LOCAL_WHEEL_PATH"
  echo "[6/6] install skipped (DO_INSTALL=false)"
fi

echo "Done"
if [[ -n "$WHEEL_PATH" ]]; then
  echo "Installed wheel: $LOCAL_WHEEL_PATH"
fi
echo "Verify with: uv run --python 3.13 python -c \"import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))\""
