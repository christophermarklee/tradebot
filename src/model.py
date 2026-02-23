"""
TensorFlow entry signal classifier — GPU-accelerated.

Training: Builds a labeled dataset from historical 1-minute bars, trains a deep
dense neural network on GPU with mixed-precision float16, and saves the full model
(including the Normalization layer) to src/model.keras.

Inference: Loads model.keras once via a module-level cache and returns the predicted
probability that the current bar is a good entry. Returns None gracefully when
TensorFlow is unavailable or no model file exists, allowing the caller to fall back
to the existing rule-based signal.

Feature vector (FEATURE_WINDOW + 4 values):
  - FEATURE_WINDOW normalized closes  (each relative to the current close)
  - SMA crossover ratio               (short/long SMA - 1, centered near 0)
  - RSI normalized to [0, 1]
  - Short-term momentum pct           (clipped to [-5, 5] then scaled to [-1, 1])
  - Short-term volatility             (std of last 20 bar returns)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "model.keras"

# Number of recent normalized closes included as positional price features.
# Larger window gives the model more price-history context.
FEATURE_WINDOW = 50

# Module-level inference cache — avoids reloading the model on every poll cycle
_cached_model = None
_cached_model_path: Optional[str] = None
_cached_model_mtime: Optional[float] = None


def configure_gpu() -> str:
    """Configure TensorFlow GPU: memory growth, mixed precision, XLA JIT.

    Returns a string describing the active device ('GPU:0 (mixed-precision)' or 'CPU').
    Silently no-ops if TensorFlow or cuDNN is unavailable.
    """
    try:
        import tensorflow as tf

        # Allow GPU memory to grow incrementally instead of claiming all VRAM up front
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if gpus:
            # Mixed precision: compute in float16 (fast on Tensor Cores),
            # accumulate gradients and store weights in float32 for stability
            from tensorflow.keras import mixed_precision  # type: ignore[attr-defined]
            mixed_precision.set_global_policy("mixed_float16")

            # XLA JIT : fuses kernel operations for additional GPU speedup
            tf.config.optimizer.set_jit(True)

            return f"GPU:0 (mixed-precision float16, XLA enabled, {len(gpus)} device(s))"

        return "CPU (no GPU detected)"
    except Exception as exc:  # pragma: no cover
        return f"CPU (GPU config error: {exc})"


def extract_features(
    closes: List[float],
    short_window: int = 9,
    long_window: int = 26,
    rsi_window: int = 14,
    momentum_lookback: int = 5,
) -> Optional[np.ndarray]:
    """Build a fixed-length feature vector from a list of close prices.

    Returns None if there is insufficient data.
    """
    needed = max(long_window, rsi_window + 1, momentum_lookback + 1, FEATURE_WINDOW)
    if len(closes) < needed:
        return None

    current = closes[-1]
    if current == 0.0:
        return None

    # Normalized recent closes (scale-invariant — each is a ratio to current price)
    norm_closes = [c / current for c in closes[-FEATURE_WINDOW:]]

    # SMA crossover ratio, centered near 0
    short_sma = sum(closes[-short_window:]) / short_window
    long_sma = sum(closes[-long_window:]) / long_window
    sma_ratio = (short_sma / long_sma) - 1.0

    # RSI normalized to [0, 1]
    gains = losses = 0.0
    for i in range(len(closes) - rsi_window, len(closes)):
        chg = closes[i] - closes[i - 1]
        if chg > 0:
            gains += chg
        else:
            losses += -chg
    rsi_val = 1.0 if losses == 0.0 else (100.0 - 100.0 / (1.0 + gains / losses)) / 100.0

    # Momentum pct clipped to [-5, 5] then normalized to [-1, 1]
    mom_base = closes[-1 - momentum_lookback]
    raw_mom = ((current / mom_base) - 1.0) * 100.0
    mom_norm = max(-5.0, min(5.0, raw_mom)) / 5.0

    # Volatility: std of last 20 bar-over-bar returns
    tail = closes[-21:] if len(closes) >= 21 else closes
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    volatility = float(np.std(rets)) if rets else 0.0

    return np.array(norm_closes + [sma_ratio, rsi_val, mom_norm, volatility], dtype=np.float32)


def build_training_data(
    bars: List[dict],
    target_pct: float = 0.5,
    max_horizon: int = 60,
    short_window: int = 9,
    long_window: int = 26,
    rsi_window: int = 14,
    momentum_lookback: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from a list of bar dicts (each with key "c" for close).

    Label y[i] = 1 if the maximum close in the next `max_horizon` bars is at least
    `target_pct`% above closes[i], else 0.
    """
    closes = [float(b["c"]) for b in bars]
    needed = max(long_window, rsi_window + 1, momentum_lookback + 1, FEATURE_WINDOW)

    X: List[np.ndarray] = []
    y: List[int] = []

    for i in range(needed, len(closes) - max_horizon):
        feats = extract_features(
            closes[: i + 1], short_window, long_window, rsi_window, momentum_lookback
        )
        if feats is None:
            continue
        target_price = closes[i] * (1.0 + target_pct / 100.0)
        future_high = max(closes[i + 1 : i + 1 + max_horizon])
        y.append(1 if future_high >= target_price else 0)
        X.append(feats)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(
    bars: List[dict],
    target_pct: float = 0.5,
    max_horizon: int = 60,
    short_window: int = 9,
    long_window: int = 26,
    rsi_window: int = 14,
    momentum_lookback: int = 5,
    model_path: Path = MODEL_PATH,
    quiet: bool = True,
) -> Dict[str, object]:
    """Train a new entry signal classifier on GPU and save it to model_path.

    Uses mixed-precision float16 on NVIDIA GPUs (Turing+) for faster
    Tensor Core throughput. The Normalization layer is adapted on training data
    and baked into the saved model file — no external scaler is needed at
    inference time.

    Raises ImportError if tensorflow is not installed.
    Returns a dict of training metrics.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        from tensorflow import keras
        from tensorflow.keras import layers  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "tensorflow is required for model training. "
            "Install it with: pip install tensorflow"
        ) from exc

    device_info = configure_gpu()

    X, y = build_training_data(
        bars, target_pct, max_horizon, short_window, long_window, rsi_window, momentum_lookback
    )

    if len(X) < 100:
        raise ValueError(
            f"Not enough training samples ({len(X)}). "
            "Increase MODEL_TRAIN_DAYS in env.json."
        )

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    if not quiet:
        print(f"[model] device={device_info} samples={len(X)} positive={pos_count} negative={neg_count}")

    # Inverse-frequency class weights to handle imbalanced labels
    class_weight = {0: 1.0, 1: float(neg_count) / max(pos_count, 1)}

    # Shuffle and split 80 / 20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    # Normalization baked into the model so it is saved in model.keras
    norm_layer = layers.Normalization(axis=-1)
    norm_layer.adapt(X_train)

    # Deeper network — the RTX 5090's Tensor Cores handle the wider layers
    # with negligible overhead; depth improves pattern recognition.
    # Output kernel uses float32 regardless of mixed-precision policy so
    # the sigmoid activation is numerically stable.
    model = keras.Sequential(
        [
            norm_layer,
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, restore_best_weights=True,
            mode="max", verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    # Large batch leverages VRAM bandwidth of the 5090; GPU keeps utilization high.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    val_auc = float(max(history.history.get("val_auc", [0.0])))
    val_loss = float(min(history.history.get("val_loss", [float("inf")])))
    epochs_run = len(history.history["loss"])

    model.save(str(model_path))

    # Refresh module-level cache so inference uses the new weights immediately
    global _cached_model, _cached_model_path, _cached_model_mtime
    _cached_model = model
    _cached_model_path = str(model_path)
    _cached_model_mtime = model_path.stat().st_mtime if model_path.exists() else None

    if not quiet:
        print(
            f"[model] saved → {model_path} | device={device_info} | "
            f"val_accuracy={val_acc:.4f} val_auc={val_auc:.4f} val_loss={val_loss:.4f} epochs={epochs_run}"
        )

    return {
        "samples": len(X),
        "positive_rate": round(pos_count / max(len(y), 1), 3),
        "val_accuracy": round(val_acc, 4),
        "val_auc": round(val_auc, 4),
        "val_loss": round(val_loss, 4),
        "epochs_trained": epochs_run,
        "device": device_info,
    }


def predict_entry(
    closes: List[float],
    short_window: int = 9,
    long_window: int = 26,
    rsi_window: int = 14,
    momentum_lookback: int = 5,
    model_path: Path = MODEL_PATH,
) -> Optional[float]:
    """Return the model's predicted entry probability, or None on any failure.

    None means: no model file, TensorFlow not installed, or insufficient data.
    The caller should fall back to rule-based logic when None is returned.
    """
    if not model_path.exists():
        return None

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf  # noqa: F401
    except ImportError:
        return None

    feats = extract_features(closes, short_window, long_window, rsi_window, momentum_lookback)
    if feats is None:
        return None

    try:
        global _cached_model, _cached_model_path, _cached_model_mtime
        model_mtime = model_path.stat().st_mtime
        if (
            _cached_model is None
            or _cached_model_path != str(model_path)
            or _cached_model_mtime != model_mtime
        ):
            from tensorflow import keras  # type: ignore[attr-defined]

            _cached_model = keras.models.load_model(str(model_path))
            _cached_model_path = str(model_path)
            _cached_model_mtime = model_mtime

        return float(_cached_model.predict(feats.reshape(1, -1), verbose=0)[0][0])
    except Exception:
        return None
