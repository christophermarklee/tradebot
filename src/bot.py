import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()

SRC_DIR = Path(__file__).resolve().parent

# Load environment defaults from env.json
ENV_JSON_PATH = SRC_DIR / "env.json"
ENV_DEFAULTS: Dict[str, str] = {}
if ENV_JSON_PATH.exists():
    with open(ENV_JSON_PATH, "r") as f:
        ENV_DEFAULTS = json.load(f)


def get_config_value(name: str, default: str = "") -> str:
    """Get configuration value from environment, falling back to env.json defaults."""
    return os.getenv(name, ENV_DEFAULTS.get(name, default))


@dataclass
class Config:
    api_key: str
    api_secret: str
    trade_base_url: str
    data_base_url: str
    trade_symbol: str
    data_symbol: str
    starting_balance_usd: float
    target_profit_usd: float
    stop_loss_usd: float
    take_profit_buffer_usd: float
    max_hold_minutes: int
    cooldown_minutes: int
    poll_seconds: int
    bar_limit: int
    short_window: int
    long_window: int
    rsi_window: int
    rsi_min: float
    rsi_max: float
    momentum_lookback: int
    min_momentum_pct: float
    entry_buffer_pct: float
    trade_log_file: str
    log_pretty: bool
    backtest_days: int
    model_confidence_threshold: float
    model_trend_slope_min: float
    model_high_volatility_threshold: float
    model_high_volatility_threshold_boost: float
    model_extreme_confidence_threshold: float
    model_drift_zscore_limit: float
    model_train_days: int
    auto_model_retrain: bool
    model_refresh_minutes: int
    model_train_python: str
    buy_fill_timeout_minutes: int
    sell_fill_timeout_minutes: int
    max_stale_polls: int
    max_bar_age_seconds: int
    stale_event_reset_after: int


def get_env_float(name: str, default: float) -> float:
    value = get_config_value(name)
    if not value:
        return default
    return float(value)


def get_env_int(name: str, default: int) -> int:
    value = get_config_value(name)
    if not value:
        return default
    return int(value)


def get_env_bool(name: str, default: bool) -> bool:
    value = get_config_value(name)
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_trade_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/v2"):
        return normalized[:-3]
    return normalized


def load_config() -> Config:
    # API credentials must be in environment variables (not in env.json)
    api_key = os.getenv("APCA_API_KEY_ID", "")
    api_secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not api_key or not api_secret:
        raise ValueError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")

    trade_base_url = normalize_trade_base_url(
        get_config_value("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    )

    return Config(
        api_key=api_key,
        api_secret=api_secret,
        trade_base_url=trade_base_url,
        data_base_url=get_config_value("APCA_DATA_BASE_URL", "https://data.alpaca.markets"),
        trade_symbol=get_config_value("TRADE_SYMBOL", "BTCUSD"),
        data_symbol=get_config_value("DATA_SYMBOL", "BTC/USD"),
        starting_balance_usd=get_env_float("STARTING_BALANCE_USD", 10000.0),
        target_profit_usd=get_env_float("TARGET_PROFIT_USD", 50.0),
        stop_loss_usd=get_env_float("STOP_LOSS_USD", -50.0),
        take_profit_buffer_usd=get_env_float("TAKE_PROFIT_BUFFER_USD", 10.0),
        max_hold_minutes=get_env_int("MAX_HOLD_MINUTES", 180),
        cooldown_minutes=get_env_int("COOLDOWN_MINUTES", 10),
        poll_seconds=get_env_int("POLL_SECONDS", 20),
        bar_limit=get_env_int("BAR_LIMIT", 60),
        short_window=get_env_int("SHORT_WINDOW", 9),
        long_window=get_env_int("LONG_WINDOW", 26),
        rsi_window=get_env_int("RSI_WINDOW", 14),
        rsi_min=get_env_float("RSI_MIN", 45.0),
        rsi_max=get_env_float("RSI_MAX", 70.0),
        momentum_lookback=get_env_int("MOMENTUM_LOOKBACK", 5),
        min_momentum_pct=get_env_float("MIN_MOMENTUM_PCT", 0.05),
        entry_buffer_pct=get_env_float("ENTRY_BUFFER_PCT", 0.001),
        trade_log_file=get_config_value("TRADE_LOG_FILE", "trade_log.csv"),
        log_pretty=get_env_bool("LOG_PRETTY", False),
        backtest_days=get_env_int("BACKTEST_DAYS", 3),
        model_confidence_threshold=get_env_float("MODEL_CONFIDENCE_THRESHOLD", 0.55),
        model_trend_slope_min=get_env_float("MODEL_TREND_SLOPE_MIN", 0.0002),
        model_high_volatility_threshold=get_env_float("MODEL_HIGH_VOLATILITY_THRESHOLD", 0.004),
        model_high_volatility_threshold_boost=get_env_float("MODEL_HIGH_VOLATILITY_THRESHOLD_BOOST", 0.05),
        model_extreme_confidence_threshold=get_env_float("MODEL_EXTREME_CONFIDENCE_THRESHOLD", 0.80),
        model_drift_zscore_limit=get_env_float("MODEL_DRIFT_ZSCORE_LIMIT", 3.0),
        model_train_days=get_env_int("MODEL_TRAIN_DAYS", 30),
        auto_model_retrain=get_env_bool("AUTO_MODEL_RETRAIN", True),
        model_refresh_minutes=get_env_int("MODEL_REFRESH_MINUTES", 240),
        model_train_python=get_config_value("MODEL_TRAIN_PYTHON", ".venv/bin/python"),
        buy_fill_timeout_minutes=get_env_int("BUY_FILL_TIMEOUT_MINUTES", 15),
        sell_fill_timeout_minutes=get_env_int("SELL_FILL_TIMEOUT_MINUTES", 15),
        max_stale_polls=get_env_int("MAX_STALE_POLLS", 4),
        max_bar_age_seconds=get_env_int("MAX_BAR_AGE_SECONDS", 180),
        stale_event_reset_after=get_env_int("STALE_EVENT_RESET_AFTER", 3),
    )


class AlpacaRest:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.trade_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "APCA-API-KEY-ID": cfg.api_key,
            "APCA-API-SECRET-KEY": cfg.api_secret,
        }
        self.data_headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": cfg.api_key,
            "APCA-API-SECRET-KEY": cfg.api_secret,
        }

    def get_account(self) -> dict:
        url = f"{self.cfg.trade_base_url}/v2/account"
        response = requests.get(url, headers=self.trade_headers, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_position(self) -> Optional[dict]:
        url = f"{self.cfg.trade_base_url}/v2/positions/{self.cfg.trade_symbol}"
        response = requests.get(url, headers=self.trade_headers, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_recent_bars(self, limit: int, end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        url = f"{self.cfg.data_base_url}/v1beta3/crypto/us/bars"
        params = {
            "symbols": self.cfg.data_symbol,
            "timeframe": "1Min",
            "limit": limit,
            "sort": "desc",
        }
        if end is not None:
            params["end"] = end.isoformat().replace("+00:00", "Z")
        response = requests.get(url, headers=self.data_headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        bars_by_symbol = data.get("bars", {})
        bars = bars_by_symbol.get(self.cfg.data_symbol, [])
        bars.reverse()
        return bars

    def get_closes(self, limit: int, end: Optional[datetime] = None) -> List[float]:
        bars = self.get_recent_bars(limit=limit, end=end)
        return [float(bar["c"]) for bar in bars]

    def get_historical_bars(self, days: int) -> List[Dict[str, Any]]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        url = f"{self.cfg.data_base_url}/v1beta3/crypto/us/bars"
        bars: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "symbols": self.cfg.data_symbol,
                "timeframe": "1Min",
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "sort": "asc",
                "limit": 1000,
            }
            if page_token:
                params["page_token"] = page_token
            response = requests.get(url, headers=self.data_headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            bars.extend(payload.get("bars", {}).get(self.cfg.data_symbol, []))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return bars

    def submit_buy_notional(self, notional_usd: float) -> dict:
        url = f"{self.cfg.trade_base_url}/v2/orders"
        payload = {
            "symbol": self.cfg.trade_symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "notional": round(notional_usd, 2),
        }
        response = requests.post(url, headers=self.trade_headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    def submit_sell_qty(self, qty: str) -> dict:
        url = f"{self.cfg.trade_base_url}/v2/orders"
        payload = {
            "symbol": self.cfg.trade_symbol,
            "side": "sell",
            "type": "market",
            "time_in_force": "gtc",
            "qty": qty,
        }
        response = requests.post(url, headers=self.trade_headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_order(self, order_id: str) -> Optional[dict]:
        url = f"{self.cfg.trade_base_url}/v2/orders/{order_id}"
        response = requests.get(url, headers=self.trade_headers, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def cancel_order(self, order_id: str) -> None:
        url = f"{self.cfg.trade_base_url}/v2/orders/{order_id}"
        response = requests.delete(url, headers=self.trade_headers, timeout=20)
        if response.status_code in (404, 422):
            return
        response.raise_for_status()


def close_open_position(api: AlpacaRest) -> None:
    position = api.get_position()
    if position is None:
        print("No open position to close.")
        return

    qty = position.get("qty", "0")
    if float(qty) <= 0:
        print("Position quantity is zero. Nothing to close.")
        return

    api.submit_sell_qty(qty)
    print(f"Submitted shutdown sell order for qty={qty}.")


def sma(values: List[float], window: int) -> float:
    chunk = values[-window:]
    return sum(chunk) / len(chunk)


def rsi(values: List[float], window: int) -> Optional[float]:
    if len(values) < window + 1:
        return None

    gains = 0.0
    losses = 0.0
    for i in range(len(values) - window, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains += change
        elif change < 0:
            losses += -change

    if losses == 0:
        return 100.0

    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_return(closes: List[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    return (closes[-1] / closes[-1 - lookback]) - 1.0


def _calc_ema(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (window + 1.0)
    ema_val = values[0]
    for value in values[1:]:
        ema_val = alpha * value + (1.0 - alpha) * ema_val
    return ema_val


def _trend_regime(closes: List[float]) -> Dict[str, float | str]:
    ret_5m = _calc_return(closes, 5)
    ret_15m = _calc_return(closes, 15)
    ret_60m = _calc_return(closes, 60)
    ema20 = _calc_ema(closes[-120:], 20)
    ema60 = _calc_ema(closes[-180:], 60)
    slope_ratio = (ema20 / max(ema60, 1e-9)) - 1.0
    tail = closes[-21:] if len(closes) >= 21 else closes
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    volatility = float(sum((r - (sum(rets) / max(len(rets), 1))) ** 2 for r in rets) / max(len(rets), 1)) ** 0.5 if rets else 0.0

    if ret_15m > 0 and slope_ratio > 0:
        regime = "bull"
    elif ret_15m < 0 and slope_ratio < 0:
        regime = "bear"
    else:
        regime = "chop"

    return {
        "ret_5m": ret_5m,
        "ret_15m": ret_15m,
        "ret_60m": ret_60m,
        "ema_slope_ratio": slope_ratio,
        "volatility": volatility,
        "regime": regime,
    }


def _dynamic_model_threshold(cfg: Config, trend: Dict[str, float | str]) -> float:
    threshold = cfg.model_confidence_threshold
    volatility = float(trend["volatility"])
    regime = str(trend["regime"])
    if volatility >= cfg.model_high_volatility_threshold:
        threshold += cfg.model_high_volatility_threshold_boost
    if regime == "bull":
        threshold -= 0.02
    elif regime == "bear":
        threshold += 0.03
    return max(0.05, min(0.98, threshold))


def build_entry_signal(closes: List[float], cfg: Config, use_model: bool = True, emit_events: bool = True) -> dict:
    needed = max(cfg.long_window, cfg.rsi_window + 1, cfg.momentum_lookback + 1, 61)
    if len(closes) < needed:
        if emit_events:
            emit(
                "data_insufficient",
                bars_received=len(closes),
                bars_needed=needed,
                message=f"Need {needed} bars, got {len(closes)}",
            )
        return {
            "enter": False,
            "enough_data": False,
            "short_sma": None,
            "long_sma": None,
            "trend_ok": False,
            "rsi": None,
            "rsi_ok": False,
            "momentum_pct": None,
            "momentum_ok": False,
            "model_probability": None,
            "model_used": False,
            "model_skip_reason": f"insufficient_rule_bars_need_{needed}_got_{len(closes)}",
            "dynamic_threshold": cfg.model_confidence_threshold,
            "trend_regime": "unknown",
            "trend_slope_15m": 0.0,
            "trend_volatility": 0.0,
            "drift_zscore": None,
        }

    short_sma = sma(closes, cfg.short_window)
    long_sma = sma(closes, cfg.long_window)
    trend_ok = short_sma > long_sma * (1 + cfg.entry_buffer_pct)

    rsi_value = rsi(closes, cfg.rsi_window)
    rsi_ok = (
        rsi_value is not None
        and rsi_value >= cfg.rsi_min
        and rsi_value <= cfg.rsi_max
    )

    momentum_base = closes[-1 - cfg.momentum_lookback]
    momentum_pct = ((closes[-1] / momentum_base) - 1.0) * 100.0
    momentum_ok = momentum_pct >= cfg.min_momentum_pct

    rules_enter = trend_ok and rsi_ok and momentum_ok
    trend = _trend_regime(closes)
    dynamic_threshold = _dynamic_model_threshold(cfg, trend)

    # Try model prediction — returns None when model is unavailable (graceful fallback)
    model_prob: Optional[float] = None
    model_skip_reason: Optional[str] = None
    drift_zscore: Optional[float] = None
    if use_model:
        try:
            try:
                import model as _model_module
            except ModuleNotFoundError:
                from src import model as _model_module  # type: ignore[no-redef]
            model_feature_window = int(getattr(_model_module, "FEATURE_WINDOW", 50))
            model_needed = max(cfg.long_window, cfg.rsi_window + 1, cfg.momentum_lookback + 1, model_feature_window)
            if len(closes) < model_needed:
                model_skip_reason = f"insufficient_model_bars_need_{model_needed}_got_{len(closes)}"
            else:
                if hasattr(_model_module, "predict_entry_with_diagnostics"):
                    diagnostics = _model_module.predict_entry_with_diagnostics(
                        closes,
                        short_window=cfg.short_window,
                        long_window=cfg.long_window,
                        rsi_window=cfg.rsi_window,
                        momentum_lookback=cfg.momentum_lookback,
                    )
                    raw_prob = diagnostics.get("probability")
                    model_prob = float(raw_prob) if isinstance(raw_prob, (float, int)) else None
                    raw_drift = diagnostics.get("drift_zscore")
                    drift_zscore = float(raw_drift) if isinstance(raw_drift, (float, int)) else None
                    raw_reason = diagnostics.get("reason")
                    if isinstance(raw_reason, str) and raw_reason != "ok":
                        model_skip_reason = raw_reason
                else:
                    model_prob = _model_module.predict_entry(
                        closes,
                        short_window=cfg.short_window,
                        long_window=cfg.long_window,
                        rsi_window=cfg.rsi_window,
                        momentum_lookback=cfg.momentum_lookback,
                    )
                if model_prob is None and model_skip_reason is None:
                    model_skip_reason = "predict_entry_returned_none"
        except Exception as err:
            model_prob = None
            model_skip_reason = f"predict_entry_exception_{type(err).__name__}"
    else:
        model_skip_reason = "model_disabled"

    if drift_zscore is not None and drift_zscore > cfg.model_drift_zscore_limit:
        model_prob = None
        model_skip_reason = f"drift_zscore_{drift_zscore:.2f}_gt_{cfg.model_drift_zscore_limit:.2f}"

    if model_prob is not None:
        regime = str(trend["regime"])
        slope_15m = float(trend["ret_15m"])
        regime_ok = regime != "bear" or model_prob >= cfg.model_extreme_confidence_threshold
        slope_ok = slope_15m >= cfg.model_trend_slope_min or model_prob >= cfg.model_extreme_confidence_threshold
        enter = model_prob >= dynamic_threshold and regime_ok and slope_ok
    else:
        enter = rules_enter

    return {
        "enter": enter,
        "enough_data": True,
        "short_sma": short_sma,
        "long_sma": long_sma,
        "trend_ok": trend_ok,
        "rsi": rsi_value,
        "rsi_ok": rsi_ok,
        "momentum_pct": momentum_pct,
        "momentum_ok": momentum_ok,
        "model_probability": model_prob,
        "model_used": model_prob is not None,
        "model_skip_reason": model_skip_reason,
        "dynamic_threshold": dynamic_threshold,
        "trend_regime": trend["regime"],
        "trend_slope_15m": trend["ret_15m"],
        "trend_volatility": trend["volatility"],
        "drift_zscore": drift_zscore,
    }


def should_enter(closes: List[float], cfg: Config, use_model: bool = True, emit_events: bool = True) -> bool:
    return bool(build_entry_signal(closes, cfg, use_model=use_model, emit_events=emit_events)["enter"])


def append_trade_log(
    cfg: Config,
    side: str,
    qty: float,
    price: float,
    pnl: float,
    reason: str,
) -> None:
    header = "timestamp_utc,side,qty,price,pnl,reason\n"
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    line = f"{ts},{side},{qty:.8f},{price:.2f},{pnl:.2f},{reason}\n"

    if not os.path.exists(cfg.trade_log_file):
        with open(cfg.trade_log_file, "w", encoding="utf-8") as file:
            file.write(header)

    with open(cfg.trade_log_file, "a", encoding="utf-8") as file:
        file.write(line)


def emit(event: str, **fields: object) -> None:
    if EMIT_PRETTY:
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[{event}] {details}")
        else:
            print(f"[{event}]")
        return

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    print(json.dumps(payload))


EMIT_PRETTY = False


def _maybe_retrain_model(cfg: Config) -> None:
    """Retrain TF entry model via dedicated ML Python interpreter."""
    try:
        train_python = cfg.model_train_python.strip()
        if not train_python:
            emit("model_train_skip", reason="empty_model_train_python")
            return

        train_python_path = Path(train_python)
        if train_python_path.is_absolute():
            python_cmd = train_python_path
        else:
            python_cmd = SRC_DIR.parent / train_python_path

        if not python_cmd.exists():
            emit(
                "model_train_skip",
                reason="model_train_python_not_found",
                model_train_python=str(python_cmd),
            )
            return

        command = [str(python_cmd), str(SRC_DIR / "model.py"), "--days", str(cfg.model_train_days), "--quiet"]

        emit(
            "model_train_start",
            days=cfg.model_train_days,
            model_train_python=str(python_cmd),
        )

        result = subprocess.run(
            command,
            cwd=str(SRC_DIR.parent),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )

        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            if not message:
                message = f"trainer exited with code {result.returncode}"
            emit("error", message=f"Model training failed: {message}")
            return

        model_path = SRC_DIR / "model.keras"
        emit(
            "model_train_complete",
            model_path=str(model_path),
            model_exists=model_path.exists(),
            model_mtime_utc=(
                datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if model_path.exists()
                else None
            ),
        )
    except Exception as err:
        emit("error", message=f"Model training failed: {err}")


def maybe_refresh_model(cfg: Config, now_ts: float) -> None:
    if not cfg.auto_model_retrain:
        return

    model_path = SRC_DIR / "model.keras"
    refresh_seconds = max(cfg.model_refresh_minutes, 1) * 60
    if model_path.exists():
        age_seconds = now_ts - model_path.stat().st_mtime
        if age_seconds < refresh_seconds:
            return

    emit(
        "decision",
        action="auto_model_retrain_start",
        model_path=str(model_path),
        model_refresh_minutes=cfg.model_refresh_minutes,
        model_train_days=cfg.model_train_days,
    )
    _maybe_retrain_model(cfg)


def run() -> None:
    cfg = load_config()
    api = AlpacaRest(cfg)
    global EMIT_PRETTY
    EMIT_PRETTY = cfg.log_pretty
    in_flight_side: Optional[str] = None
    position_opened_at: Optional[float] = None
    cooldown_until: float = 0.0
    last_seen_bar_ts: Optional[datetime] = None
    stale_poll_count = 0
    stale_event_count = 0
    in_flight_order_id: Optional[str] = None
    in_flight_submitted_at: Optional[float] = None

    # Fetch actual account cash at startup
    try:
        account_info = api.get_account()
        account_cash = float(account_info.get("cash", 0.0))
        account_equity = float(account_info.get("equity", 0.0))
    except Exception as e:
        emit("error", message=f"Failed to fetch account info at startup: {e}")
        account_cash = 0.0
        account_equity = 0.0

    emit(
        "startup",
        symbol=cfg.trade_symbol,
        target_profit_usd=cfg.target_profit_usd,
        account_cash_usd=round(account_cash, 2),
        account_equity_usd=round(account_equity, 2),
        stop_loss_usd=cfg.stop_loss_usd,
        take_profit_buffer_usd=cfg.take_profit_buffer_usd,
        max_hold_minutes=cfg.max_hold_minutes,
        cooldown_minutes=cfg.cooldown_minutes,
        bar_limit=cfg.bar_limit,
        rsi_window=cfg.rsi_window,
        rsi_min=cfg.rsi_min,
        rsi_max=cfg.rsi_max,
        momentum_lookback=cfg.momentum_lookback,
        min_momentum_pct=cfg.min_momentum_pct,
        max_stale_polls=cfg.max_stale_polls,
        max_bar_age_seconds=cfg.max_bar_age_seconds,
        stale_event_reset_after=cfg.stale_event_reset_after,
        trade_log_file=cfg.trade_log_file,
        auto_model_retrain=cfg.auto_model_retrain,
        model_refresh_minutes=cfg.model_refresh_minutes,
        model_train_python=cfg.model_train_python,
        model_min_bars_required=max(cfg.long_window, cfg.rsi_window + 1, cfg.momentum_lookback + 1, 50),
    )

    # Warmup period: wait for sufficient historical data
    warmup_needed = max(cfg.long_window, cfg.rsi_window + 1, cfg.momentum_lookback + 1)
    emit("warmup", message=f"Waiting for {warmup_needed} bars of data before trading...")
    warmup_attempts = 0
    max_warmup_attempts = 10
    while warmup_attempts < max_warmup_attempts:
        try:
            closes = api.get_closes(cfg.bar_limit)
            if len(closes) >= warmup_needed:
                emit("warmup_complete", bars_available=len(closes), bars_needed=warmup_needed)
                break
            emit("warmup_progress", bars_available=len(closes), bars_needed=warmup_needed, attempt=warmup_attempts + 1)
            warmup_attempts += 1
            time.sleep(30)
        except Exception as err:
            emit("warmup_error", message=str(err), attempt=warmup_attempts + 1)
            warmup_attempts += 1
            time.sleep(30)
    
    if warmup_attempts >= max_warmup_attempts:
        emit("warmup_failed", message="Failed to get sufficient data after warmup period")

    try:
        while True:
            try:
                now_ts = time.time()
                maybe_refresh_model(cfg, now_ts)

                position = api.get_position()

                if in_flight_side == "buy":
                    if position is None:
                        order_status = "unknown"
                        if in_flight_order_id:
                            order_data = api.get_order(in_flight_order_id)
                            if order_data is not None:
                                order_status = str(order_data.get("status", "unknown")).lower()
                            else:
                                order_status = "missing"

                        buy_timeout_seconds = max(cfg.buy_fill_timeout_minutes, 1) * 60
                        buy_fill_wait_seconds = (
                            int(now_ts - in_flight_submitted_at)
                            if in_flight_submitted_at is not None
                            else 0
                        )

                        if order_status in {"canceled", "rejected", "expired", "suspended", "missing"}:
                            emit(
                                "error",
                                message=(
                                    "Buy order did not fill and is no longer active "
                                    f"(status={order_status}); resetting in-flight state"
                                ),
                            )
                            in_flight_side = None
                            in_flight_order_id = None
                            in_flight_submitted_at = None
                            time.sleep(cfg.poll_seconds)
                            continue

                        if buy_fill_wait_seconds >= buy_timeout_seconds:
                            if in_flight_order_id:
                                try:
                                    api.cancel_order(in_flight_order_id)
                                except Exception as cancel_err:
                                    emit("error", message=f"Failed to cancel stale buy order: {cancel_err}")
                            emit(
                                "decision",
                                action="buy_fill_timeout_reset",
                                wait_seconds=buy_fill_wait_seconds,
                                timeout_seconds=buy_timeout_seconds,
                                order_status=order_status,
                            )
                            in_flight_side = None
                            in_flight_order_id = None
                            in_flight_submitted_at = None
                            time.sleep(cfg.poll_seconds)
                            continue

                        emit(
                            "decision",
                            action="wait_buy_fill",
                            wait_seconds=buy_fill_wait_seconds,
                            timeout_seconds=buy_timeout_seconds,
                            order_status=order_status,
                        )
                        time.sleep(cfg.poll_seconds)
                        continue
                    in_flight_side = None
                    in_flight_order_id = None
                    in_flight_submitted_at = None
                    position_opened_at = now_ts
                    emit("decision", action="buy_filled")

                if in_flight_side == "sell":
                    if position is not None:
                        order_status = "unknown"
                        if in_flight_order_id:
                            order_data = api.get_order(in_flight_order_id)
                            if order_data is not None:
                                order_status = str(order_data.get("status", "unknown")).lower()
                            else:
                                order_status = "missing"

                        sell_timeout_seconds = max(cfg.sell_fill_timeout_minutes, 1) * 60
                        sell_fill_wait_seconds = (
                            int(now_ts - in_flight_submitted_at)
                            if in_flight_submitted_at is not None
                            else 0
                        )

                        if sell_fill_wait_seconds >= sell_timeout_seconds:
                            if in_flight_order_id:
                                try:
                                    api.cancel_order(in_flight_order_id)
                                except Exception as cancel_err:
                                    emit("error", message=f"Failed to cancel stale sell order: {cancel_err}")
                            emit(
                                "decision",
                                action="sell_fill_timeout_reset",
                                wait_seconds=sell_fill_wait_seconds,
                                timeout_seconds=sell_timeout_seconds,
                                order_status=order_status,
                            )
                            in_flight_side = None
                            in_flight_order_id = None
                            in_flight_submitted_at = None
                            cooldown_until = now_ts + (cfg.cooldown_minutes * 60)
                            time.sleep(cfg.poll_seconds)
                            continue

                        emit(
                            "decision",
                            action="wait_sell_fill",
                            wait_seconds=sell_fill_wait_seconds,
                            timeout_seconds=sell_timeout_seconds,
                            order_status=order_status,
                        )
                        time.sleep(cfg.poll_seconds)
                        continue
                    in_flight_side = None
                    in_flight_order_id = None
                    in_flight_submitted_at = None
                    cooldown_until = now_ts + (cfg.cooldown_minutes * 60)
                    position_opened_at = None
                    emit("decision", action="sell_filled_cooldown_started")

                if position is not None:
                    unrealized = float(position.get("unrealized_pl", 0.0))
                    qty = position.get("qty", "0")
                    qty_float = float(qty)
                    current_price = float(position.get("current_price", 0.0))
                    if position_opened_at is None:
                        position_opened_at = now_ts

                    hold_minutes = int((now_ts - position_opened_at) / 60)
                    target_with_buffer = cfg.target_profit_usd + cfg.take_profit_buffer_usd

                    emit(
                        "position_data",
                        qty=qty_float,
                        current_price=round(current_price, 2),
                        unrealized_pl=round(unrealized, 2),
                        hold_minutes=hold_minutes,
                        target_with_buffer=round(target_with_buffer, 2),
                        stop_loss=round(cfg.stop_loss_usd, 2),
                    )

                    exit_reason = ""
                    if unrealized >= target_with_buffer:
                        exit_reason = "take_profit"
                    elif unrealized <= cfg.stop_loss_usd:
                        exit_reason = "stop_loss"
                    elif cfg.max_hold_minutes > 0 and hold_minutes >= cfg.max_hold_minutes:
                        exit_reason = "max_hold"

                    if exit_reason:
                        sell_order = api.submit_sell_qty(qty)
                        in_flight_side = "sell"
                        in_flight_order_id = str(sell_order.get("id", "")) or None
                        in_flight_submitted_at = now_ts
                        append_trade_log(
                            cfg=cfg,
                            side="sell",
                            qty=qty_float,
                            price=current_price,
                            pnl=unrealized,
                            reason=exit_reason,
                        )
                        emit(
                            "decision",
                            action="submit_sell",
                            reason=exit_reason,
                            qty=qty_float,
                            price=round(current_price, 2),
                            unrealized_pl=round(unrealized, 2),
                        )
                    else:
                        emit("decision", action="hold_position")
                else:
                    if now_ts < cooldown_until:
                        wait_seconds = int(cooldown_until - now_ts)
                        emit("decision", action="cooldown_wait", seconds_remaining=wait_seconds)
                        time.sleep(cfg.poll_seconds)
                        continue

                    bars = api.get_recent_bars(
                        limit=cfg.bar_limit,
                        end=datetime.now(timezone.utc),
                    )
                    if len(bars) == 0:
                        emit("error", message="No bars received from API")
                        time.sleep(cfg.poll_seconds)
                        continue

                    closes = [float(bar["c"]) for bar in bars]
                    latest_bar_ts = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
                    bar_age_seconds = int((datetime.now(timezone.utc) - latest_bar_ts).total_seconds())

                    stale_reasons: List[str] = []
                    if last_seen_bar_ts is not None and latest_bar_ts <= last_seen_bar_ts:
                        stale_poll_count += 1
                        stale_reasons.append("latest_bar_not_advanced")
                    else:
                        stale_poll_count = 0
                        last_seen_bar_ts = latest_bar_ts

                    if bar_age_seconds > cfg.max_bar_age_seconds:
                        stale_reasons.append("latest_bar_too_old")

                    if stale_reasons and stale_poll_count >= cfg.max_stale_polls:
                        stale_event_count += 1
                        emit(
                            "market_data_stale",
                            reasons=stale_reasons,
                            stale_poll_count=stale_poll_count,
                            stale_event_count=stale_event_count,
                            latest_bar_utc=latest_bar_ts.isoformat().replace("+00:00", "Z"),
                            bar_age_seconds=bar_age_seconds,
                            max_stale_polls=cfg.max_stale_polls,
                            max_bar_age_seconds=cfg.max_bar_age_seconds,
                            stale_event_reset_after=cfg.stale_event_reset_after,
                        )

                        if stale_event_count >= max(cfg.stale_event_reset_after, 1):
                            api = AlpacaRest(cfg)
                            last_seen_bar_ts = None
                            stale_poll_count = 0
                            stale_event_count = 0
                            emit(
                                "data_client_reset",
                                reason="stale_market_data_threshold",
                                action="recreate_alpaca_client",
                            )

                        emit("decision", action="wait_stale_market_data")
                        time.sleep(cfg.poll_seconds)
                        continue
                    
                    last_price = closes[-1]
                    signal = build_entry_signal(closes, cfg)
                    enter_now = bool(signal["enter"])
                    emit(
                        "market_data",
                        last_price=round(last_price, 2),
                        short_sma=round(signal["short_sma"], 2) if signal["short_sma"] is not None else None,
                        long_sma=round(signal["long_sma"], 2) if signal["long_sma"] is not None else None,
                        trend_ok=signal["trend_ok"],
                        rsi=round(signal["rsi"], 2) if signal["rsi"] is not None else None,
                        rsi_ok=signal["rsi_ok"],
                        momentum_pct=round(signal["momentum_pct"], 4) if signal["momentum_pct"] is not None else None,
                        momentum_ok=signal["momentum_ok"],
                        enough_data=signal["enough_data"],
                        model_probability=round(signal["model_probability"], 4) if signal.get("model_probability") is not None else None,
                        model_used=signal.get("model_used", False),
                        model_skip_reason=signal.get("model_skip_reason"),
                        dynamic_threshold=round(float(signal.get("dynamic_threshold", cfg.model_confidence_threshold)), 4),
                        trend_regime=signal.get("trend_regime"),
                        trend_slope_15m=round(float(signal.get("trend_slope_15m", 0.0)), 6),
                        trend_volatility=round(float(signal.get("trend_volatility", 0.0)), 6),
                        drift_zscore=round(float(signal["drift_zscore"]), 3) if signal.get("drift_zscore") is not None else None,
                        latest_bar_utc=latest_bar_ts.isoformat().replace("+00:00", "Z"),
                        bar_age_seconds=bar_age_seconds,
                        stale_poll_count=stale_poll_count,
                        signal_enter=enter_now,
                    )

                    if enter_now and bar_age_seconds > cfg.max_bar_age_seconds:
                        emit(
                            "decision",
                            action="skip_buy_stale_bar",
                            bar_age_seconds=bar_age_seconds,
                            max_bar_age_seconds=cfg.max_bar_age_seconds,
                            latest_bar_utc=latest_bar_ts.isoformat().replace("+00:00", "Z"),
                        )
                        time.sleep(cfg.poll_seconds)
                        continue

                    if enter_now:
                        account = api.get_account()
                        cash = float(account.get("cash", 0.0))
                        # Use actual account cash (98% for safety margin)
                        order_notional = cash * 0.98

                        if order_notional > 10:
                            buy_order = api.submit_buy_notional(order_notional)
                            entry_price = closes[-1]
                            approx_qty = order_notional / entry_price
                            append_trade_log(
                                cfg=cfg,
                                side="buy",
                                qty=approx_qty,
                                price=entry_price,
                                pnl=0.0,
                                reason="signal",
                            )
                            in_flight_side = "buy"
                            in_flight_order_id = str(buy_order.get("id", "")) or None
                            in_flight_submitted_at = now_ts
                            emit(
                                "decision",
                                action="submit_buy",
                                order_notional=round(order_notional, 2),
                                approx_qty=round(approx_qty, 8),
                                entry_price=round(entry_price, 2),
                                cash=round(cash, 2),
                                order_id=in_flight_order_id,
                            )
                        else:
                            emit(
                                "decision",
                                action="skip_buy_low_cash",
                                cash=round(cash, 2),
                                computed_notional=round(order_notional, 2),
                            )
                    else:
                        emit("decision", action="wait_no_signal")

            except Exception as err:
                emit("error", message=str(err))

            time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        try:
            emit("shutdown", action="keyboard_interrupt_close_position")
        except KeyboardInterrupt:
            pass
        try:
            close_open_position(api)
            in_flight_side = None
        except KeyboardInterrupt:
            pass
        except Exception as err:
            try:
                emit("error", message=f"Failed to close position on shutdown: {err}")
            except KeyboardInterrupt:
                pass
        try:
            emit("shutdown", action="exit_clean")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run()
