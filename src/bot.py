import os
import json
import time
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
    sweep_quiet: bool
    auto_sweep: bool
    sweep_refresh_minutes: int
    runtime_params_file: str
    model_confidence_threshold: float
    model_train_days: int


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
        sweep_quiet=get_env_bool("SWEEP_QUIET", True),
        auto_sweep=get_env_bool("AUTO_SWEEP", True),
        sweep_refresh_minutes=get_env_int("SWEEP_REFRESH_MINUTES", 240),
        runtime_params_file=get_config_value(
            "RUNTIME_PARAMS_FILE",
            str(SRC_DIR / "runtime_params.json"),
        ),
        model_confidence_threshold=get_env_float("MODEL_CONFIDENCE_THRESHOLD", 0.55),
        model_train_days=get_env_int("MODEL_TRAIN_DAYS", 30),
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

    def get_closes(self, limit: int) -> List[float]:
        url = f"{self.cfg.data_base_url}/v1beta3/crypto/us/bars"
        params = {
            "symbols": self.cfg.data_symbol,
            "timeframe": "1Min",
            "limit": limit,
        }
        response = requests.get(url, headers=self.data_headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        bars_by_symbol = data.get("bars", {})
        bars = bars_by_symbol.get(self.cfg.data_symbol, [])
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


def build_entry_signal(closes: List[float], cfg: Config) -> dict:
    needed = max(cfg.long_window, cfg.rsi_window + 1, cfg.momentum_lookback + 1)
    if len(closes) < needed:
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

    # Try model prediction — returns None when model is unavailable (graceful fallback)
    model_prob: Optional[float] = None
    try:
        try:
            import model as _model_module
        except ModuleNotFoundError:
            from src import model as _model_module  # type: ignore[no-redef]
        model_prob = _model_module.predict_entry(
            closes,
            short_window=cfg.short_window,
            long_window=cfg.long_window,
            rsi_window=cfg.rsi_window,
            momentum_lookback=cfg.momentum_lookback,
        )
    except Exception:
        model_prob = None

    enter = (
        model_prob >= cfg.model_confidence_threshold
        if model_prob is not None
        else rules_enter
    )

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
    }


def should_enter(closes: List[float], cfg: Config) -> bool:
    return bool(build_entry_signal(closes, cfg)["enter"])


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


def load_runtime_params(path: str) -> Dict[str, Any]:
    runtime_path = Path(path)
    if not runtime_path.exists():
        return {}
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("best_params"), dict):
        return dict(data["best_params"])
    if isinstance(data, dict):
        return data
    return {}


def apply_runtime_params(cfg: Config, params: Dict[str, Any]) -> None:
    float_map = {
        "RSI_MIN": "rsi_min",
        "RSI_MAX": "rsi_max",
        "MIN_MOMENTUM_PCT": "min_momentum_pct",
        "TAKE_PROFIT_BUFFER_USD": "take_profit_buffer_usd",
    }
    int_map = {
        "MOMENTUM_LOOKBACK": "momentum_lookback",
        "MAX_HOLD_MINUTES": "max_hold_minutes",
    }

    for key, field in float_map.items():
        if key in params:
            try:
                setattr(cfg, field, float(params[key]))
            except (TypeError, ValueError):
                continue

    for key, field in int_map.items():
        if key in params:
            try:
                setattr(cfg, field, int(float(params[key])))
            except (TypeError, ValueError):
                continue


def _maybe_retrain_model(cfg: Config, api: "AlpacaRest") -> None:
    """Fetch historical bars and retrain the TF entry signal model."""
    try:
        try:
            import model as _model_module
        except ModuleNotFoundError:
            from src import model as _model_module  # type: ignore[no-redef]
        emit("model_train_start", days=cfg.model_train_days)
        bars = api.get_historical_bars(cfg.model_train_days)
        if len(bars) < 200:
            emit("model_train_skip", reason="insufficient_data", bars=len(bars))
            return
        target_pct = (cfg.target_profit_usd / max(cfg.starting_balance_usd, 1.0)) * 100.0
        metrics = _model_module.train_model(
            bars=bars,
            target_pct=target_pct,
            max_horizon=cfg.max_hold_minutes,
            short_window=cfg.short_window,
            long_window=cfg.long_window,
            rsi_window=cfg.rsi_window,
            momentum_lookback=cfg.momentum_lookback,
            quiet=cfg.sweep_quiet,
        )
        emit("model_train_complete", **metrics)
    except Exception as err:
        emit("error", message=f"Model training failed: {err}")


def maybe_refresh_runtime_params(cfg: Config, now_ts: float, api: Optional["AlpacaRest"] = None) -> None:
    if not cfg.auto_sweep:
        return

    runtime_path = Path(cfg.runtime_params_file)
    refresh_seconds = max(cfg.sweep_refresh_minutes, 1) * 60
    if runtime_path.exists():
        age_seconds = now_ts - runtime_path.stat().st_mtime
        if age_seconds < refresh_seconds:
            return

    emit(
        "decision",
        action="auto_sweep_start",
        backtest_days=cfg.backtest_days,
        runtime_params_file=cfg.runtime_params_file,
    )
    try:
        try:
            import sweeps
        except ModuleNotFoundError:
            from src import sweeps

        best = sweeps.run_sweep(
            backtest_days=cfg.backtest_days,
            sweep_quiet=cfg.sweep_quiet,
            runtime_params_path=Path(cfg.runtime_params_file),
        )
        emit(
            "decision",
            action="auto_sweep_complete",
            score=round(float(best["score"]), 2),
            equity=round(float(best["equity"]), 2),
            pnl=round(float(best["pnl"]), 2),
        )
        if api is not None:
            _maybe_retrain_model(cfg, api)
    except Exception as err:
        emit("error", message=f"Auto sweep failed: {err}")


def run() -> None:
    cfg = load_config()
    api = AlpacaRest(cfg)
    global EMIT_PRETTY
    EMIT_PRETTY = cfg.log_pretty
    in_flight_side: Optional[str] = None
    position_opened_at: Optional[float] = None
    cooldown_until: float = 0.0
    runtime_signature = ""

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
        rsi_window=cfg.rsi_window,
        rsi_min=cfg.rsi_min,
        rsi_max=cfg.rsi_max,
        momentum_lookback=cfg.momentum_lookback,
        min_momentum_pct=cfg.min_momentum_pct,
        trade_log_file=cfg.trade_log_file,
        auto_sweep=cfg.auto_sweep,
        sweep_refresh_minutes=cfg.sweep_refresh_minutes,
        runtime_params_file=cfg.runtime_params_file,
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
                maybe_refresh_runtime_params(cfg, now_ts, api=api)

                runtime_params = load_runtime_params(cfg.runtime_params_file)
                if runtime_params:
                    apply_runtime_params(cfg, runtime_params)
                    current_signature = json.dumps(runtime_params, sort_keys=True)
                    if current_signature != runtime_signature:
                        runtime_signature = current_signature
                        emit(
                            "runtime_params_applied",
                            rsi_min=cfg.rsi_min,
                            rsi_max=cfg.rsi_max,
                            momentum_lookback=cfg.momentum_lookback,
                            min_momentum_pct=cfg.min_momentum_pct,
                            take_profit_buffer_usd=cfg.take_profit_buffer_usd,
                            max_hold_minutes=cfg.max_hold_minutes,
                            source=cfg.runtime_params_file,
                        )

                position = api.get_position()

                if in_flight_side == "buy":
                    if position is None:
                        emit("decision", action="wait_buy_fill")
                        time.sleep(cfg.poll_seconds)
                        continue
                    in_flight_side = None
                    position_opened_at = now_ts
                    emit("decision", action="buy_filled")

                if in_flight_side == "sell":
                    if position is not None:
                        emit("decision", action="wait_sell_fill")
                        time.sleep(cfg.poll_seconds)
                        continue
                    in_flight_side = None
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
                        api.submit_sell_qty(qty)
                        in_flight_side = "sell"
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

                    closes = api.get_closes(cfg.bar_limit)
                    if len(closes) == 0:
                        emit("error", message="No bars received from API")
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
                        signal_enter=enter_now,
                    )

                    if enter_now:
                        account = api.get_account()
                        cash = float(account.get("cash", 0.0))
                        # Use actual account cash (98% for safety margin)
                        order_notional = cash * 0.98

                        if order_notional > 10:
                            api.submit_buy_notional(order_notional)
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
                            emit(
                                "decision",
                                action="submit_buy",
                                order_notional=round(order_notional, 2),
                                approx_qty=round(approx_qty, 8),
                                entry_price=round(entry_price, 2),
                                cash=round(cash, 2),
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
        emit("shutdown", action="keyboard_interrupt_close_position")
        try:
            close_open_position(api)
            in_flight_side = None
        except Exception as err:
            emit("error", message=f"Failed to close position on shutdown: {err}")
        emit("shutdown", action="exit_clean")


if __name__ == "__main__":
    run()
