from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot import (  # noqa: E402
    AlpacaRest,
    Config,
    apply_runtime_params,
    get_config_value,
    load_config,
    load_runtime_params,
    should_enter,
)


def fetch_recent_bars(cfg: Config, days: int) -> List[Dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    url = f"{cfg.data_base_url}/v1beta3/crypto/us/bars"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": cfg.api_key,
        "APCA-API-SECRET-KEY": cfg.api_secret,
    }

    bars: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        params = {
            "symbols": cfg.data_symbol,
            "timeframe": "1Min",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "sort": "asc",
            "limit": 1000,
        }
        if page_token:
            params["page_token"] = page_token

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        symbol_bars = payload.get("bars", {}).get(cfg.data_symbol, [])
        bars.extend(symbol_bars)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    return bars


def analyze_bar_quality(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    if not bars:
        return {
            "total_bars": 0,
            "unique_close_count": 0,
            "max_same_close_run": 0,
            "same_close_adjacent_pairs": 0,
        }

    closes = [float(bar["c"]) for bar in bars]
    unique_close_count = len(set(closes))

    max_same_close_run = 1
    current_run = 1
    same_close_adjacent_pairs = 0
    for i in range(1, len(closes)):
        if closes[i] == closes[i - 1]:
            current_run += 1
            same_close_adjacent_pairs += 1
            if current_run > max_same_close_run:
                max_same_close_run = current_run
        else:
            current_run = 1

    return {
        "total_bars": len(bars),
        "unique_close_count": unique_close_count,
        "max_same_close_run": max_same_close_run,
        "same_close_adjacent_pairs": same_close_adjacent_pairs,
    }


def run_backtest() -> None:
    cfg = load_config()
    days = int(get_config_value("BACKTEST_DAYS", "3"))
    runtime_params_file = get_config_value("RUNTIME_PARAMS_FILE", "src/runtime_params.json")
    runtime_params = load_runtime_params(runtime_params_file)
    if runtime_params:
        apply_runtime_params(cfg, runtime_params)

    # Fetch actual account cash to use in backtest simulation
    try:
        api = AlpacaRest(cfg)
        account_info = api.get_account()
        starting_cash = float(account_info.get("cash", 0.0))
        if starting_cash == 0.0:
            print("Warning: Account cash is $0. Using configured starting_balance_usd.")
            starting_cash = cfg.starting_balance_usd
    except Exception as e:
        print(f"Warning: Could not fetch account cash ({e}). Using configured starting_balance_usd.")
        starting_cash = cfg.starting_balance_usd

    bars = fetch_recent_bars(cfg, days)
    if not bars:
        print("No bars returned for requested period. Check symbol and API credentials.")
        return

    quality = analyze_bar_quality(bars)
    max_stale_run_allowed = int(get_config_value("BACKTEST_MAX_STALE_RUN", "30"))
    if quality["max_same_close_run"] >= max_stale_run_allowed:
        raise RuntimeError(
            "Backtest aborted due to stale data quality: "
            f"max_same_close_run={int(quality['max_same_close_run'])} "
            f">= BACKTEST_MAX_STALE_RUN={max_stale_run_allowed}"
        )

    cash = starting_cash
    qty = 0.0
    entry_price = 0.0
    entry_notional = 0.0
    entry_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    closes: List[float] = []
    trades: List[Dict[str, Any]] = []
    max_equity_seen = starting_cash
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0

    for bar in bars:
        price = float(bar["c"])
        ts = bar["t"]
        ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        closes.append(price)

        if qty > 0:
            unrealized = (price - entry_price) * qty
            target_with_buffer = cfg.target_profit_usd + cfg.take_profit_buffer_usd

            should_exit = False
            reason = ""
            if unrealized >= target_with_buffer:
                should_exit = True
                reason = "take_profit"
            elif unrealized <= cfg.stop_loss_usd:
                should_exit = True
                reason = "stop_loss"
            elif (
                cfg.max_hold_minutes > 0
                and entry_time is not None
                and (ts_dt - entry_time).total_seconds() >= cfg.max_hold_minutes * 60
            ):
                should_exit = True
                reason = "max_hold"

            if should_exit:
                proceeds = qty * price
                pnl = proceeds - entry_notional
                cash += proceeds
                trades.append(
                    {
                        "side": "sell",
                        "time": ts,
                        "price": price,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
                qty = 0.0
                entry_price = 0.0
                entry_notional = 0.0
                entry_time = None
                cooldown_until = ts_dt + timedelta(minutes=cfg.cooldown_minutes)

            mark_to_market_equity = cash + (qty * price)
            if mark_to_market_equity > max_equity_seen:
                max_equity_seen = mark_to_market_equity
            drawdown_usd = max_equity_seen - mark_to_market_equity
            drawdown_pct = (drawdown_usd / max_equity_seen * 100.0) if max_equity_seen > 0 else 0.0
            if drawdown_usd > max_drawdown_usd:
                max_drawdown_usd = drawdown_usd
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
            continue

        if cooldown_until is not None and ts_dt < cooldown_until:
            mark_to_market_equity = cash
            if mark_to_market_equity > max_equity_seen:
                max_equity_seen = mark_to_market_equity
            drawdown_usd = max_equity_seen - mark_to_market_equity
            drawdown_pct = (drawdown_usd / max_equity_seen * 100.0) if max_equity_seen > 0 else 0.0
            if drawdown_usd > max_drawdown_usd:
                max_drawdown_usd = drawdown_usd
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
            continue

        if should_enter(closes, cfg):
            # Use actual available cash (98% for safety margin)
            notional = cash * 0.98
            if notional > 10:
                buy_qty = notional / price
                cash -= notional
                qty = buy_qty
                entry_price = price
                entry_notional = notional
                entry_time = ts_dt
                trades.append(
                    {
                        "side": "buy",
                        "time": ts,
                        "price": price,
                        "qty": qty,
                    }
                )

        mark_to_market_equity = cash + (qty * price)
        if mark_to_market_equity > max_equity_seen:
            max_equity_seen = mark_to_market_equity
        drawdown_usd = max_equity_seen - mark_to_market_equity
        drawdown_pct = (drawdown_usd / max_equity_seen * 100.0) if max_equity_seen > 0 else 0.0
        if drawdown_usd > max_drawdown_usd:
            max_drawdown_usd = drawdown_usd
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    if qty > 0:
        force_price = float(bars[-1]["c"])
        force_time = bars[-1]["t"]
        proceeds = qty * force_price
        pnl = proceeds - entry_notional
        cash += proceeds
        trades.append(
            {
                "side": "sell",
                "time": force_time,
                "price": force_price,
                "qty": qty,
                "pnl": pnl,
                "reason": "force_close_eod",
            }
        )
        qty = 0.0
        entry_price = 0.0
        entry_notional = 0.0
        entry_time = None

    last_price = float(bars[-1]["c"])
    open_value = qty * last_price
    equity = cash + open_value

    completed_trades = [t for t in trades if t["side"] == "sell"]
    realized_pnl = sum(t["pnl"] for t in completed_trades)

    print(f"=== Backtest: Last {days} Day(s) (1Min bars) ===")
    print(f"Symbol: {cfg.data_symbol}")
    print(f"Backtest days: {days}")
    print(f"Runtime params file: {runtime_params_file}")
    if runtime_params:
        print(
            "Runtime params applied: "
            f"RSI_MIN={cfg.rsi_min}, RSI_MAX={cfg.rsi_max}, "
            f"MOMENTUM_LOOKBACK={cfg.momentum_lookback}, "
            f"MIN_MOMENTUM_PCT={cfg.min_momentum_pct}, "
            f"TAKE_PROFIT_BUFFER_USD={cfg.take_profit_buffer_usd}, "
            f"MAX_HOLD_MINUTES={cfg.max_hold_minutes}"
        )
    else:
        print("Runtime params applied: none (using .env defaults)")
    print(f"Bars: {len(bars)}")
    print(
        "Bar quality: "
        f"unique_closes={int(quality['unique_close_count'])}, "
        f"same_close_adjacent_pairs={int(quality['same_close_adjacent_pairs'])}, "
        f"max_same_close_run={int(quality['max_same_close_run'])}"
    )
    print(f"Starting cash (from account): ${starting_cash:,.2f}")
    print(f"Ending equity: ${equity:,.2f}")
    print(f"Realized PnL: ${realized_pnl:,.2f}")
    print(f"Max drawdown (USD): ${max_drawdown_usd:,.2f}")
    print(f"Max drawdown (%): {max_drawdown_pct:.2f}")
    print(f"Round trips closed: {len(completed_trades)}")
    print(f"Open position qty: {qty:.8f}")
    print("Recent events:")
    for event in trades[-10:]:
        if event["side"] == "buy":
            print(
                f"  BUY  {event['time']} qty={event['qty']:.8f} "
                f"@ ${event['price']:.2f}"
            )
        else:
            print(
                f"  SELL {event['time']} qty={event['qty']:.8f} "
                f"@ ${event['price']:.2f} pnl=${event['pnl']:.2f} ({event.get('reason', 'n/a')})"
            )


if __name__ == "__main__":
    os.environ.setdefault("APCA_DATA_BASE_URL", "https://data.alpaca.markets")
    run_backtest()
