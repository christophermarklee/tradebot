from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent
BACKTEST_PATH = ROOT / "tests" / "backtest.py"
DEFAULT_RUNTIME_PARAMS_PATH = ROOT / "runtime_params.json"


def run_backtest_with_env(overrides: Dict[str, str]) -> Optional[Dict[str, float]]:
    env = os.environ.copy()
    env.update(overrides)

    try:
        result = subprocess.run(
            [sys.executable, str(BACKTEST_PATH)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        return {
            "error": f"Backtest failed: {exc.stderr.strip() or exc.stdout.strip()}",
            "output": exc.stdout or "",
        }

    output = result.stdout
    equity_match = re.search(r"Ending equity:\s*\$([0-9,]+\.[0-9]+)", output)
    pnl_match = re.search(r"Realized PnL:\s*\$(-?[0-9,]+\.[0-9]+)", output)
    dd_match = re.search(r"Max drawdown \(USD\):\s*\$([0-9,]+\.[0-9]+)", output)
    trades_match = re.search(r"Round trips closed:\s*(\d+)", output)

    if not equity_match or not pnl_match or not dd_match or not trades_match:
        raise ValueError(f"Could not parse backtest output:\n{output}")

    equity = float(equity_match.group(1).replace(",", ""))
    pnl = float(pnl_match.group(1).replace(",", ""))
    max_drawdown_usd = float(dd_match.group(1).replace(",", ""))
    trades = int(trades_match.group(1))
    score = equity + (trades * 5.0) - (max_drawdown_usd * 0.25)

    return {
        "equity": equity,
        "pnl": pnl,
        "max_drawdown_usd": max_drawdown_usd,
        "trades": trades,
        "score": score,
        "output": output,
    }


def build_grid() -> List[Dict[str, str]]:
    rsi_min_values = [46, 48, 50]
    rsi_max_values = [66, 68]
    momentum_lookback_values = [5, 6, 8]
    min_momentum_pct_values = [0.05, 0.07, 0.08]
    take_profit_buffer_values = [12, 15]
    max_hold_values = [120, 150]

    configs: List[Dict[str, str]] = []
    for rsi_min, rsi_max, lookback, min_mom, buffer_usd, max_hold in itertools.product(
        rsi_min_values,
        rsi_max_values,
        momentum_lookback_values,
        min_momentum_pct_values,
        take_profit_buffer_values,
        max_hold_values,
    ):
        if rsi_min >= rsi_max:
            continue
        configs.append(
            {
                "RSI_MIN": str(rsi_min),
                "RSI_MAX": str(rsi_max),
                "MOMENTUM_LOOKBACK": str(lookback),
                "MIN_MOMENTUM_PCT": str(min_mom),
                "TAKE_PROFIT_BUFFER_USD": str(buffer_usd),
                "MAX_HOLD_MINUTES": str(max_hold),
            }
        )

    return configs


def write_runtime_params(path: Path, best: Dict[str, object], backtest_days: int) -> None:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backtest_days": backtest_days,
        "best_params": {
            "RSI_MIN": best["RSI_MIN"],
            "RSI_MAX": best["RSI_MAX"],
            "MOMENTUM_LOOKBACK": best["MOMENTUM_LOOKBACK"],
            "MIN_MOMENTUM_PCT": best["MIN_MOMENTUM_PCT"],
            "TAKE_PROFIT_BUFFER_USD": best["TAKE_PROFIT_BUFFER_USD"],
            "MAX_HOLD_MINUTES": best["MAX_HOLD_MINUTES"],
        },
        "metrics": {
            "equity": round(float(best["equity"]), 2),
            "pnl": round(float(best["pnl"]), 2),
            "max_drawdown_usd": round(float(best["max_drawdown_usd"]), 2),
            "trades": int(best["trades"]),
            "score": round(float(best["score"]), 2),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sweep(
    backtest_days: int = 3,
    sweep_quiet: bool = True,
    runtime_params_path: Path | str = DEFAULT_RUNTIME_PARAMS_PATH,
) -> Dict[str, object]:
    configs = build_grid()
    results: List[Dict[str, object]] = []

    if not sweep_quiet:
        print(f"Running {len(configs)} backtests (BACKTEST_DAYS={backtest_days})...")

    for index, cfg in enumerate(configs, start=1):
        metrics = run_backtest_with_env({**cfg, "BACKTEST_DAYS": str(backtest_days)})
        if not metrics or "error" in metrics:
            if not sweep_quiet:
                print(f"[{index}/{len(configs)}] failed | {cfg}")
            continue

        row = {**cfg, **metrics}
        results.append(row)

        if not sweep_quiet:
            print(
                f"[{index}/{len(configs)}] equity=${metrics['equity']:.2f}, "
                f"pnl=${metrics['pnl']:.2f}, dd=${metrics['max_drawdown_usd']:.2f}, "
                f"trades={metrics['trades']}, score={metrics['score']:.2f} | {cfg}"
            )
        elif index % 20 == 0 or index == len(configs):
            print(f"Progress: {index}/{len(configs)}")

    if not results:
        raise RuntimeError("Sweep completed but no successful backtests were returned.")

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    best = results[0]

    runtime_path = Path(runtime_params_path)
    write_runtime_params(runtime_path, best, backtest_days)

    return best


def main() -> None:
    backtest_days = int(os.getenv("BACKTEST_DAYS", "3"))
    sweep_quiet = os.getenv("SWEEP_QUIET", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    best = run_sweep(backtest_days=backtest_days, sweep_quiet=sweep_quiet)

    print("\nBest config for .env:")
    print(f"RSI_MIN={best['RSI_MIN']}")
    print(f"RSI_MAX={best['RSI_MAX']}")
    print(f"MOMENTUM_LOOKBACK={best['MOMENTUM_LOOKBACK']}")
    print(f"MIN_MOMENTUM_PCT={best['MIN_MOMENTUM_PCT']}")
    print(f"TAKE_PROFIT_BUFFER_USD={best['TAKE_PROFIT_BUFFER_USD']}")
    print(f"MAX_HOLD_MINUTES={best['MAX_HOLD_MINUTES']}")


if __name__ == "__main__":
    main()
