from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
BACKTEST_PATH = ROOT / "tests" / "backtest.py"
DEFAULT_RUNTIME_PARAMS_PATH = SRC_DIR / "runtime_params.json"

# Load environment defaults from env.json
ENV_JSON_PATH = SRC_DIR / "env.json"
ENV_DEFAULTS: Dict[str, str] = {}
if ENV_JSON_PATH.exists():
    with open(ENV_JSON_PATH, "r") as f:
        ENV_DEFAULTS = json.load(f)


def get_config_value(name: str, default: str = "") -> str:
    """Get configuration value from environment, falling back to env.json defaults."""
    return os.getenv(name, ENV_DEFAULTS.get(name, default))


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
    """Run the parameter sweep in parallel using all available CPU cores.

    On an Intel Ultra 9 285K (24 P-cores + 16 E-cores) this runs the full
    216-combination grid in parallel batches instead of serially, reducing
    wall time from ~10 minutes to under a minute.
    """
    configs = build_grid()
    results: List[Dict[str, object]] = []
    total = len(configs)
    completed = 0

    # Use at most cpu_count workers; each worker is an independent Python
    # subprocess running backtest.py so there is no GIL contention.
    # SWEEP_WORKERS=0 (default) means use all logical CPU cores.
    _cfg_workers = int(get_config_value("SWEEP_WORKERS", "0") or "0")
    workers = _cfg_workers if _cfg_workers > 0 else (os.cpu_count() or 4)
    workers = max(1, min(workers, total))

    if not sweep_quiet:
        print(f"Running {total} backtests (BACKTEST_DAYS={backtest_days}, workers={workers})...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_cfg = {
            executor.submit(
                run_backtest_with_env,
                {**cfg, "BACKTEST_DAYS": str(backtest_days)},
            ): cfg
            for cfg in configs
        }

        for future in as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            completed += 1
            try:
                metrics = future.result()
            except Exception as exc:
                if not sweep_quiet:
                    print(f"[{completed}/{total}] exception | {cfg} | {exc}")
                continue

            if not metrics or "error" in metrics:
                if not sweep_quiet:
                    print(f"[{completed}/{total}] failed | {cfg}")
                continue

            row = {**cfg, **metrics}
            results.append(row)

            if not sweep_quiet:
                print(
                    f"[{completed}/{total}] equity=${metrics['equity']:.2f}, "
                    f"pnl=${metrics['pnl']:.2f}, dd=${metrics['max_drawdown_usd']:.2f}, "
                    f"trades={metrics['trades']}, score={metrics['score']:.2f} | {cfg}"
                )
            elif completed % 20 == 0 or completed == total:
                print(f"Progress: {completed}/{total}")

    if not results:
        raise RuntimeError("Sweep completed but no successful backtests were returned.")

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    best = results[0]

    runtime_path = Path(runtime_params_path)
    write_runtime_params(runtime_path, best, backtest_days)

    return best


def main() -> None:
    backtest_days = int(get_config_value("BACKTEST_DAYS", "3"))
    sweep_quiet = get_config_value("SWEEP_QUIET", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    best = run_sweep(backtest_days=backtest_days, sweep_quiet=sweep_quiet)

    print("\nBest config for env.json:")
    print(f"RSI_MIN={best['RSI_MIN']}")
    print(f"RSI_MAX={best['RSI_MAX']}")
    print(f"MOMENTUM_LOOKBACK={best['MOMENTUM_LOOKBACK']}")
    print(f"MIN_MOMENTUM_PCT={best['MIN_MOMENTUM_PCT']}")
    print(f"TAKE_PROFIT_BUFFER_USD={best['TAKE_PROFIT_BUFFER_USD']}")
    print(f"MAX_HOLD_MINUTES={best['MAX_HOLD_MINUTES']}")


if __name__ == "__main__":
    main()
