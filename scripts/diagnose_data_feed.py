from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bot log market_data against real Alpaca bars for a time window."
    )
    parser.add_argument("--log", required=True, help="Path to bot log file")
    parser.add_argument("--start", required=True, help="Start UTC (e.g. 2026-02-23T01:01:00Z)")
    parser.add_argument("--end", required=True, help="End UTC (e.g. 2026-02-23T06:41:00Z)")
    parser.add_argument("--symbol", default="BTC/USD", help="Data symbol for Alpaca API")
    parser.add_argument(
        "--data-base-url",
        default=os.getenv("APCA_DATA_BASE_URL", "https://data.alpaca.markets"),
        help="Alpaca market data base URL",
    )
    return parser.parse_args()


def load_log_market_rows(log_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "market_data":
            rows.append(row)
    return rows


def fetch_bars(start: str, end: str, symbol: str, data_base_url: str) -> List[Dict[str, Any]]:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY in environment/.env")

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    url = f"{data_base_url.rstrip('/')}/v1beta3/crypto/us/bars"

    bars: List[Dict[str, Any]] = []
    page_token = None
    while True:
        params: Dict[str, Any] = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start,
            "end": end,
            "sort": "asc",
            "limit": 1000,
        }
        if page_token:
            params["page_token"] = page_token

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        chunk = payload.get("bars", {}).get(symbol, [])
        bars.extend(chunk)
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    return bars


def max_same_run(values: List[float]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def main() -> None:
    load_dotenv()
    args = parse_args()

    log_path = Path(args.log)
    log_rows = load_log_market_rows(log_path)
    if not log_rows:
        raise RuntimeError("No market_data rows found in log")

    log_rows = [
        row
        for row in log_rows
        if args.start <= str(row.get("timestamp_utc", "")) <= args.end
    ]
    if not log_rows:
        raise RuntimeError("No market_data rows in requested time range")

    log_closes = [float(row["last_price"]) for row in log_rows]

    bars = fetch_bars(args.start, args.end, args.symbol, args.data_base_url)
    if not bars:
        raise RuntimeError("No bars returned from Alpaca API for range")
    api_closes = [float(bar["c"]) for bar in bars]

    out = {
        "window": {"start": args.start, "end": args.end},
        "log": {
            "rows": len(log_rows),
            "first_ts": log_rows[0]["timestamp_utc"],
            "last_ts": log_rows[-1]["timestamp_utc"],
            "unique_closes": len(set(log_closes)),
            "max_same_close_run": max_same_run(log_closes),
        },
        "api": {
            "bars": len(bars),
            "first_ts": bars[0]["t"],
            "last_ts": bars[-1]["t"],
            "unique_closes": len(set(api_closes)),
            "max_same_close_run": max_same_run(api_closes),
        },
        "comparison": {
            "log_flat_vs_api_flat_mismatch": (
                max_same_run(log_closes) > 30 and max_same_run(api_closes) <= 3
            )
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
