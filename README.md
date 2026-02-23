# Extremely Simple Alpaca Crypto Bot

This bot does exactly three things:
- **Buy** when a simple momentum setup appears
- **Hold** the position
- **Sell** once unrealized profit reaches **$50**

It uses a default starting budget assumption of **$10,000** per trade cycle.

## Strategy (MVP)
- Pulls 1-minute crypto bars from Alpaca data API
- Computes short SMA (9) and long SMA (26)
- Uses RSI and short-term momentum as additional entry filters
- Buys when trend + RSI + momentum all agree
- Exits on take-profit, stop-loss, or max hold time
- Then waits for the next setup

## 1) Setup

1. Create a Python virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Alpaca paper keys.

## 2) Run

```bash
python src/bot.py
```

## Config
Use `.env` values:
- `STARTING_BALANCE_USD=10000`
- `TARGET_PROFIT_USD=50`
- `STOP_LOSS_USD=-50`
- `TAKE_PROFIT_BUFFER_USD=10`
- `MAX_HOLD_MINUTES=180`
- `COOLDOWN_MINUTES=10`
- `RSI_WINDOW=14`
- `RSI_MIN=45`
- `RSI_MAX=70`
- `MOMENTUM_LOOKBACK=5`
- `MIN_MOMENTUM_PCT=0.05`
- `TRADE_LOG_FILE=trade_log.csv`
- `LOG_PRETTY=false` (set `true` for human-readable logs)
- `BACKTEST_DAYS=3`
- `SWEEP_QUIET=true`
- `AUTO_SWEEP=true`
- `SWEEP_REFRESH_MINUTES=240`
- `RUNTIME_PARAMS_FILE=src/runtime_params.json`
- `TRADE_SYMBOL=BTCUSD`
- `DATA_SYMBOL=BTC/USD`
- `POLL_SECONDS=20`

## Backtest

Run a simple backtest (defaults to 3 days of 1-minute bars) that reuses bot logic:

```bash
python tests/backtest.py
```

It force-closes any open position on the last bar so results are fully realized.
Set `BACKTEST_DAYS` in `.env` to change the window.

## Sweep (auto-tune simple signals)

Run a small grid search over RSI/momentum/hold settings:

```bash
python src/sweeps.py
```

It uses `BACKTEST_DAYS` (default `3`) and ranks by a combined score:
- higher ending equity
- more closed trades
- lower max drawdown

It writes the best parameters to `src/runtime_params.json`.

`bot.py` auto-runs sweeps when `AUTO_SWEEP=true` and `src/runtime_params.json` is missing/stale, then reloads and applies those params each loop.

## Notes
- Start on **paper trading** first.
- This is intentionally minimal and not financial advice.
- No guarantee of profits; markets can move against you.

## GitHub Actions (GitHub-hosted)

This repo includes a continuous-style workflow at `.github/workflows/tradebot-continuous.yml`.

### Requirements
- Repository secret `BOT_ENV` containing full `.env` content
- Repo setting: **Actions > General > Workflow permissions = Read and write permissions**

### Start it
1. Push this repo.
2. Add the `BOT_ENV` secret.
3. Run workflow **TradeBot Continuous** via `workflow_dispatch`.

### How it stays running
- Each job runs the bot in a long chunk (`CHUNK_MINUTES=340`).
- At the end of each run, the workflow dispatches itself again.
- A backup cron trigger runs every hour.

This is the closest safe continuous mode in Actions. For truly always-on execution, a long-lived system service on the runner is still more reliable.
