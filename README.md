# Extremely Simple Alpaca Crypto Bot

This bot does exactly three things:
- **Buy** when a simple momentum setup appears
- **Hold** the position
- **Sell** once unrealized profit reaches **$50**

The bot automatically uses your actual Alpaca account cash balance (98% for safety margin).

## Strategy (MVP)
- Pulls 1-minute crypto bars from Alpaca data API
- Computes short SMA (9) and long SMA (26)
- Uses RSI and short-term momentum as additional entry filters
- Buys when trend + RSI + momentum all agree
- Detects stale/frozen market data and pauses entries until feed resumes
- Exits on take-profit, stop-loss, or max hold time
- Then waits for the next setup

## 1) Setup

1. Install `uv` if needed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create a Python virtual environment and activate it:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```
   This project is pinned to Python 3.13 and includes TensorFlow in the main environment.

### Linux CUDA / GPU checklist (TensorFlow pip)

For GPU acceleration, verify NVIDIA driver visibility first:

```bash
nvidia-smi
```

Then verify TensorFlow sees your GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU is listed, follow TensorFlow's Linux pip guidance to ensure CUDA components are correctly available in the environment.

Then train and save the model file used by bot inference:

```bash
python src/model.py
```

This writes `src/model.keras`.

When running `src/bot.py`, the bot can also retrain this model periodically using:
- `AUTO_MODEL_RETRAIN=true`
- `MODEL_REFRESH_MINUTES=240`
- `MODEL_TRAIN_PYTHON=.venv/bin/python`

If `src/model.keras` is missing or older than `MODEL_REFRESH_MINUTES`, bot auto-triggers retraining.

### RTX 5090: build TensorFlow from source (sm_120)

If prebuilt TensorFlow wheels do not run reliably on compute capability `12.0`, build a local wheel targeting `sm_120`:

```bash
scripts/build_tensorflow_sm120.sh
```

This script will:
- sync deps with `uv`
- clone/update TensorFlow source in `.tensorflow-src/`
- configure CUDA build with `TF_CUDA_COMPUTE_CAPABILITIES=12.0`
- build a wheel via `bazelisk`
- install that wheel into `.venv`

Useful overrides:

```bash
TF_REF=master CUDA_CC=12.0 JOBS=32 scripts/build_tensorflow_sm120.sh
```

After install, verify GPU visibility:

```bash
uv run --python 3.13 python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
4. Copy `.env.example` to `.env` and add your Alpaca API credentials:
   ```
   APCA_API_KEY_ID=your_key_here
   APCA_API_SECRET_KEY=your_secret_here
   ```

## 2) Run

```bash
python src/bot.py
```

## Production session (12-hour run)

Run a bounded production session with one command:

```bash
bash scripts/run_production_session.sh
```

Defaults:
- `DURATION_HOURS=12`
- `PREPARE_MODEL=true` (runs GPU warmup + model training before bot loop)
- hard timeout with graceful kill window

Useful overrides:

```bash
DURATION_HOURS=12 PREPARE_MODEL=true MODEL_DAYS=7 SESSION_TAG=prod bash scripts/run_production_session.sh
```

After completion, the script writes:
- `logs/<session>_bot.log` (raw session output)
- `logs/<session>_summary.json` (event/action/error summary)
- `logs/<session>.status` (bot exit code; `124` means timeout reached cleanly)
- `logs/<session>.meta.json` (session metadata)

You can re-run summary manually for any session log:

```bash
python scripts/review_bot_session.py --log logs/<session>_bot.log --exit-code 124
```

## Config

Configuration is managed in two places:

### **src/env.json** - Bot settings (committed to git)
All bot behavior settings are stored here. You can edit this file to change default parameters:
- `STARTING_BALANCE_USD`: Fallback value if account cash cannot be fetched (bot uses actual Alpaca account cash)
- `TARGET_PROFIT_USD`: Target profit per trade
- `STOP_LOSS_USD`: Stop loss threshold
- `TAKE_PROFIT_BUFFER_USD`: Buffer above target profit before selling
- `MAX_HOLD_MINUTES`: Maximum time to hold a position
- `COOLDOWN_MINUTES`: Cooldown period after selling
- `BAR_LIMIT`: Number of historical bars to fetch
- `RSI_WINDOW`, `RSI_MIN`, `RSI_MAX`: RSI indicator settings
- `MOMENTUM_LOOKBACK`, `MIN_MOMENTUM_PCT`: Momentum filter settings
- `MAX_STALE_POLLS`: Number of consecutive stale polls before entry is paused
- `MAX_BAR_AGE_SECONDS`: Max allowed age for latest bar before feed is treated as stale
- `STALE_EVENT_RESET_AFTER`: Number of stale-data events before bot recreates Alpaca client automatically
- `BACKTEST_MAX_STALE_RUN`: Abort backtest if too many consecutive identical closes are seen
- And more...

**Note:** The bot automatically uses your actual Alpaca account cash balance. It will invest 98% of available cash per trade for safety.

### **.env** - Secrets only (not committed to git)
Only API credentials belong here:
- `APCA_API_KEY_ID`: Your Alpaca API key
- `APCA_API_SECRET_KEY`: Your Alpaca API secret

**Note:** Any setting from `src/env.json` can be overridden by setting it in `.env` or as an environment variable.
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
Backtest now aborts early on stale/frozen data runs to avoid tuning on bad inputs.
By default, backtest uses rule-based signals only for speed (`BACKTEST_USE_MODEL=false`). Set `BACKTEST_USE_MODEL=true` to include model inference.

## Diagnose feed vs market data

Compare bot log market_data rows with actual Alpaca bars for the same UTC window:

```bash
python scripts/diagnose_data_feed.py --log logs/bot_20260223_010150_utc.log --start 2026-02-23T01:01:00Z --end 2026-02-23T06:41:00Z
```

This prints JSON with unique-close counts and max repeated-close run lengths for both sources.

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
