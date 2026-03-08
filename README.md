# Kinematic Engine

A state-of-the-art backtesting environment that models market price action as a physical object moving through a resistive medium.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Polygon.io API key
echo "POLYGON_API_KEY=your_key_here" > .env

# Run the full pipeline
python main.py --tickers SPY --start 2024-01-01 --end 2024-12-31

# Run with cached data only (skip download)
python main.py --skip-download --tickers SPY
```

## What This Project Does

The engine answers three questions:

1. Is there a valid setup right now? (`Strategy` signals)
2. What typically happens after that setup? (`Oracle` forward metrics: MFE/MAE/confidence)
3. If traded with explicit exits, is it actually profitable? (`TradeSimulator` for Market Impulse)

## Architecture (End-to-End Flow)

| Module       | Path              | Purpose                                      |
|:-------------|:------------------|:---------------------------------------------|
| **Chronos**  | `src/chronos/`    | Polygon.io data pipeline with Parquet cache   |
| **Newton**   | `src/newton/`     | Physics engine (Velocity, Acceleration, Jerk) |
| **Strategy** | `src/strategy/`   | Configurable strategy agents                  |
| **Oracle**   | `src/oracle/`     | MFE/MAE metrics and reporting                 |

### Timezone And Session Handling

- Raw bars are stored with UTC timestamps.
- All market-session filters (open/close windows) are evaluated in `America/New_York` time.
- This avoids accidental use of pre/post-market bars when a strategy is intended for regular options session logic.

### Data + Backtest Flow

```text
Polygon API (1-min OHLCV)
  -> Chronos client fetches in chunks with pagination/retries
  -> Chronos storage saves daily parquet: data/<TICKER>/<YYYY-MM-DD>.parquet
  -> load_bars(ticker, start, end)
  -> Newton enrich():
       velocity_1m, accel_1m, jerk_1m, ema_*, volume_ma_*,
       internal_strength, directional_mass, directional_mass_ma_*, vpoc_4h
  -> Strategy generate_signals():
       signal (and signal_direction for directional strategies)
  -> Oracle:
       add_forward_metrics() and summarise_signals()
       or TradeSimulator.simulate() for trade-level P&L
  -> Reporter:
       data/results/<timestamp>_<ticker>_<strategy>/
         experiment.json, trade_log.csv, summary.txt
```

## How Metrics Are Calculated

### Physics Features (Newton)

- `velocity_1m = close[t] - close[t-1]`
- `accel_1m = velocity_1m[t] - velocity_1m[t-1]`
- `jerk_1m = accel_1m[t] - accel_1m[t-1]`
- `ema_p`: exponential moving average of `close` for period `p`
- `volume_ma_n`: rolling mean of `volume` over `n` bars
- `internal_strength = ((close - low) - (high - close)) / (high - low)` (0 when `high == low`)
- `directional_mass = volume * internal_strength`
- `directional_mass_ma_n`: rolling mean of `directional_mass` over `n` bars
- `vpoc_4h`: most volume-concentrated rounded price in rolling `vpoc_lookback_bars` window (default 240 bars)

### Forward Metrics (Oracle)

For each bar `i` (using future bars only):

- `entry_price = close[i]`
- `future_high = max(high[i+1 : i+1+N])`
- `future_low = min(low[i+1 : i+1+N])`
- `MFE = future_high - entry_price`
- `MAE = entry_price - future_low`
- `win = MFE > 2 * MAE` (default 2:1 reward:risk threshold)

Where `N = forward_window_bars` (default 15 bars = 15 minutes).

Summary metrics are computed on rows where `signal == True` and forward metrics are non-null:

- `total_signals`, `wins`, `losses`
- `confidence_score = wins / total_signals`
- distribution stats: `avg/median/max MFE` and `avg/median/max MAE`

### Trade-Level Metrics (Market Impulse Simulator)

The `TradeSimulator` converts directional signals into non-overlapping trades:

- Entry: at signal bar close
- Exit:
  - Long stop: first full 1-min bar with `high < vma_10_5m`
  - Short stop: first full 1-min bar with `low > vma_10_5m`
  - Otherwise exit at end of session (`15:59` bar)

Per-trade P&L:

- Long: `pnl = exit_price - entry_price`
- Short: `pnl = entry_price - exit_price`

Aggregate metrics:

- `win_rate = winners / total_trades`
- `avg_winner`, `avg_loser`
- `profit_factor = gross_wins / abs(gross_losses)`
- `expectancy = total_pnl / total_trades`
- `total_pnl`

## Strategy Logic (Current)

### EMA Momentum (`src/strategy/ema_momentum.py`)

Signal when all gates are true:

- Trend gate: `EMA(4) > EMA(8) > EMA(12)`
- Location gate: `close > vpoc_4h`
- Force gate: `volume > volume_ma_20`

### Market Impulse (`src/strategy/market_impulse.py`)

- Regime from 5-min VWMA stack: bullish or bearish
- Trigger on 1-min cross-and-reclaim of VMA
- Time filter: default entry window `09:33` to `10:30` ET
- Emits both `signal` and `signal_direction` (`long`/`short`)

### Elastic Band Reversion (`src/strategy/elastic_band_reversion.py`)

- Uses dynamic stretch via rolling z-score of VPOC distance (instead of fixed % stretch).
- Long/short still require velocity+jerk exhaustion flip.
- Participation gate uses directional mass polarity:
  - Long requires `directional_mass > 0`
  - Short requires `directional_mass < 0`

### Regime Router (`src/strategy/regime_router.py`)

- Uses rolling volatility ratio and trend-velocity to classify regime.
- Routes to `Kinematic Ladder` (trend) or `Compression Breakout` (compression).
- Emits `route_regime` with directional signals.

## How We Backtest

### Standard Pipeline Backtest (EMA-style)

Run:

```bash
python main.py --tickers SPY --start 2024-01-01 --end 2024-12-31
```

This does:

1. Load/download bars.
2. Enrich bars with Newton features.
3. Generate signals.
4. Compute forward MFE/MAE over configured lookahead.
5. Summarize signal quality and save immutable experiment artifacts.

### Trade Simulation Backtest (Market Impulse)

Run:

```bash
python scripts/run_market_impulse.py --tickers SPY QQQ IWM
```

This does:

1. Load cached bars.
2. Enrich with both base physics and Market Impulse MTF columns.
3. Generate directional entries.
4. Walk bar-by-bar to exit each trade with explicit stop/EOD logic.
5. Report P&L metrics and write trade logs to `data/results/`.

### Research Strategy Backtests (Novel Ideas)

Run:

```bash
python scripts/run_novel_ideas.py --tickers SPY QQQ IWM --start 2025-01-01 --end 2026-02-28
```

This does:

1. Runs directional strategy candidates (`Elastic Band`, `Kinematic Ladder`, `Compression Breakout`, `Regime Router`).
2. Produces classic directional summaries (signal counts, 2:1 confidence, MFE/MAE diagnostics).
3. Produces ratio-grid robustness (`1.0, 1.25, 1.5, 2.0`) with Monte Carlo probability of positive expectancy.
4. Saves outputs in `data/results/` as summary and robustness CSV/JSON artifacts.

### Walk-Forward Out-of-Sample Validation

Run:

```bash
python scripts/run_walk_forward_novel.py --tickers SPY QQQ IWM --start 2025-01-01 --end 2026-02-28 --train-months 6 --test-months 3
```

This does:

1. Splits history into rolling train/test windows.
2. Picks best reward:risk ratio on train data.
3. Tests selected ratio on the following out-of-sample window.
4. Saves detailed and aggregated OOS results in `data/results/`.

### Convergence Gate Pipeline

Run:

```bash
python scripts/run_convergence_pipeline.py --cost-grid 0.05,0.08,0.12
```

This does:

1. Runs walk-forward across multiple friction assumptions.
2. Aggregates candidate robustness across costs.
3. Applies promotion gates (windows, signal count, OOS hit-rate, expectancy floor).
4. Produces a ranked shortlist for holdout promotion.

### Holdout Validation (Promoted Candidates Only)

Run:

```bash
python scripts/run_holdout_validation.py
```

This does:

1. Loads only candidates promoted by convergence gates.
2. Fits ratio on calibration period only.
3. Evaluates holdout-only expectancy across friction stress assumptions.
4. Emits final holdout pass/fail promotion decisions.

## How Success Is Measured

Use both layers below; they answer different questions:

1. Signal quality (Oracle forward metrics):
   - High `confidence_score`
   - Higher `avg_mfe` than `avg_mae`
   - Stable median values (not only max outliers)
2. Executability/profitability (Trade simulator):
   - `profit_factor > 1.0` (preferably > 1.3-1.5)
   - Positive `expectancy`
   - Acceptable drawdown proxy via loser size and MAE behavior

Practical interpretation:

- A setup can have good forward excursion stats but still fail as a tradable system if exits/holding rules produce poor expectancy.
- A robust strategy should pass both filters: probabilistic edge and executable trade economics.

## Running Tests

```bash
./.venv/bin/pytest tests/ -v
```

## Configuration

All defaults are in `src/config.py` and are overridable via `.env`:

| Variable           | Default           | Description                     |
|:-------------------|:------------------|:--------------------------------|
| `POLYGON_API_KEY`  | *(required)*      | Polygon.io API key              |
| `DEFAULT_TICKERS`  | SPY, QQQ, IWM     | Tickers to download             |
| `LOOKBACK_YEARS`   | 2                  | Years of historical data        |
| `VPOC_LOOKBACK_BARS` | 240              | Rolling window for VPOC (bars)  |
| `EMA_PERIODS`      | 4, 8, 12          | EMA stack periods               |
| `FORWARD_WINDOW_BARS` | 15             | Forward-look window for MFE/MAE |
