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

## Architecture

| Module       | Path              | Purpose                                      |
|:-------------|:------------------|:---------------------------------------------|
| **Chronos**  | `src/chronos/`    | Polygon.io data pipeline with Parquet cache   |
| **Newton**   | `src/newton/`     | Physics engine (Velocity, Acceleration, Jerk) |
| **Strategy** | `src/strategy/`   | Configurable strategy agents                  |
| **Oracle**   | `src/oracle/`     | MFE/MAE metrics and reporting                 |

## Running Tests

```bash
pytest tests/ -v
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
