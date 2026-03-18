#!/usr/bin/env python3
"""
Standalone backtest for Jerk-Pivot Momentum Strategy.

Legacy exploratory script.
Prefer the registry-backed research flow for current evaluation work.

By default generates synthetic 1-min OHLCV data for SPY/IWM using realistic
GBM parameters. Use --real-data to fetch real market data from Polygon.io.

Usage:
    python3 scripts/legacy/run_jerk_pivot_backtest.py                          # synthetic
    python3 scripts/legacy/run_jerk_pivot_backtest.py --real-data              # real data from Polygon
    python3 scripts/legacy/run_jerk_pivot_backtest.py --real-data --tickers SPY IWM --start 2024-01-01 --end 2026-02-28
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.strategy.jerk_pivot_momentum import JerkPivotMomentumStrategy
from src.config import settings
from src.chronos.client import PolygonClient
from src.chronos.storage import LocalStorage

console = Console()

# ── Synthetic data config ──────────────────────────────────────────────────
TICKER_PARAMS = {
    "SPY": {"price": 450.0, "annual_vol": 0.18, "daily_drift": 0.0003},
    "IWM": {"price": 185.0, "annual_vol": 0.22, "daily_drift": 0.0002},
    "QQQ": {"price": 370.0, "annual_vol": 0.21, "daily_drift": 0.0003},
}


def _generate_synthetic_bars(
    ticker: str,
    trading_days: int = 504,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate realistic 1-min OHLCV bars for *ticker* using GBM with
    intraday volume profile (U-shape) and microstructure noise.
    """
    rng = np.random.default_rng(seed)

    params = TICKER_PARAMS.get(ticker.upper(), TICKER_PARAMS["SPY"])
    start_price = params["price"]
    annual_vol = params["annual_vol"]
    daily_drift = params["daily_drift"]

    # 1-min vol from annual vol
    min_vol = annual_vol / np.sqrt(252 * 390)

    # Build timestamps: 9:30 – 15:59 each trading day
    # Start from 2023-01-03 (first trading day of 2023)
    base_date = datetime(2023, 1, 3, tzinfo=timezone.utc)
    # Skip weekends to get trading_days
    all_timestamps: list[datetime] = []
    d = base_date
    days_counted = 0
    while days_counted < trading_days:
        if d.weekday() < 5:  # Mon-Fri
            for h in range(9, 16):
                for m in range(60):
                    if h == 9 and m < 30:
                        continue
                    if h == 15 and m >= 60:
                        continue
                    ts = d.replace(hour=h, minute=m)
                    all_timestamps.append(ts)
            days_counted += 1
        d += timedelta(days=1)

    n_bars = len(all_timestamps)
    timestamps = np.array(all_timestamps, dtype="datetime64[us]")

    # Build intraday U-shaped volume profile (higher near open and close)
    minutes_in_day = 390
    intraday_idx = np.tile(np.arange(minutes_in_day), trading_days)[:n_bars]
    # U-shaped: high at start and end
    x = intraday_idx / (minutes_in_day - 1)  # 0 to 1
    volume_profile = 1.5 + 2.0 * (x ** 2 - x + 0.5) ** 2  # approximate U

    # GBM for close prices
    log_returns = rng.normal(
        loc=daily_drift / minutes_in_day,
        scale=min_vol,
        size=n_bars,
    )
    # Add regime changes: occasional vol spikes for realism
    vol_multiplier = np.ones(n_bars)
    spike_starts = rng.choice(n_bars, size=n_bars // 500, replace=False)
    for s in spike_starts:
        end = min(s + rng.integers(20, 120), n_bars)
        vol_multiplier[s:end] *= rng.uniform(2.0, 4.0)

    log_returns *= vol_multiplier
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Build O/H/L around close
    intrabar_range = close_prices * rng.uniform(0.0003, 0.0015, n_bars)
    open_prices = close_prices - log_returns * close_prices * rng.uniform(0.3, 0.7, n_bars)
    high_prices = np.maximum(close_prices, open_prices) + intrabar_range * rng.uniform(0.3, 0.7, n_bars)
    low_prices = np.minimum(close_prices, open_prices) - intrabar_range * rng.uniform(0.3, 0.7, n_bars)

    # Base volume: ~5M shares/day for SPY, scaled by profile
    base_daily_vol = 5_000_000 if ticker.upper() == "SPY" else 2_000_000
    base_min_vol = base_daily_vol / minutes_in_day
    volume = (base_min_vol * volume_profile * rng.lognormal(0, 0.4, n_bars)).astype(int)

    df = pl.DataFrame({
        "timestamp": pl.Series(timestamps).cast(pl.Datetime("us", "UTC")),
        "ticker": ticker.upper(),
        "open": open_prices.astype(float),
        "high": high_prices.astype(float),
        "low": low_prices.astype(float),
        "close": close_prices.astype(float),
        "volume": volume.astype(int),
    })

    return df


def _evaluate_ratio_grid(
    mfe: np.ndarray,
    mae: np.ndarray,
    ratios: list[float],
    cost_r: float = 0.05,
    bootstrap_iters: int = 2000,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    if rng is None:
        rng = np.random.default_rng(7)
    rows = []
    n = len(mfe)
    if n == 0:
        return rows
    for ratio in ratios:
        wins = mfe >= (ratio * mae)
        p_hat = float(np.mean(wins))
        p_be = float((1.0 + cost_r) / (1.0 + ratio))
        exp_r = p_hat * ratio - (1.0 - p_hat) - cost_r
        p_boot = rng.binomial(n=n, p=p_hat, size=bootstrap_iters) / n
        exp_boot = p_boot * ratio - (1.0 - p_boot) - cost_r
        rows.append({
            "ratio": ratio,
            "signals": n,
            "win_rate": round(p_hat, 4),
            "breakeven_confidence": round(p_be, 4),
            "edge_vs_breakeven": round(p_hat - p_be, 4),
            "expectancy_r": round(float(exp_r), 4),
            "exp_r_p05": round(float(np.quantile(exp_boot, 0.05)), 4),
            "exp_r_p95": round(float(np.quantile(exp_boot, 0.95)), 4),
            "prob_positive_exp": round(float(np.mean(exp_boot > 0.0)), 4),
        })
    return rows


def _compute_mfe_mae_eod(
    df_sig: pl.DataFrame,
    forward_window: int = 15,
) -> pl.DataFrame:
    """
    Add forward_mfe_eod and forward_mae_eod columns manually
    (fallback if MetricsCalculator.add_directional_forward_metrics is missing).
    """
    close = df_sig["close"].to_numpy()
    high = df_sig["high"].to_numpy()
    low = df_sig["low"].to_numpy()
    n = len(df_sig)

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    for i in range(n):
        end = min(i + 1 + forward_window, n)
        if end > i + 1:
            entry = close[i]
            future_high = high[i + 1:end].max()
            future_low = low[i + 1:end].min()
            mfe[i] = future_high - entry
            mae[i] = entry - future_low

    return df_sig.with_columns([
        pl.Series("forward_mfe_eod", mfe),
        pl.Series("forward_mae_eod", mae),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Jerk-Pivot Momentum backtest")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "IWM"])
    parser.add_argument("--days", type=int, default=504, help="Trading days to simulate (synthetic mode)")
    parser.add_argument("--forward-window", type=int, default=15, help="Forward MFE/MAE window (bars)")
    parser.add_argument("--cost-r", type=float, default=0.05)
    parser.add_argument("--real-data", action="store_true", help="Use real data from Polygon.io instead of synthetic")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1), help="Start date (YYYY-MM-DD) for real data")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28), help="End date (YYYY-MM-DD) for real data")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use cached data only")
    args = parser.parse_args()

    ratios = [1.0, 1.25, 1.5, 2.0]
    rng = np.random.default_rng(7)

    physics = PhysicsEngine()
    metrics = MetricsCalculator()

    strategies = [
        JerkPivotMomentumStrategy(
            vpoc_proximity_pct=0.005,
            jerk_lookback=20,
            volume_multiplier=1.0,
            volume_ma_period=settings.volume_ma_period,
            use_volume_filter=True,
            use_time_filter=True,
        ),
        JerkPivotMomentumStrategy(
            vpoc_proximity_pct=0.003,
            jerk_lookback=10,
            volume_multiplier=1.1,
            volume_ma_period=settings.volume_ma_period,
            use_volume_filter=True,
            use_time_filter=True,
            strategy_label="Jerk-Pivot Momentum (tight prox/fast jerk)",
        ),
    ]

    all_results: list[dict] = []
    all_robustness: list[dict] = []

    if args.real_data:
        console.rule("[bold green]Jerk-Pivot Momentum Backtest (REAL DATA from Polygon)[/]")
        console.print(
            f"Tickers: {args.tickers} | Range: {args.start} → {args.end} | "
            f"Forward window: {args.forward_window} bars | cost_r={args.cost_r}"
        )
    else:
        console.rule("[bold green]Jerk-Pivot Momentum Backtest (Synthetic Data)[/]")
        console.print(
            f"Tickers: {args.tickers} | Days: {args.days} | "
            f"Forward window: {args.forward_window} bars | cost_r={args.cost_r}"
        )

    # Initialize Polygon client and storage for real data mode
    polygon_client = None
    storage = None
    if args.real_data:
        try:
            polygon_client = PolygonClient()
            storage = LocalStorage()
            console.print(f"[dim]Polygon API connected successfully[/]")
        except Exception as e:
            console.print(f"[red]Failed to connect to Polygon API: {e}[/]")
            return

    for ticker in args.tickers:
        console.rule(f"[bold cyan]{ticker}")

        if args.real_data:
            # REAL DATA MODE: Fetch from Polygon
            console.print(f"[dim]Fetching real data from Polygon for {args.start} → {args.end}…[/]")
            
            if not args.skip_download:
                # Download missing data
                missing = storage.missing_dates(ticker, args.start, args.end)
                if missing:
                    console.print(f"  Downloading {len(missing)} missing trading days…")
                    bars = polygon_client.fetch_aggs_chunked(ticker, args.start, args.end)
                    storage.save_bars(ticker, bars)
                    console.print(f"  Downloaded {len(bars):,} bars")
                else:
                    console.print(f"  All dates cached, skipping download.")
            
            # Load from storage
            df = storage.load_bars(ticker, args.start, args.end)
            if df.is_empty():
                console.print(f"[yellow]⚠ No data found for {ticker}, skipping.[/]")
                continue
            console.print(f"  Loaded {len(df):,} bars from storage")
        else:
            # SYNTHETIC MODE: Generate data
            console.print(f"[dim]Generating {args.days} trading days of synthetic 1-min data…[/]")
            seed = hash(ticker) % (2**31)
            df = _generate_synthetic_bars(ticker, trading_days=args.days, seed=seed)
            console.print(f"  Generated {len(df):,} bars")

        df = physics.enrich(df)
        console.print(f"  Physics enriched: {len(df.columns)} columns")

        for strategy in strategies:
            df_sig = strategy.generate_signals(df.clone())
            signal_count = df_sig.filter(pl.col("signal")).height

            if signal_count == 0:
                console.print(f"[yellow]{strategy.name}: no signals[/]")
                continue

            # Add forward metrics
            try:
                df_eval = metrics.add_directional_forward_metrics(
                    df_sig, snapshot_windows=(args.forward_window, args.forward_window * 4)
                )
            except Exception:
                df_eval = _compute_mfe_mae_eod(df_sig, forward_window=args.forward_window)

            # Pull signal rows
            # Determine which MFE/MAE columns are available
            mfe_col = "forward_mfe_eod"
            mae_col = "forward_mae_eod"
            if mfe_col not in df_eval.columns:
                # Fall back to snapshot window columns
                snap_w = args.forward_window
                mfe_col = f"forward_mfe_{snap_w}"
                mae_col = f"forward_mae_{snap_w}"
            if mfe_col not in df_eval.columns:
                console.print(f"[yellow]{strategy.name}: no MFE/MAE columns found[/]")
                continue

            sig_rows = (
                df_eval.filter(pl.col("signal"))
                .drop_nulls(subset=[mfe_col, mae_col])
            )
            n_valid = len(sig_rows)

            if n_valid == 0:
                console.print(f"[yellow]{strategy.name}: no valid forward metrics[/]")
                continue

            mfe_arr = sig_rows[mfe_col].to_numpy().astype(float)
            mae_arr = sig_rows[mae_col].to_numpy().astype(float)
            # Remove float NaNs that survive drop_nulls (last bars of day)
            valid_mask = ~(np.isnan(mfe_arr) | np.isnan(mae_arr))
            mfe_arr = mfe_arr[valid_mask]
            mae_arr = mae_arr[valid_mask]

            if len(mfe_arr) == 0:
                console.print(f"[yellow]{strategy.name}: all MFE/MAE values NaN after float filter[/]")
                continue

            # Directional breakdown
            longs = sig_rows.filter(pl.col("signal_direction") == "long") if "signal_direction" in sig_rows.columns else pl.DataFrame()
            shorts = sig_rows.filter(pl.col("signal_direction") == "short") if "signal_direction" in sig_rows.columns else pl.DataFrame()

            # Summary table
            table = Table(title=f"{ticker} | {strategy.name}", show_lines=True)
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("Total Signals", str(n_valid))
            table.add_row("Long Signals", str(len(longs)))
            table.add_row("Short Signals", str(len(shorts)))
            table.add_row("Avg MFE (pts)", f"{float(np.mean(mfe_arr)):.4f}")
            table.add_row("Avg MAE (pts)", f"{float(np.mean(mae_arr)):.4f}")
            table.add_row("Median MFE (pts)", f"{float(np.median(mfe_arr)):.4f}")
            table.add_row("Median MAE (pts)", f"{float(np.median(mae_arr)):.4f}")
            table.add_row("Avg MFE/MAE Ratio", f"{float(np.mean(mfe_arr / (mae_arr + 1e-9))):.3f}")
            console.print(table)

            # Ratio grid
            ratio_rows = _evaluate_ratio_grid(
                mfe=mfe_arr,
                mae=mae_arr,
                ratios=ratios,
                cost_r=args.cost_r,
                bootstrap_iters=2000,
                rng=rng,
            )

            rtable = Table(title=f"{ticker} | {strategy.name} | R:R Grid", show_lines=True)
            rtable.add_column("R:R", justify="right")
            rtable.add_column("Signals", justify="right")
            rtable.add_column("Win Rate", justify="right")
            rtable.add_column("BE Conf", justify="right")
            rtable.add_column("P(E>0)", justify="right")
            rtable.add_column("Exp(R)", justify="right")
            for row in ratio_rows:
                rtable.add_row(
                    f"{float(row['ratio']):.2f}",
                    str(row["signals"]),
                    f"{float(row['win_rate']):.2%}",
                    f"{float(row['breakeven_confidence']):.2%}",
                    f"{float(row['prob_positive_exp']):.1%}",
                    f"{float(row['expectancy_r']):+.3f}",
                )
            console.print(rtable)

            # Collect for output
            for row in ratio_rows:
                all_robustness.append({
                    "ticker": ticker,
                    "strategy": strategy.name,
                    **row,
                })

            all_results.append({
                "ticker": ticker,
                "strategy": strategy.name,
                "total_signals": n_valid,
                "long_signals": len(longs),
                "short_signals": len(shorts),
                "avg_mfe_pts": round(float(np.mean(mfe_arr)), 5),
                "avg_mae_pts": round(float(np.mean(mae_arr)), 5),
                "median_mfe_pts": round(float(np.median(mfe_arr)), 5),
                "median_mae_pts": round(float(np.median(mae_arr)), 5),
                "avg_mfe_mae_ratio": round(float(np.mean(mfe_arr / (mae_arr + 1e-9))), 4),
                "win_rate_2to1": next(
                    (r["win_rate"] for r in ratio_rows if r["ratio"] == 2.0), None
                ),
                "expectancy_r_2to1": next(
                    (r["expectancy_r"] for r in ratio_rows if r["ratio"] == 2.0), None
                ),
                "prob_positive_exp_2to1": next(
                    (r["prob_positive_exp"] for r in ratio_rows if r["ratio"] == 2.0), None
                ),
            })

    if not all_results:
        console.print("[red]No results produced.[/]")
        return

    # Save outputs
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_path = out_dir / f"jerk_pivot_summary_{stamp}.json"
    robustness_path = out_dir / f"jerk_pivot_robustness_{stamp}.json"
    summary_csv_path = out_dir / f"jerk_pivot_summary_{stamp}.csv"

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(robustness_path, "w") as f:
        json.dump(all_robustness, f, indent=2, default=str)

    pl.DataFrame(all_results).write_csv(summary_csv_path)

    console.print(f"\n[green]Saved:[/] {summary_path}")
    console.print(f"[green]Saved:[/] {robustness_path}")
    console.print(f"[green]Saved:[/] {summary_csv_path}")

    return all_results, all_robustness


if __name__ == "__main__":
    result = main()
