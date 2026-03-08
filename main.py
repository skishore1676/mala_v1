#!/usr/bin/env python3
"""
Kinematic Engine – Main Entry Point

Orchestrates the full pipeline:
  1. Chronos  → Download / load market data
  2. Newton   → Enrich with physics columns
  3. Strategy → Generate signals
  4. Oracle   → Compute forward metrics & report

Usage:
    python main.py                          # use defaults from config
    python main.py --tickers SPY QQQ        # override tickers
    python main.py --start 2024-01-01       # custom date range
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.chronos.client import PolygonClient
from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.strategy.ema_momentum import EMAMomentumStrategy
from src.oracle.metrics import MetricsCalculator
from src.oracle.reporting import ExperimentReporter

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kinematic Engine Backtester")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=settings.default_tickers,
        help="Tickers to process (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=365 * settings.lookback_years),
        help="Start date in YYYY-MM-DD (default: %(default)s)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date in YYYY-MM-DD (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download, use only cached data.",
    )
    return parser.parse_args()


def download_data(
    client: PolygonClient,
    storage: LocalStorage,
    tickers: list[str],
    start: date,
    end: date,
) -> None:
    """Download missing data for each ticker."""
    for ticker in tickers:
        missing = storage.missing_dates(ticker, start, end)
        if not missing:
            logger.info("{}: all dates cached, skipping download.", ticker)
            continue

        logger.info("{}: downloading {} missing dates …", ticker, len(missing))
        # Group consecutive missing dates into ranges for efficiency
        bars = client.fetch_aggs_chunked(ticker, missing[0], missing[-1])
        storage.save_bars(ticker, bars)


def run_pipeline(
    tickers: list[str],
    start: date,
    end: date,
    skip_download: bool = False,
) -> None:
    """Execute the full Chronos → Newton → Strategy → Oracle pipeline."""
    storage = LocalStorage()
    physics = PhysicsEngine()
    strategy = EMAMomentumStrategy()
    oracle = MetricsCalculator()
    reporter = ExperimentReporter()

    # ── Step 1: Data ─────────────────────────────────────────────────────
    if not skip_download:
        client = PolygonClient()
        download_data(client, storage, tickers, start, end)

    for ticker in tickers:
        console.rule(f"[bold cyan]{ticker}")

        # ── Step 2: Load ─────────────────────────────────────────────────
        df = storage.load_bars(ticker, start, end)
        if df.is_empty():
            console.print(f"[yellow]⚠ No data for {ticker}, skipping.[/]")
            continue

        # ── Step 3: Physics Enrichment ───────────────────────────────────
        df = physics.enrich(df)

        # ── Step 4: Strategy Signals ─────────────────────────────────────
        df = strategy.generate_signals(df)

        # ── Step 5: Forward Metrics ──────────────────────────────────────
        df = oracle.add_forward_metrics(df)

        # ── Step 6: Report ───────────────────────────────────────────────
        summary = oracle.summarise_signals(df)
        if summary.is_empty():
            console.print(f"[yellow]⚠ No actionable signals for {ticker}.[/]")
            continue

        _print_summary(ticker, strategy.name, summary)

        # ── Step 7: Save Experiment ──────────────────────────────────────
        trade_log = oracle.trade_log(df)
        exp_dir = reporter.save_experiment(
            ticker=ticker,
            strategy_name=strategy.name,
            strategy_params={
                "ema_periods": strategy.ema_periods,
                "volume_ma_period": strategy.volume_ma_period,
            },
            date_range=(start, end),
            total_bars=len(df),
            enriched_columns=df.columns,
            summary_df=summary,
            trade_log_df=trade_log,
            physics_params={
                "vpoc_lookback_bars": physics.vpoc_lookback,
                "ema_periods": physics.ema_periods,
                "volume_ma_period": physics.volume_ma_period,
            },
            oracle_params={
                "forward_window_bars": oracle.forward_window,
            },
        )
        console.print(f"  Experiment saved → [green]{exp_dir}[/]")


def _print_summary(ticker: str, strategy_name: str, summary) -> None:
    """Pretty-print the summary table using Rich."""
    table = Table(title=f"{ticker} │ {strategy_name}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    row = summary.row(0)
    labels = summary.columns
    for label, value in zip(labels, row):
        if isinstance(value, float):
            if "confidence" in label:
                display = f"{value:.2%}"
            else:
                display = f"{value:.4f}"
        else:
            display = str(value)
        table.add_row(label.replace("_", " ").title(), display)

    console.print(table)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args = parse_args()
    console.print(
        f"[bold green]Kinematic Engine[/] — "
        f"Tickers: {args.tickers}  "
        f"Range: {args.start} → {args.end}"
    )

    run_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        skip_download=args.skip_download,
    )
    console.print("[bold green]✓ Pipeline complete.[/]")


if __name__ == "__main__":
    main()
