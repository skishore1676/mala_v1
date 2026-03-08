#!/usr/bin/env python3
"""
Market Impulse Strategy – Run Script

Executes the full pipeline with the Market Impulse strategy:
  1. Chronos  → Load cached 1-min data
  2. Newton   → Filter to market hours, enrich with physics + impulse
  3. Strategy → Generate cross-and-reclaim signals
  4. Simulator → Bar-by-bar P&L with VMA-based exits

Usage:
    python scripts/run_market_impulse.py
    python scripts/run_market_impulse.py --tickers SPY QQQ IWM
    python scripts/run_market_impulse.py --start 2025-02-01 --end 2026-02-28
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import settings
from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.strategy.market_impulse import MarketImpulseStrategy
from src.oracle.trade_simulator import TradeSimulator

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Impulse Strategy Backtest"
    )
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
        help="Start date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date in YYYY-MM-DD",
    )
    return parser.parse_args()


def print_simulation_results(ticker: str, result, label: str = "") -> None:
    """Pretty-print simulation results with Rich."""
    if result.total_trades == 0:
        console.print(f"[yellow]⚠ No trades to show for {label}[/]")
        return

    title = f"📊 {ticker}"
    if label:
        title += f" │ {label}"

    # Win rate color
    wr = result.win_rate
    wr_color = "green" if wr >= 0.5 else "yellow" if wr >= 0.4 else "red"

    # Profit factor color
    pf = result.profit_factor
    pf_color = "green" if pf >= 1.5 else "yellow" if pf >= 1.0 else "red"

    # Expectancy color
    exp = result.expectancy
    exp_color = "green" if exp > 0 else "red"

    table = Table(show_lines=True, title=title, title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Winners", f"[green]{len(result.winners)}[/]")
    table.add_row("Losers", f"[red]{len(result.losers)}[/]")
    table.add_row("Win Rate", f"[{wr_color}]{wr:.1%}[/]")
    table.add_row("", "")
    table.add_row("Avg Winner", f"[green]+${result.avg_winner:.4f}[/]")
    table.add_row("Avg Loser", f"[red]${result.avg_loser:.4f}[/]")
    table.add_row("Profit Factor", f"[{pf_color}]{pf:.2f}[/]")
    table.add_row("", "")
    table.add_row("Expectancy/Trade", f"[{exp_color}]${exp:.4f}[/]")
    table.add_row("Total P&L", f"[{exp_color}]${result.total_pnl:.2f}[/]")

    # Exit reason breakdown
    vma_stops = sum(1 for t in result.trades if t.exit_reason == "vma_stop")
    eod_exits = sum(1 for t in result.trades if t.exit_reason == "eod")
    table.add_row("", "")
    table.add_row("VMA Stops", str(vma_stops))
    table.add_row("EOD Exits", str(eod_exits))

    # Avg bars held
    avg_bars = sum(t.bars_held for t in result.trades) / result.total_trades
    table.add_row("Avg Bars Held", f"{avg_bars:.0f} min")

    console.print(table)


def run() -> None:
    """Execute the Market Impulse pipeline with trade simulation."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args = parse_args()

    console.print()
    console.rule("[bold green]Market Impulse Strategy Backtest[/]")
    console.print(
        f"  Tickers: {args.tickers}  │  "
        f"Range: {args.start} → {args.end}"
    )
    console.print(
        f"  Entry: 1-min VMA cross-and-reclaim │ "
        f"Exit: full bar vs 5-min VMA or EOD"
    )
    console.print()

    storage = LocalStorage()
    physics = PhysicsEngine()
    strategy = MarketImpulseStrategy(
        entry_buffer_minutes=settings.impulse_entry_buffer_minutes,
        entry_window_minutes=settings.impulse_entry_window_minutes,
        market_open_hour=settings.market_open_hour,
        market_open_minute=settings.market_open_minute,
    )
    simulator = TradeSimulator(vma_5m_col="vma_10_5m")

    for ticker in args.tickers:
        console.rule(f"[bold cyan]{ticker}")

        # ── Load data ────────────────────────────────────────────────────
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            console.print(f"[yellow]⚠ No data for {ticker}, skipping.[/]")
            continue

        console.print(f"  Raw bars loaded: {len(df):,}")

        # ── Physics enrichment (basic) ───────────────────────────────────
        df = physics.enrich(df)

        # ── Market Impulse enrichment (filters to market hours) ──────────
        df = physics.enrich_market_impulse(
            df,
            vma_length=settings.vma_length,
            vwma_periods=tuple(settings.vwma_periods),
        )

        console.print(f"  Market-hours bars: {len(df):,}")

        # ── Strategy signals ─────────────────────────────────────────────
        df = strategy.generate_signals(df)

        signal_count = df.filter(df["signal"]).height
        if signal_count == 0:
            console.print(f"[yellow]⚠ No signals for {ticker}, skipping.[/]")
            continue

        # ── Trade Simulation ─────────────────────────────────────────────
        result = simulator.simulate(df)
        console.print()

        # ── Combined results ─────────────────────────────────────────────
        print_simulation_results(ticker, result, "All Trades")
        console.print()

        # ── Long-only results ────────────────────────────────────────────
        long_result = result.long_trades()
        if long_result.total_trades > 0:
            print_simulation_results(ticker, long_result, "Long Only")
            console.print()

        # ── Short-only results ───────────────────────────────────────────
        short_result = result.short_trades()
        if short_result.total_trades > 0:
            print_simulation_results(ticker, short_result, "Short Only")
            console.print()

        # ── Save trade log ───────────────────────────────────────────────
        trade_df = result.to_dataframe()
        if not trade_df.is_empty():
            out_path = Path(f"data/results/market_impulse_{ticker}_trades.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trade_df.write_csv(out_path)
            console.print(f"  💾 Trade log → [green]{out_path}[/]")

    console.rule("[bold green]✓ Market Impulse Backtest Complete[/]")


if __name__ == "__main__":
    run()
