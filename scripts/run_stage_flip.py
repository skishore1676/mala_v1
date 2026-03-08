#!/usr/bin/env python3
"""
Simplified Market Impulse – Stage Flip Strategy

Entry: When 1-min impulse stage flips to:
  - "acceleration" (bullish + close ≥ VMA) → enter LONG
  - "deceleration" (bearish + close ≤ VMA) → enter SHORT

Exit: When stage changes away from the entry stage.

Time filter: Entry only between 9:33 – 10:30 ET.
             Exits allowed anytime during market hours.
             Time stop at 15:59 ET.

Runs multiple experiments with different VMA lengths (2, 3, 4).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, time as dt_time, timedelta
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.chronos.storage import LocalStorage
from src.newton.market_impulse import enrich_impulse_columns

console = Console()


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl: float = 0.0
    bars_held: int = 0
    entry_stage: str = ""

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class SimResult:
    trades: List[Trade] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def win_rate(self) -> float:
        return self.winners / self.n if self.n else 0.0

    @property
    def avg_winner(self) -> float:
        w = [t.pnl for t in self.trades if t.is_winner]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loser(self) -> float:
        l = [t.pnl for t in self.trades if not t.is_winner]
        return sum(l) / len(l) if l else 0.0

    @property
    def profit_factor(self) -> float:
        gw = sum(t.pnl for t in self.trades if t.is_winner)
        gl = abs(sum(t.pnl for t in self.trades if not t.is_winner))
        return gw / gl if gl > 0 else float("inf")

    @property
    def expectancy(self) -> float:
        return sum(t.pnl for t in self.trades) / self.n if self.n else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    def by_direction(self, d: str) -> "SimResult":
        return SimResult(trades=[t for t in self.trades if t.direction == d])

    def by_exit_reason(self, r: str) -> int:
        return sum(1 for t in self.trades if t.exit_reason == r)

    def to_dataframe(self) -> pl.DataFrame:
        if not self.trades:
            return pl.DataFrame()
        return pl.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "exit_reason": t.exit_reason,
                "pnl": round(t.pnl, 4),
                "bars_held": t.bars_held,
                "entry_stage": t.entry_stage,
            }
            for t in self.trades
        ])


def filter_market_hours(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only 9:30 – 16:00 ET bars."""
    mkt_open = dt_time(9, 30)
    mkt_close = dt_time(16, 0)
    before = len(df)
    df = df.filter(
        (pl.col("timestamp").dt.time() >= mkt_open)
        & (pl.col("timestamp").dt.time() <= mkt_close)
    )
    logger.info("Market-hours filter: {} → {} bars", before, len(df))
    return df


def compute_5m_regime(df: pl.DataFrame, vma_length: int = 3) -> pl.DataFrame:
    """
    Resample 1-min market-hours data to 5-min bars, compute impulse
    regime, then join it back to the 1-min bars via forward-fill.

    Returns the 1-min DataFrame with 'impulse_regime_5m' column added.
    """
    df_5m = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every="5m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )
    df_5m = enrich_impulse_columns(
        df_5m, vma_length=vma_length, vwma_periods=(8, 21, 34), suffix="_5m"
    )

    join_df = df_5m.select(["timestamp", "impulse_regime_5m"]).rename(
        {"timestamp": "ts_5m"}
    )

    df = df.sort("timestamp").join_asof(
        join_df.sort("ts_5m"),
        left_on="timestamp",
        right_on="ts_5m",
        strategy="backward",
    )
    return df


def run_stage_flip_simulation(
    df: pl.DataFrame,
    vma_length: int,
    require_5m_confirmation: bool = False,
    entry_start: dt_time = dt_time(9, 33),
    entry_end: dt_time = dt_time(10, 30),
    market_close: dt_time = dt_time(15, 59),
) -> SimResult:
    """
    Simulate the stage-flip strategy.

    Entry:  stage flips to acceleration/deceleration within time window.
            If require_5m_confirmation=True, also requires 5-min regime
            to agree: long only if 5m=bullish, short only if 5m=bearish.
    Exit:   stage changes away from entry stage, or EOD.
    """
    # Enrich with 1-min impulse for this VMA length
    df_enriched = enrich_impulse_columns(
        df, vma_length=vma_length, vwma_periods=(8, 21, 34), suffix=""
    )

    # Compute 5-min regime if confirmation is needed
    regime_5m = None
    if require_5m_confirmation:
        df_enriched = compute_5m_regime(df_enriched, vma_length=vma_length)
        regime_5m = df_enriched["impulse_regime_5m"].to_list()

    stages = df_enriched["impulse_stage"].to_list()
    close = df_enriched["close"].to_numpy()
    timestamps = df_enriched["timestamp"].to_list()
    bar_times = df_enriched.select(
        pl.col("timestamp").dt.time().alias("t")
    )["t"].to_list()
    dates = df_enriched.select(
        pl.col("timestamp").dt.date().alias("d")
    )["d"].to_list()

    n = len(df_enriched)
    trades: List[Trade] = []
    i = 1

    while i < n:
        current_stage = stages[i]
        prev_stage = stages[i - 1]

        if current_stage == prev_stage:
            i += 1
            continue

        if current_stage not in ("acceleration", "deceleration"):
            i += 1
            continue

        if bar_times[i] < entry_start or bar_times[i] > entry_end:
            i += 1
            continue

        direction = "long" if current_stage == "acceleration" else "short"

        # 5-min confirmation check
        if require_5m_confirmation and regime_5m is not None:
            regime = regime_5m[i]
            if direction == "long" and regime != "bullish":
                i += 1
                continue
            if direction == "short" and regime != "bearish":
                i += 1
                continue

        entry_idx = i
        entry_time = timestamps[i]
        entry_price = close[i]
        entry_date = dates[i]
        entry_stage = current_stage

        j = i + 1
        exit_reason = "eod"

        while j < n:
            if dates[j] != entry_date:
                j -= 1
                exit_reason = "eod"
                break
            if bar_times[j] >= market_close:
                exit_reason = "eod"
                break
            if stages[j] != entry_stage:
                exit_reason = "stage_change"
                break
            j += 1

        exit_idx = min(j, n - 1)
        exit_price = close[exit_idx]
        exit_time = timestamps[exit_idx]
        bars_held = exit_idx - entry_idx
        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)

        trades.append(Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            exit_reason=exit_reason,
            pnl=round(pnl, 4),
            bars_held=bars_held,
            entry_stage=entry_stage,
        ))

        i = exit_idx + 1

    return SimResult(trades=trades)





def print_result_row(label: str, r: SimResult) -> list:
    """Return a row for the summary table."""
    if r.n == 0:
        return [label, "0", "-", "-", "-", "-", "-", "-", "-"]
    avg_bars = sum(t.bars_held for t in r.trades) / r.n
    return [
        label,
        str(r.n),
        f"{r.win_rate:.1%}",
        f"+${r.avg_winner:.3f}" if r.avg_winner else "-",
        f"${r.avg_loser:.3f}" if r.avg_loser else "-",
        f"{r.profit_factor:.2f}",
        f"${r.expectancy:.4f}",
        f"${r.total_pnl:.2f}",
        f"{avg_bars:.0f}",
    ]


def run():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    console.print()
    console.rule("[bold green]Stage Flip VMA=3 │ SPY vs QQQ vs IWM[/]")

    storage = LocalStorage()
    tickers = ["SPY", "QQQ", "IWM"]
    start = date.today() - timedelta(days=365 * settings.lookback_years)
    end = date.today()
    vma_len = 3

    all_combined: dict = {}

    for ticker in tickers:
        console.rule(f"[bold cyan]{ticker}")

        df = storage.load_bars(ticker, start, end)
        if df.is_empty():
            console.print(f"[yellow]⚠ No data for {ticker}[/]")
            continue

        df = filter_market_hours(df)
        console.print(f"  Market-hours bars: {len(df):,}")

        result = run_stage_flip_simulation(df, vma_length=vma_len)
        all_combined[ticker] = result

        cols = ["Segment", "Trades", "Win%", "Avg Win", "Avg Loss",
                "PF", "Exp/Trade", "Total $", "Avg Bars"]
        table = Table(
            title=f"📊 {ticker} │ VMA={vma_len} Stage Flip",
            show_lines=True,
            title_style="bold cyan",
        )
        for col in cols:
            table.add_column(col, justify="right" if col != "Segment" else "left")

        table.add_row(*print_result_row("Combined", result))
        table.add_row(*print_result_row("Long", result.by_direction("long")))
        table.add_row(*print_result_row("Short", result.by_direction("short")))
        console.print(table)
        console.print()

        out_path = Path(f"data/results/stage_flip_{ticker}_vma{vma_len}_baseline.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_dataframe().write_csv(out_path)
        console.print(f"  💾 → [green]{out_path}[/]")
        console.print()

    # ── Cross-ticker comparison ───────────────────────────────────────────
    console.rule("[bold]Cross-Ticker Comparison (Combined)[/]")
    cmp = Table(show_lines=True)
    cmp.add_column("Ticker", style="bold")
    cmp.add_column("Trades", justify="right")
    cmp.add_column("Win%", justify="right")
    cmp.add_column("Avg Win", justify="right")
    cmp.add_column("Avg Loss", justify="right")
    cmp.add_column("PF", justify="right")
    cmp.add_column("Exp/Trade", justify="right")
    cmp.add_column("Total $", justify="right")

    for tkr, r in all_combined.items():
        cmp.add_row(*print_result_row(tkr, r)[:-1])

    console.print(cmp)
    console.rule("[bold green]✓ Done[/]")



if __name__ == "__main__":
    run()
