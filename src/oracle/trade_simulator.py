"""
Trade Simulator for Market Impulse Strategy

Walks bar-by-bar after each signal entry, checking exit conditions:
  - Long stop:  full 1-min bar below 5-min VMA  (bar HIGH < VMA_5m)
  - Short stop: full 1-min bar above 5-min VMA  (bar LOW > VMA_5m)
  - Time stop:  exit at 16:00 ET close if neither triggered

Produces trade-level P&L with win rate, profit factor, and expectancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class Trade:
    """Record of a single simulated trade."""
    entry_time: object  # datetime
    exit_time: object   # datetime
    direction: str      # "long" or "short"
    entry_price: float
    exit_price: float
    exit_reason: str    # "vma_stop" or "eod"
    pnl: float = 0.0
    bars_held: int = 0
    vma_5m_at_entry: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class SimulationResult:
    """Aggregate results from the trade simulation."""
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> List[Trade]:
        return [t for t in self.trades if t.is_winner]

    @property
    def losers(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_winner]

    @property
    def win_rate(self) -> float:
        return len(self.winners) / self.total_trades if self.total_trades else 0.0

    @property
    def avg_winner(self) -> float:
        wins = [t.pnl for t in self.winners]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loser(self) -> float:
        losses = [t.pnl for t in self.losers]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.pnl for t in self.winners)
        gross_losses = abs(sum(t.pnl for t in self.losers))
        return gross_wins / gross_losses if gross_losses > 0 else float("inf")

    @property
    def expectancy(self) -> float:
        """Average P&L per trade."""
        return sum(t.pnl for t in self.trades) / self.total_trades if self.total_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    def long_trades(self) -> "SimulationResult":
        return SimulationResult(trades=[t for t in self.trades if t.direction == "long"])

    def short_trades(self) -> "SimulationResult":
        return SimulationResult(trades=[t for t in self.trades if t.direction == "short"])

    def to_dataframe(self) -> pl.DataFrame:
        """Convert trades list to a Polars DataFrame."""
        if not self.trades:
            return pl.DataFrame()
        return pl.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl": round(t.pnl, 4),
                "bars_held": t.bars_held,
                "vma_5m_at_entry": round(t.vma_5m_at_entry, 4),
            }
            for t in self.trades
        ])


class TradeSimulator:
    """
    Bar-by-bar trade simulator with VMA-based exits.

    Exit rules:
      - Long: full 1-min bar HIGH < vma_10_5m → stopped out
      - Short: full 1-min bar LOW > vma_10_5m → stopped out
      - EOD: close position at market close (16:00 ET)
    """

    def __init__(
        self,
        vma_5m_col: str = "vma_10_5m",
        market_close: dt_time = dt_time(15, 59),
    ) -> None:
        self.vma_5m_col = vma_5m_col
        self.market_close = market_close

    def simulate(self, df: pl.DataFrame) -> SimulationResult:
        """
        Run the simulation on a DataFrame that has 'signal',
        'signal_direction', and the 5-min VMA column.

        Returns a SimulationResult with all trades.
        """
        required = {"timestamp", "close", "high", "low", "signal", "signal_direction", self.vma_5m_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"TradeSimulator requires columns: {missing}")

        # Convert to numpy for fast iteration
        timestamps = df["timestamp"].to_list()
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        signal = df["signal"].to_list()
        direction = df["signal_direction"].to_list()
        vma_5m = df[self.vma_5m_col].to_numpy()

        # Pre-compute bar times for EOD check
        bar_times = df.select(pl.col("timestamp").dt.time().alias("t"))["t"].to_list()

        # Pre-compute dates for session boundary tracking
        dates = df.select(pl.col("timestamp").dt.date().alias("d"))["d"].to_list()

        n = len(df)
        trades: List[Trade] = []
        i = 0

        while i < n:
            # Look for signal entry
            if not signal[i] or direction[i] is None:
                i += 1
                continue

            # Skip if VMA_5m is NaN at entry
            if np.isnan(vma_5m[i]):
                i += 1
                continue

            entry_idx = i
            entry_time = timestamps[i]
            entry_price = close[i]
            entry_direction = direction[i]
            entry_date = dates[i]
            entry_vma_5m = vma_5m[i]

            # Walk forward to find exit
            j = i + 1
            exit_reason = "eod"

            while j < n:
                # Session boundary — if we crossed into a new day, exit at last bar of entry day
                if dates[j] != entry_date:
                    j = j - 1  # back to last bar of entry day
                    exit_reason = "eod"
                    break

                # EOD time stop
                if bar_times[j] >= self.market_close:
                    exit_reason = "eod"
                    break

                # Skip if VMA_5m is NaN
                if np.isnan(vma_5m[j]):
                    j += 1
                    continue

                # VMA stop check
                if entry_direction == "long":
                    # Full bar below 5-min VMA: bar HIGH < VMA_5m
                    if high[j] < vma_5m[j]:
                        exit_reason = "vma_stop"
                        break
                elif entry_direction == "short":
                    # Full bar above 5-min VMA: bar LOW > VMA_5m
                    if low[j] > vma_5m[j]:
                        exit_reason = "vma_stop"
                        break

                j += 1

            # Clamp to valid index
            exit_idx = min(j, n - 1)
            exit_price = close[exit_idx]
            exit_time = timestamps[exit_idx]
            bars_held = exit_idx - entry_idx

            # Calculate P&L
            if entry_direction == "long":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction=entry_direction,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                exit_reason=exit_reason,
                pnl=round(pnl, 4),
                bars_held=bars_held,
                vma_5m_at_entry=round(entry_vma_5m, 4),
            ))

            # Move past the exit bar to avoid overlapping trades
            i = exit_idx + 1

        result = SimulationResult(trades=trades)
        logger.info(
            "Simulation complete: {} trades, {:.1%} win rate, "
            "${:.4f} expectancy, {:.2f} profit factor",
            result.total_trades,
            result.win_rate,
            result.expectancy,
            result.profit_factor,
        )
        return result
