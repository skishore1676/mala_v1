"""
Trade Simulator for Market Impulse Strategy

Walks bar-by-bar after each signal entry and delegates exit decisions
to a pluggable exit policy.

Produces trade-level P&L with win rate, profit factor, and expectancy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import List

import numpy as np
import polars as pl
from loguru import logger
from src.time_utils import et_time_expr, et_date_expr


@dataclass
class Trade:
    """Record of a single simulated trade."""
    entry_time: object  # datetime
    exit_time: object   # datetime
    direction: str      # "long" or "short"
    entry_price: float
    exit_price: float
    exit_reason: str    # e.g. "vma_stop", "take_profit", "stop_loss", "eod"
    pnl: float = 0.0
    bars_held: int = 0
    vma_5m_at_entry: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True, slots=True)
class BarSnapshot:
    """Minimal bar view passed into exit-policy checks."""

    idx: int
    timestamp: object
    close: float
    high: float
    low: float
    bar_time: dt_time
    trade_date: object
    values: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class OpenTrade:
    """State captured at entry for exit-policy evaluation."""

    entry_idx: int
    entry_time: object
    direction: str
    entry_price: float
    entry_date: object
    entry_values: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """Exit decision returned by a policy when a bar triggers an exit."""

    reason: str
    exit_price: float | None = None


class ExitPolicy(ABC):
    """Interface for bar-by-bar trade exits."""

    policy_name: str

    @property
    def required_columns(self) -> set[str]:
        return set()

    def entry_is_valid(self, entry_bar: BarSnapshot, direction: str) -> bool:
        return True

    @abstractmethod
    def should_exit(self, trade: OpenTrade, bar: BarSnapshot) -> ExitDecision | None:
        ...


@dataclass(frozen=True, slots=True)
class VmaTrailingExitPolicy(ExitPolicy):
    """Replicates the legacy VMA trailing stop behavior."""

    vma_col: str = "vma_10_5m"
    policy_name: str = "vma_trailing"

    @property
    def required_columns(self) -> set[str]:
        return {self.vma_col}

    def entry_is_valid(self, entry_bar: BarSnapshot, direction: str) -> bool:
        value = entry_bar.values.get(self.vma_col)
        return value is not None and not np.isnan(value)

    def should_exit(self, trade: OpenTrade, bar: BarSnapshot) -> ExitDecision | None:
        vma_value = bar.values.get(self.vma_col)
        if vma_value is None or np.isnan(vma_value):
            return None

        if trade.direction == "long" and bar.high < vma_value:
            return ExitDecision(reason="vma_stop")
        if trade.direction == "short" and bar.low > vma_value:
            return ExitDecision(reason="vma_stop")
        return None


@dataclass(frozen=True, slots=True)
class FixedRewardRiskExitPolicy(ExitPolicy):
    """Exit at a fixed stop distance or fixed reward multiple from entry."""

    stop_loss: float
    reward_multiple: float = 2.0
    policy_name: str = "fixed_rr"

    def __post_init__(self) -> None:
        if self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive for fixed_rr exits.")
        if self.reward_multiple <= 0:
            raise ValueError("reward_multiple must be positive for fixed_rr exits.")

    def should_exit(self, trade: OpenTrade, bar: BarSnapshot) -> ExitDecision | None:
        reward_distance = self.stop_loss * self.reward_multiple

        if trade.direction == "long":
            stop_price = trade.entry_price - self.stop_loss
            target_price = trade.entry_price + reward_distance
            # Conservative ordering when both thresholds are touched intra-bar.
            if bar.low <= stop_price:
                return ExitDecision(reason="stop_loss", exit_price=stop_price)
            if bar.high >= target_price:
                return ExitDecision(reason="take_profit", exit_price=target_price)
            return None

        stop_price = trade.entry_price + self.stop_loss
        target_price = trade.entry_price - reward_distance
        if bar.high >= stop_price:
            return ExitDecision(reason="stop_loss", exit_price=stop_price)
        if bar.low <= target_price:
            return ExitDecision(reason="take_profit", exit_price=target_price)
        return None


@dataclass(frozen=True, slots=True)
class FixedPercentRewardRiskExitPolicy(ExitPolicy):
    """Exit at a fixed percent stop and reward multiple from entry."""

    stop_loss_pct: float
    reward_multiple: float = 2.0
    policy_name: str = "fixed_rr_underlying"

    def __post_init__(self) -> None:
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive for fixed_rr_underlying exits.")
        if self.reward_multiple <= 0:
            raise ValueError("reward_multiple must be positive for fixed_rr_underlying exits.")

    def should_exit(self, trade: OpenTrade, bar: BarSnapshot) -> ExitDecision | None:
        risk_distance = trade.entry_price * self.stop_loss_pct
        reward_distance = risk_distance * self.reward_multiple

        if trade.direction == "long":
            stop_price = trade.entry_price - risk_distance
            target_price = trade.entry_price + reward_distance
            if bar.low <= stop_price:
                return ExitDecision(reason="stop_loss_underlying", exit_price=stop_price)
            if bar.high >= target_price:
                return ExitDecision(reason="take_profit_underlying", exit_price=target_price)
            return None

        stop_price = trade.entry_price + risk_distance
        target_price = trade.entry_price - reward_distance
        if bar.high >= stop_price:
            return ExitDecision(reason="stop_loss_underlying", exit_price=stop_price)
        if bar.low <= target_price:
            return ExitDecision(reason="take_profit_underlying", exit_price=target_price)
        return None


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
    Bar-by-bar trade simulator with pluggable exit policies.

    Defaults to the legacy VMA trailing-stop behavior, but can also run
    fixed reward/risk or other explicit exit policies.
    """

    def __init__(
        self,
        vma_5m_col: str = "vma_10_5m",
        market_close: dt_time = dt_time(15, 59),
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        self.market_close = market_close
        self.exit_policy = exit_policy or VmaTrailingExitPolicy(vma_col=vma_5m_col)
        self.vma_5m_col = (
            self.exit_policy.vma_col
            if isinstance(self.exit_policy, VmaTrailingExitPolicy)
            else vma_5m_col
        )

    def simulate(self, df: pl.DataFrame) -> SimulationResult:
        """
        Run the simulation on a DataFrame that has 'signal',
        'signal_direction', and the 5-min VMA column.

        Returns a SimulationResult with all trades.
        """
        required = {
            "timestamp",
            "close",
            "high",
            "low",
            "signal",
            "signal_direction",
            *self.exit_policy.required_columns,
        }
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
        # Pre-compute bar times for EOD check
        bar_times = df.select(et_time_expr("timestamp").alias("t"))["t"].to_list()

        # Pre-compute dates for session boundary tracking
        dates = df.select(et_date_expr("timestamp").alias("d"))["d"].to_list()
        policy_arrays = {
            column: df[column].to_numpy()
            for column in self.exit_policy.required_columns
        }

        n = len(df)
        trades: List[Trade] = []
        i = 0

        def bar_snapshot(idx: int) -> BarSnapshot:
            return BarSnapshot(
                idx=idx,
                timestamp=timestamps[idx],
                close=float(close[idx]),
                high=float(high[idx]),
                low=float(low[idx]),
                bar_time=bar_times[idx],
                trade_date=dates[idx],
                values={
                    column: float(policy_arrays[column][idx])
                    if policy_arrays[column][idx] is not None
                    else None
                    for column in self.exit_policy.required_columns
                },
            )

        while i < n:
            # Look for signal entry
            if not signal[i] or direction[i] is None:
                i += 1
                continue

            entry_bar = bar_snapshot(i)
            if not self.exit_policy.entry_is_valid(entry_bar, str(direction[i])):
                i += 1
                continue

            entry_idx = i
            entry_time = timestamps[i]
            entry_price = close[i]
            entry_direction = direction[i]
            entry_date = dates[i]
            open_trade = OpenTrade(
                entry_idx=entry_idx,
                entry_time=entry_time,
                direction=str(entry_direction),
                entry_price=float(entry_price),
                entry_date=entry_date,
                entry_values=entry_bar.values,
            )

            # Walk forward to find exit
            j = i + 1
            exit_reason = "eod"
            exit_price_override: float | None = None

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

                decision = self.exit_policy.should_exit(open_trade, bar_snapshot(j))
                if decision is not None:
                    exit_reason = decision.reason
                    exit_price_override = decision.exit_price
                    break

                j += 1

            # Clamp to valid index
            exit_idx = min(j, n - 1)
            exit_price = exit_price_override if exit_price_override is not None else close[exit_idx]
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
                vma_5m_at_entry=round(open_trade.entry_values.get(self.vma_5m_col) or 0.0, 4),
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
