"""
Market Impulse Strategy

Multi-timeframe entry strategy based on the TOS Market Pulse indicator.

Regime detection (5-min chart):
  - Bullish: VWMA(8) > VWMA(21) > VWMA(34)
  - Bearish: VWMA(8) < VWMA(21) < VWMA(34)

Entry trigger (1-min chart):
  - Long:  Bullish 5-min regime + price cross-and-reclaims VMA from above
           (bar low dips to/below VMA, close recovers above VMA)
  - Short: Bearish 5-min regime + price cross-and-reclaims VMA from below
           (bar high reaches to/above VMA, close falls back below VMA)

Time filter:
  - Only enter between market_open + buffer and market_open + max_window
  - Default: 9:33 – 10:30 ET
"""

from __future__ import annotations

from datetime import time
from typing import Optional

import polars as pl
from loguru import logger

from src.strategy.base import BaseStrategy


class MarketImpulseStrategy(BaseStrategy):
    """Multi-timeframe Market Impulse strategy with cross-and-reclaim entry."""

    def __init__(
        self,
        entry_buffer_minutes: int = 3,
        entry_window_minutes: int = 60,
        market_open_hour: int = 9,
        market_open_minute: int = 30,
        vma_col: str = "vma_10",
        regime_col: str = "impulse_regime_5m",
    ) -> None:
        self.entry_buffer_minutes = entry_buffer_minutes
        self.entry_window_minutes = entry_window_minutes
        self.market_open = time(market_open_hour, market_open_minute)
        self.vma_col = vma_col
        self.regime_col = regime_col

        # Compute the valid entry window bounds
        open_minutes = market_open_hour * 60 + market_open_minute
        self._entry_start = self._minutes_to_time(open_minutes + entry_buffer_minutes)
        self._entry_end = self._minutes_to_time(open_minutes + entry_window_minutes)

    @property
    def name(self) -> str:
        return "Market Impulse (Cross & Reclaim)"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the Market Impulse entry logic.

        Required columns:
          - timestamp (datetime)
          - close, high, low
          - vma_10  (1-min VMA from Market Impulse indicator)
          - impulse_regime_5m (5-min regime: "bullish" / "bearish" / "neutral")
        """
        required = {"timestamp", "close", "high", "low", self.vma_col, self.regime_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns: {missing}"
            )

        # ── Time Filter ─────────────────────────────────────────────────
        # Extract time-of-day from timestamp (in ET – data should be ET)
        df = df.with_columns(
            pl.col("timestamp").dt.time().alias("_bar_time")
        )

        time_filter = (
            (pl.col("_bar_time") >= self._entry_start)
            & (pl.col("_bar_time") <= self._entry_end)
        )

        # ── Cross-and-Reclaim Logic ─────────────────────────────────────
        # Long: bar low dips to/below VMA AND close recovers above VMA
        long_cross_reclaim = (
            (pl.col("low") <= pl.col(self.vma_col))
            & (pl.col("close") > pl.col(self.vma_col))
        )

        # Short: bar high reaches to/above VMA AND close falls below VMA
        short_cross_reclaim = (
            (pl.col("high") >= pl.col(self.vma_col))
            & (pl.col("close") < pl.col(self.vma_col))
        )

        # ── Regime Filter ───────────────────────────────────────────────
        bullish_regime = pl.col(self.regime_col) == "bullish"
        bearish_regime = pl.col(self.regime_col) == "bearish"

        # ── Combine Signals ─────────────────────────────────────────────
        long_signal = time_filter & bullish_regime & long_cross_reclaim
        short_signal = time_filter & bearish_regime & short_cross_reclaim

        df = df.with_columns([
            (long_signal | short_signal).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ])

        # Clean up temp column
        df = df.drop("_bar_time")

        # Log summary
        total_signals = df.filter(pl.col("signal")).height
        long_count = df.filter(pl.col("signal_direction") == "long").height
        short_count = df.filter(pl.col("signal_direction") == "short").height
        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) "
            "out of {} bars (window: {} – {})",
            self.name,
            total_signals,
            long_count,
            short_count,
            len(df),
            self._entry_start,
            self._entry_end,
        )
        return df

    @staticmethod
    def _minutes_to_time(total_minutes: int) -> time:
        """Convert total minutes since midnight to a time object."""
        return time(total_minutes // 60, total_minutes % 60)

    def __repr__(self) -> str:
        return (
            f"<MarketImpulseStrategy window={self._entry_start}-{self._entry_end} "
            f"vma={self.vma_col} regime={self.regime_col}>"
        )
