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

import re
from datetime import time
from typing import Any, Optional

import polars as pl
from loguru import logger

from src.newton.market_impulse import (
    market_impulse_vwma_feature_spec,
    validate_vwma_periods,
)
from src.research.models import (
    ConstraintSpec,
    DomainSpec,
    MonotonicOrdering,
    ObjectiveSpec,
    ParameterSpec,
    StrategySearchSpec,
)
from src.strategy.base import BaseStrategy
from src.time_utils import et_time_expr


class MarketImpulseStrategy(BaseStrategy):
    """Multi-timeframe Market Impulse strategy with cross-and-reclaim entry."""

    def __init__(
        self,
        entry_buffer_minutes: int = 3,
        entry_window_minutes: int = 60,
        market_open_hour: int = 9,
        market_open_minute: int = 30,
        vma_col: str | None = None,
        regime_timeframe: str = "5m",
        regime_col: str | None = None,
        vma_length: int | None = None,
        vwma_periods: tuple[int, ...] = (8, 21, 34),
    ) -> None:
        self.entry_buffer_minutes = entry_buffer_minutes
        self.entry_window_minutes = entry_window_minutes
        self.market_open = time(market_open_hour, market_open_minute)
        self.vma_length = _resolve_vma_length(vma_length=vma_length, vma_col=vma_col)
        self.vwma_periods = validate_vwma_periods(tuple(vwma_periods))
        self.vma_col = vma_col or f"vma_{self.vma_length}"
        self.regime_timeframe = _resolve_regime_timeframe(
            regime_timeframe=regime_timeframe,
            regime_col=regime_col,
        )
        self.regime_col = regime_col or f"impulse_regime_{self.regime_timeframe}"

        # Compute the valid entry window bounds
        open_minutes = market_open_hour * 60 + market_open_minute
        self._entry_start = self._minutes_to_time(open_minutes + entry_buffer_minutes)
        self._entry_end = self._minutes_to_time(open_minutes + entry_window_minutes)

    @property
    def name(self) -> str:
        return "Market Impulse (Cross & Reclaim)"

    @property
    def required_features(self) -> set[str]:
        return {
            "timestamp",
            "close",
            "high",
            "low",
            self.vma_col,
            self.regime_col,
        }

    @property
    def feature_requests(self) -> set[str]:
        return {market_impulse_vwma_feature_spec(self.vwma_periods)}

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "entry_buffer_minutes": [3, 5],
            "entry_window_minutes": [45, 60, 90],
            "regime_timeframe": ["5m", "15m", "30m", "1h"],
            "vwma_periods": [(5, 13, 21), (8, 21, 34), (10, 20, 40)],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    @property
    def search_spec(self) -> StrategySearchSpec:
        return StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="entry_buffer_minutes",
                    type="discrete",
                    domain=DomainSpec(values=[3, 5]),
                    default=self.entry_buffer_minutes,
                    prior_center=3,
                ),
                ParameterSpec(
                    name="entry_window_minutes",
                    type="discrete",
                    domain=DomainSpec(values=[45, 60, 90]),
                    default=self.entry_window_minutes,
                    prior_center=60,
                ),
                ParameterSpec(
                    name="regime_timeframe",
                    type="categorical",
                    domain=DomainSpec(values=["5m", "15m", "30m", "1h"]),
                    default=self.regime_timeframe,
                    prior_center="5m",
                ),
                ParameterSpec(
                    name="vwma_periods",
                    type="categorical",
                    domain=DomainSpec(values=[(5, 13, 21), (8, 21, 34), (10, 20, 40)]),
                    default=tuple(self.vwma_periods),
                    prior_center=(8, 21, 34),
                ),
            ],
            constraints=ConstraintSpec(
                monotonic_ordering=[
                    MonotonicOrdering(
                        parameters=["entry_buffer_minutes", "entry_window_minutes"],
                        direction="strictly_ascending",
                    )
                ]
            ),
            objective=ObjectiveSpec(
                primary_metric="avg_test_exp_r",
                minimum_signals=20,
                tie_breakers=["pct_positive_oos_windows", "oos_signals"],
            ),
        )

    def strategy_config(self) -> dict[str, Any]:
        return {
            "entry_buffer_minutes": self.entry_buffer_minutes,
            "entry_window_minutes": self.entry_window_minutes,
            "market_open_hour": self.market_open.hour,
            "market_open_minute": self.market_open.minute,
            "vma_length": self.vma_length,
            "regime_timeframe": self.regime_timeframe,
            "vwma_periods": list(self.vwma_periods),
        }

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the Market Impulse entry logic.

        Required columns:
          - timestamp (datetime)
          - close, high, low
          - vma_10  (1-min VMA from Market Impulse indicator)
          - impulse_regime_5m (5-min regime: "bullish" / "bearish" / "neutral")
        """
        required = self.required_features
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns: {missing}"
            )

        # ── Time Filter ─────────────────────────────────────────────────
        # Extract ET time-of-day from stored UTC timestamp
        df = df.with_columns(
            et_time_expr("timestamp").alias("_bar_time")
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


def _resolve_vma_length(*, vma_length: int | None, vma_col: str | None) -> int:
    if vma_length is not None:
        return int(vma_length)
    if vma_col:
        match = re.fullmatch(r"vma_(\d+)", vma_col)
        if match:
            return int(match.group(1))
    return 10


def _resolve_regime_timeframe(*, regime_timeframe: str, regime_col: str | None) -> str:
    if regime_col and regime_timeframe == "5m":
        match = re.fullmatch(r"impulse_regime_(.+)", regime_col)
        if match:
            return match.group(1)
    return regime_timeframe
