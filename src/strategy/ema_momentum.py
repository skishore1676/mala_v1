"""
EMA Momentum Strategy

Implements the PRD's "EMA Momentum" user story:

  Logic Gate 1 (Trend):    EMA(4) > EMA(8) > EMA(12)     → Velocity is Positive
  Logic Gate 2 (Location): Price > VPOC                   → Price escaped Gravity
  Logic Gate 3 (Force):    Volume > MA_Volume(20)         → Mass sustains momentum

All three gates must be TRUE simultaneously for a signal.
"""

from __future__ import annotations

from typing import Any, List, Optional

import polars as pl
from loguru import logger

from src.config import settings
from src.strategy.base import BaseStrategy


class EMAMomentumStrategy(BaseStrategy):
    """Configurable EMA-stack momentum strategy."""

    def __init__(
        self,
        ema_periods: Optional[List[int]] = None,
        volume_ma_period: int = settings.volume_ma_period,
    ) -> None:
        self.ema_periods = sorted(ema_periods or list(settings.ema_periods))
        self.volume_ma_period = volume_ma_period

    @property
    def name(self) -> str:
        periods_str = "/".join(str(p) for p in self.ema_periods)
        return f"EMA Momentum ({periods_str})"

    @property
    def required_features(self) -> set[str]:
        ema_cols = {f"ema_{p}" for p in self.ema_periods}
        return ema_cols | {
            "close",
            "vpoc_4h",
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "ema_periods": [list(settings.ema_periods)],
            "volume_ma_period": [self.volume_ma_period],
        }

    def strategy_config(self) -> dict[str, Any]:
        return {
            "ema_periods": list(self.ema_periods),
            "volume_ma_period": self.volume_ma_period,
        }

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the three logic gates and add a 'signal' column.

        Requirements on the input DataFrame:
          - ema_{p} columns for each period in self.ema_periods
          - vpoc_4h column
          - volume column
          - volume_ma_{self.volume_ma_period} column
          - close column
        """
        ema_cols = [f"ema_{p}" for p in self.ema_periods]
        vol_ma_col = f"volume_ma_{self.volume_ma_period}"

        required = self.required_features
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Strategy '{self.name}' requires columns: {missing}"
            )

        # Gate 1 – Trend: EMA stack ordered (fastest > … > slowest)
        trend_exprs = [
            pl.col(ema_cols[i]) > pl.col(ema_cols[i + 1])
            for i in range(len(ema_cols) - 1)
        ]
        trend_gate = trend_exprs[0]
        for expr in trend_exprs[1:]:
            trend_gate = trend_gate & expr

        # Gate 2 – Location: Price above VPOC (escaped gravity)
        location_gate = pl.col("close") > pl.col("vpoc_4h")

        # Gate 3 – Force: Volume above its moving average
        force_gate = pl.col("volume") > pl.col(vol_ma_col)

        df = df.with_columns(
            (trend_gate & location_gate & force_gate).alias("signal")
        )

        signal_count = df.filter(pl.col("signal")).height
        logger.info(
            "Strategy '{}' generated {} signals out of {} bars",
            self.name,
            signal_count,
            len(df),
        )
        return df
