"""
Elastic Band Reversion Strategy

Mean-reversion hypothesis:
- If price stretches far from VPOC and short-term kinematics show exhaustion,
  expect snap-back toward value.
"""

from __future__ import annotations

from typing import Optional

import polars as pl
from loguru import logger

from src.config import settings
from src.strategy.base import BaseStrategy


class ElasticBandReversionStrategy(BaseStrategy):
    """Directional mean-reversion strategy around VPOC stretch extremes."""

    def __init__(
        self,
        stretch_pct: float = 0.0025,
        volume_ma_period: int = settings.volume_ma_period,
        volume_multiplier: float = 1.2,
    ) -> None:
        self.stretch_pct = stretch_pct
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier

    @property
    def name(self) -> str:
        return "Elastic Band Reversion"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {
            "close",
            "vpoc_4h",
            "velocity_1m",
            "jerk_1m",
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        vol_ma_col = f"volume_ma_{self.volume_ma_period}"

        distance_pct = (pl.col("close") - pl.col("vpoc_4h")) / pl.col("vpoc_4h")
        volume_gate = pl.col("volume") > self.volume_multiplier * pl.col(vol_ma_col)

        long_signal = (
            (distance_pct <= -self.stretch_pct)
            & (pl.col("velocity_1m") < 0)
            & (pl.col("jerk_1m") > 0)
            & volume_gate
        )

        short_signal = (
            (distance_pct >= self.stretch_pct)
            & (pl.col("velocity_1m") > 0)
            & (pl.col("jerk_1m") < 0)
            & volume_gate
        )

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) out of {} bars",
            self.name,
            total,
            longs,
            shorts,
            len(df),
        )
        return df
