"""
Elastic Band Reversion Strategy

Mean-reversion hypothesis:
- If price stretches far from VPOC and short-term kinematics show exhaustion,
  expect snap-back toward value.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from src.strategy.base import BaseStrategy


class ElasticBandReversionStrategy(BaseStrategy):
    """Directional mean-reversion strategy around VPOC stretch extremes."""

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        z_score_window: int = 240,
        use_directional_mass: bool = True,
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.z_score_window = z_score_window
        self.use_directional_mass = use_directional_mass

    @property
    def name(self) -> str:
        dm = "+dm" if self.use_directional_mass else "-dm"
        return f"Elastic Band z={self.z_score_threshold}/w={self.z_score_window}{dm}"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {
            "close",
            "vpoc_4h",
            "velocity_1m",
            "jerk_1m",
        }
        if self.use_directional_mass:
            required.add("directional_mass")
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        df = df.with_columns([
            ((pl.col("close") - pl.col("vpoc_4h")) / pl.col("vpoc_4h")).alias("_dist_pct"),
        ]).with_columns([
            pl.col("_dist_pct")
            .rolling_mean(window_size=self.z_score_window)
            .alias("_dist_mean"),
            pl.col("_dist_pct")
            .rolling_std(window_size=self.z_score_window)
            .alias("_dist_std"),
        ]).with_columns([
            pl.when(pl.col("_dist_std").is_not_null() & (pl.col("_dist_std") > 0))
            .then((pl.col("_dist_pct") - pl.col("_dist_mean")) / pl.col("_dist_std"))
            .otherwise(pl.lit(None))
            .alias("_z_score"),
        ])

        long_signal = (
            (pl.col("_z_score") <= -self.z_score_threshold)
            & (pl.col("velocity_1m") < 0)
            & (pl.col("jerk_1m") > 0)
            & (pl.col("directional_mass") > 0 if self.use_directional_mass else pl.lit(True))
        )

        short_signal = (
            (pl.col("_z_score") >= self.z_score_threshold)
            & (pl.col("velocity_1m") > 0)
            & (pl.col("jerk_1m") < 0)
            & (pl.col("directional_mass") < 0 if self.use_directional_mass else pl.lit(True))
        )

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ]).drop(["_dist_pct", "_dist_mean", "_dist_std", "_z_score"])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) out of {} bars "
            "[dm={}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_directional_mass,
        )
        return df
