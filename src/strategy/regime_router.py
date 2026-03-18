"""Regime Router Strategy

Routes entries to different sub-strategies based on market regime:
- Trend regime  -> Kinematic Ladder
- Compression regime -> Compression Breakout
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger

from src.strategy.base import BaseStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy


class RegimeRouterStrategy(BaseStrategy):
    """Route between momentum and breakout logic using simple regime states."""

    def __init__(
        self,
        kinematic: KinematicLadderStrategy | None = None,
        compression: CompressionBreakoutStrategy | None = None,
        vol_short_window: int = 20,
        vol_long_window: int = 60,
        trend_vel_window: int = 30,
        trend_vol_ratio: float = 1.0,
        compression_vol_ratio: float = 0.9,
        trend_velocity_floor: float = 0.015,
    ) -> None:
        self.kinematic = kinematic or KinematicLadderStrategy(
            regime_window=30,
            accel_window=8,
            volume_multiplier=1.0,
            use_time_filter=True,
        )
        self.compression = compression or CompressionBreakoutStrategy(
            compression_window=20,
            breakout_lookback=20,
            compression_factor=0.9,
            volume_multiplier=1.05,
            use_time_filter=True,
        )
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window
        self.trend_vel_window = trend_vel_window
        self.trend_vol_ratio = trend_vol_ratio
        self.compression_vol_ratio = compression_vol_ratio
        self.trend_velocity_floor = trend_velocity_floor

    @property
    def name(self) -> str:
        return "Regime Router (Kinematic + Compression)"

    @property
    def required_features(self) -> set[str]:
        return {"timestamp", "close", "velocity_1m"} | self.kinematic.required_features | self.compression.required_features

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "vol_short_window": [20],
            "vol_long_window": [60],
            "trend_vel_window": [30],
            "trend_vol_ratio": [1.0],
            "compression_vol_ratio": [0.9],
            "trend_velocity_floor": [0.015],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    def strategy_config(self) -> dict[str, Any]:
        return {
            "vol_short_window": self.vol_short_window,
            "vol_long_window": self.vol_long_window,
            "trend_vel_window": self.trend_vel_window,
            "trend_vol_ratio": self.trend_vol_ratio,
            "compression_vol_ratio": self.compression_vol_ratio,
            "trend_velocity_floor": self.trend_velocity_floor,
            "kinematic": self.kinematic.strategy_config(),
            "compression": self.compression.strategy_config(),
        }

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {"close", "velocity_1m", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        kin_df = self.kinematic.generate_signals(df.clone()).select([
            pl.col("signal").alias("_kin_signal"),
            pl.col("signal_direction").alias("_kin_direction"),
        ])
        comp_df = self.compression.generate_signals(df.clone()).select([
            pl.col("signal").alias("_comp_signal"),
            pl.col("signal_direction").alias("_comp_direction"),
        ])

        df = pl.concat([df, kin_df, comp_df], how="horizontal")

        df = df.with_columns([
            pl.col("close").rolling_std(window_size=self.vol_short_window).alias("_vol_s"),
            pl.col("close").rolling_std(window_size=self.vol_long_window).alias("_vol_l"),
            pl.col("velocity_1m").rolling_mean(window_size=self.trend_vel_window).abs().alias("_trend_vel"),
        ])

        trend_regime = (
            (pl.col("_vol_s") >= self.trend_vol_ratio * pl.col("_vol_l"))
            & (pl.col("_trend_vel") >= self.trend_velocity_floor)
        )
        compression_regime = pl.col("_vol_s") <= self.compression_vol_ratio * pl.col("_vol_l")

        long_signal = (
            (trend_regime & pl.col("_kin_signal") & (pl.col("_kin_direction") == "long"))
            | (compression_regime & pl.col("_comp_signal") & (pl.col("_comp_direction") == "long"))
        )
        short_signal = (
            (trend_regime & pl.col("_kin_signal") & (pl.col("_kin_direction") == "short"))
            | (compression_regime & pl.col("_comp_signal") & (pl.col("_comp_direction") == "short"))
        )

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
            pl.when(trend_regime)
            .then(pl.lit("trend"))
            .when(compression_regime)
            .then(pl.lit("compression"))
            .otherwise(pl.lit("neutral"))
            .alias("route_regime"),
        ]).drop(["_kin_signal", "_kin_direction", "_comp_signal", "_comp_direction", "_vol_s", "_vol_l", "_trend_vel"])

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
