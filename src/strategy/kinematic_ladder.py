"""
Kinematic Ladder Strategy

Multi-timeframe-inspired momentum strategy:
- Higher-timeframe bias from rolling velocity/acceleration trend.
- Lower-timeframe trigger from 1-min velocity + jerk alignment.
"""

from __future__ import annotations

from datetime import time

import polars as pl
from loguru import logger

from src.config import settings
from src.strategy.base import BaseStrategy
from src.time_utils import et_time_expr


class KinematicLadderStrategy(BaseStrategy):
    """Directional momentum strategy with regime/setup/trigger layering."""

    def __init__(
        self,
        regime_window: int = 30,
        accel_window: int = 10,
        volume_ma_period: int = settings.volume_ma_period,
        volume_multiplier: float = 1.1,
        use_time_filter: bool = True,
        use_volume_filter: bool = True,
        session_start: time = time(9, 35),
        session_end: time = time(15, 30),
    ) -> None:
        self.regime_window = regime_window
        self.accel_window = accel_window
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.use_time_filter = use_time_filter
        self.use_volume_filter = use_volume_filter
        self.session_start = session_start
        self.session_end = session_end

    @property
    def name(self) -> str:
        vol = "+vol" if self.use_volume_filter else "-vol"
        return f"Kinematic Ladder rw={self.regime_window}/aw={self.accel_window}{vol}"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {
            "timestamp",
            "close",
            "ema_4",
            "ema_8",
            "ema_12",
            "velocity_1m",
            "accel_1m",
            "jerk_1m",
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        vol_ma_col = f"volume_ma_{self.volume_ma_period}"

        df = df.with_columns([
            pl.col("velocity_1m")
            .rolling_mean(window_size=self.regime_window)
            .alias("_vel_regime"),
            pl.col("accel_1m")
            .rolling_mean(window_size=self.accel_window)
            .alias("_acc_regime"),
        ])

        bullish_regime = (
            (pl.col("_vel_regime") > 0)
            & (pl.col("_acc_regime") >= 0)
            & (pl.col("ema_4") > pl.col("ema_8"))
            & (pl.col("ema_8") > pl.col("ema_12"))
        )

        bearish_regime = (
            (pl.col("_vel_regime") < 0)
            & (pl.col("_acc_regime") <= 0)
            & (pl.col("ema_4") < pl.col("ema_8"))
            & (pl.col("ema_8") < pl.col("ema_12"))
        )

        pullback_long = (
            (pl.col("close") <= pl.col("ema_8"))
            & (pl.col("close") >= pl.col("ema_12"))
        )

        pullback_short = (
            (pl.col("close") >= pl.col("ema_8"))
            & (pl.col("close") <= pl.col("ema_12"))
        )

        trigger_long = (pl.col("velocity_1m") > 0) & (pl.col("jerk_1m") > 0)
        trigger_short = (pl.col("velocity_1m") < 0) & (pl.col("jerk_1m") < 0)

        volume_gate = (
            pl.col("volume") > self.volume_multiplier * pl.col(vol_ma_col)
            if self.use_volume_filter
            else pl.lit(True)
        )

        if self.use_time_filter:
            time_gate = (
                (et_time_expr("timestamp") >= self.session_start)
                & (et_time_expr("timestamp") <= self.session_end)
            )
        else:
            time_gate = pl.lit(True)

        long_signal = bullish_regime & pullback_long & trigger_long & volume_gate & time_gate
        short_signal = bearish_regime & pullback_short & trigger_short & volume_gate & time_gate

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ]).drop(["_vel_regime", "_acc_regime"])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) out of {} bars "
            "[vol_filter={}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_volume_filter,
        )
        return df
