"""
Compression Expansion Breakout Strategy

Hypothesis:
- Periods of volatility compression are followed by expansion moves.
- Trade breakouts only when trend bias and participation confirm.
"""

from __future__ import annotations

from datetime import time
from typing import Any

import polars as pl
from loguru import logger

from src.config import settings
from src.newton.transforms import validate_periods_back, velocity_column_name
from src.strategy.base import BaseStrategy
from src.time_utils import et_time_expr


class CompressionBreakoutStrategy(BaseStrategy):
    """Directional breakout strategy after volatility compression."""

    def __init__(
        self,
        compression_window: int = 20,
        breakout_lookback: int = 20,
        compression_factor: float = 0.8,
        volume_ma_period: int = settings.volume_ma_period,
        volume_multiplier: float = 1.2,
        use_time_filter: bool = True,
        use_volume_filter: bool = True,
        session_start: time = time(9, 40),
        session_end: time = time(15, 30),
        velocity_periods_back: int = 1,
    ) -> None:
        self.compression_window = compression_window
        self.breakout_lookback = breakout_lookback
        self.compression_factor = compression_factor
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.use_time_filter = use_time_filter
        self.use_volume_filter = use_volume_filter
        self.session_start = session_start
        self.session_end = session_end
        self.velocity_periods_back = validate_periods_back(velocity_periods_back)

    @property
    def name(self) -> str:
        return "Compression Expansion Breakout"

    @property
    def required_features(self) -> set[str]:
        return {
            "timestamp",
            "close",
            "high",
            "low",
            "ema_8",
            "ema_12",
            velocity_column_name(self.velocity_periods_back),
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "compression_window": [15, 20, 30],
            "breakout_lookback": [15, 20],
            "compression_factor": [0.7, 0.8, 0.9],
            "velocity_periods_back": [1, 3, 5],
            "use_volume_filter": [True, False],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    def strategy_config(self) -> dict[str, Any]:
        return {
            "compression_window": self.compression_window,
            "breakout_lookback": self.breakout_lookback,
            "compression_factor": self.compression_factor,
            "volume_ma_period": self.volume_ma_period,
            "volume_multiplier": self.volume_multiplier,
            "use_time_filter": self.use_time_filter,
            "use_volume_filter": self.use_volume_filter,
            "velocity_periods_back": self.velocity_periods_back,
            "session_start": self.session_start.isoformat(timespec="minutes"),
            "session_end": self.session_end.isoformat(timespec="minutes"),
        }

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = self.required_features
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        vol_ma_col = f"volume_ma_{self.volume_ma_period}"
        velocity_col = velocity_column_name(self.velocity_periods_back)

        df = df.with_columns([
            pl.col("close")
            .rolling_std(window_size=self.compression_window)
            .alias("_short_vol"),
            pl.col("close")
            .rolling_std(window_size=self.compression_window * 3)
            .alias("_long_vol"),
            pl.col("high")
            .rolling_max(window_size=self.breakout_lookback)
            .shift(1)
            .alias("_prior_high"),
            pl.col("low")
            .rolling_min(window_size=self.breakout_lookback)
            .shift(1)
            .alias("_prior_low"),
        ])

        compression_state = (
            (pl.col("_short_vol").shift(1) <= self.compression_factor * pl.col("_long_vol").shift(1))
            & pl.col("_short_vol").is_not_null()
            & pl.col("_long_vol").is_not_null()
        )

        volume_gate = (
            pl.col("volume") > self.volume_multiplier * pl.col(vol_ma_col)
            if self.use_volume_filter
            else pl.lit(True)
        )

        bullish_bias = pl.col("ema_8") > pl.col("ema_12")
        bearish_bias = pl.col("ema_8") < pl.col("ema_12")

        long_breakout = pl.col("close") > pl.col("_prior_high")
        short_breakout = pl.col("close") < pl.col("_prior_low")

        trigger_long = pl.col(velocity_col) > 0
        trigger_short = pl.col(velocity_col) < 0

        if self.use_time_filter:
            time_gate = (
                (et_time_expr("timestamp") >= self.session_start)
                & (et_time_expr("timestamp") <= self.session_end)
            )
        else:
            time_gate = pl.lit(True)

        long_signal = compression_state & bullish_bias & long_breakout & trigger_long & volume_gate & time_gate
        short_signal = compression_state & bearish_bias & short_breakout & trigger_short & volume_gate & time_gate

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ]).drop(["_short_vol", "_long_vol", "_prior_high", "_prior_low"])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) out of {} bars "
            "[vol_filter={}, velocity_periods_back={}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_volume_filter,
            self.velocity_periods_back,
        )
        return df
