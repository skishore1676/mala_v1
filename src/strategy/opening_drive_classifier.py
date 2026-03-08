"""
Opening Drive Classifier Strategy

Idea:
- Classify the first N minutes after open as an opening drive up/down.
- Trade either continuation (range expansion) or failure (mean-reverting flip).
"""

from __future__ import annotations

from datetime import time, timedelta, datetime

import polars as pl
from loguru import logger

from src.strategy.base import BaseStrategy
from src.time_utils import et_date_expr, et_time_expr


def _time_plus_minutes(base: time, minutes: int) -> time:
    anchor = datetime(2000, 1, 1, base.hour, base.minute)
    shifted = anchor + timedelta(minutes=minutes)
    return shifted.time()


class OpeningDriveClassifierStrategy(BaseStrategy):
    """Classify opening drive and trigger continuation/failure directional entries."""

    def __init__(
        self,
        market_open: time = time(9, 30),
        opening_window_minutes: int = 25,
        entry_start_offset_minutes: int = 25,
        entry_end_offset_minutes: int = 120,
        min_drive_return_pct: float = 0.0015,
        breakout_buffer_pct: float = 0.0,
        volume_multiplier: float = 1.2,
        require_directional_mass: bool = True,
        allow_long: bool = True,
        allow_short: bool = True,
        enable_continue: bool = True,
        enable_fail: bool = True,
        strategy_label: str | None = None,
    ) -> None:
        self.market_open = market_open
        self.opening_window_minutes = opening_window_minutes
        self.entry_start_offset_minutes = entry_start_offset_minutes
        self.entry_end_offset_minutes = entry_end_offset_minutes
        self.min_drive_return_pct = min_drive_return_pct
        self.breakout_buffer_pct = breakout_buffer_pct
        self.volume_multiplier = volume_multiplier
        self.require_directional_mass = require_directional_mass
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.enable_continue = enable_continue
        self.enable_fail = enable_fail
        self.strategy_label = strategy_label

    @property
    def name(self) -> str:
        return self.strategy_label or "Opening Drive Classifier"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "accel_1m",
            "jerk_1m",
        }
        if self.require_directional_mass:
            required.add("directional_mass")
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        opening_end = _time_plus_minutes(self.market_open, self.opening_window_minutes)
        entry_start = _time_plus_minutes(self.market_open, self.entry_start_offset_minutes)
        entry_end = _time_plus_minutes(self.market_open, self.entry_end_offset_minutes)

        in_opening_window = (
            (et_time_expr("timestamp") >= self.market_open)
            & (et_time_expr("timestamp") < opening_end)
        )
        in_entry_window = (
            (et_time_expr("timestamp") >= entry_start)
            & (et_time_expr("timestamp") <= entry_end)
        )

        df = df.with_columns([
            et_date_expr("timestamp").alias("_trade_date"),
        ]).with_columns([
            pl.col("open")
            .filter(in_opening_window)
            .first()
            .over("_trade_date")
            .alias("_opening_open"),
            pl.col("close")
            .filter(in_opening_window)
            .last()
            .over("_trade_date")
            .alias("_opening_close"),
            pl.col("high")
            .filter(in_opening_window)
            .max()
            .over("_trade_date")
            .alias("_opening_high"),
            pl.col("low")
            .filter(in_opening_window)
            .min()
            .over("_trade_date")
            .alias("_opening_low"),
            pl.col("volume")
            .filter(in_opening_window)
            .mean()
            .over("_trade_date")
            .alias("_opening_vol_mean"),
        ]).with_columns([
            ((pl.col("_opening_close") - pl.col("_opening_open")) / pl.col("_opening_open"))
            .alias("_opening_return"),
            ((pl.col("_opening_high") + pl.col("_opening_low")) / 2.0).alias("_opening_mid"),
        ]).with_columns([
            pl.when(pl.col("_opening_return") >= self.min_drive_return_pct)
            .then(pl.lit("up"))
            .when(pl.col("_opening_return") <= -self.min_drive_return_pct)
            .then(pl.lit("down"))
            .otherwise(pl.lit(None))
            .alias("_drive_direction"),
        ])

        volume_gate = pl.col("volume") > (
            self.volume_multiplier * pl.col("_opening_vol_mean")
        )

        long_mass_gate = (
            pl.col("directional_mass") > 0 if self.require_directional_mass else pl.lit(True)
        )
        short_mass_gate = (
            pl.col("directional_mass") < 0 if self.require_directional_mass else pl.lit(True)
        )

        continue_long = (
            in_entry_window
            & (pl.col("_drive_direction") == "up")
            & (pl.col("close") >= pl.col("_opening_high") * (1.0 + self.breakout_buffer_pct))
            & (pl.col("accel_1m") > 0)
            & (pl.col("jerk_1m") > 0)
            & volume_gate
            & long_mass_gate
        )
        continue_short = (
            in_entry_window
            & (pl.col("_drive_direction") == "down")
            & (pl.col("close") <= pl.col("_opening_low") * (1.0 - self.breakout_buffer_pct))
            & (pl.col("accel_1m") < 0)
            & (pl.col("jerk_1m") < 0)
            & volume_gate
            & short_mass_gate
        )
        fail_long = (
            in_entry_window
            & (pl.col("_drive_direction") == "down")
            & (pl.col("close") > pl.col("_opening_mid"))
            & (pl.col("accel_1m") > 0)
            & (pl.col("jerk_1m") > 0)
            & volume_gate
            & long_mass_gate
        )
        fail_short = (
            in_entry_window
            & (pl.col("_drive_direction") == "up")
            & (pl.col("close") < pl.col("_opening_mid"))
            & (pl.col("accel_1m") < 0)
            & (pl.col("jerk_1m") < 0)
            & volume_gate
            & short_mass_gate
        )

        long_raw = (
            (continue_long if self.enable_continue else pl.lit(False))
            | (fail_long if self.enable_fail else pl.lit(False))
        )
        short_raw = (
            (continue_short if self.enable_continue else pl.lit(False))
            | (fail_short if self.enable_fail else pl.lit(False))
        )
        if not self.allow_long:
            long_raw = pl.lit(False)
        if not self.allow_short:
            short_raw = pl.lit(False)

        # De-duplicate to first long and first short signal per day.
        df = df.with_columns([
            long_raw.fill_null(False).alias("_long_raw"),
            short_raw.fill_null(False).alias("_short_raw"),
            continue_long.fill_null(False).alias("_continue_long"),
            continue_short.fill_null(False).alias("_continue_short"),
            fail_long.fill_null(False).alias("_fail_long"),
            fail_short.fill_null(False).alias("_fail_short"),
        ]).with_columns([
            (
                pl.col("_long_raw")
                & (pl.col("_long_raw").cast(pl.Int64).cum_sum().over("_trade_date") == 1)
            ).alias("_long_signal"),
            (
                pl.col("_short_raw")
                & (pl.col("_short_raw").cast(pl.Int64).cum_sum().over("_trade_date") == 1)
            ).alias("_short_signal"),
        ]).with_columns([
            (pl.col("_long_signal") | pl.col("_short_signal")).alias("signal"),
            pl.when(pl.col("_long_signal"))
            .then(pl.lit("long"))
            .when(pl.col("_short_signal"))
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
            pl.when(pl.col("_long_signal") & pl.col("_continue_long"))
            .then(pl.lit("continue"))
            .when(pl.col("_short_signal") & pl.col("_continue_short"))
            .then(pl.lit("continue"))
            .when(pl.col("_long_signal") & pl.col("_fail_long"))
            .then(pl.lit("fail"))
            .when(pl.col("_short_signal") & pl.col("_fail_short"))
            .then(pl.lit("fail"))
            .otherwise(pl.lit(None))
            .alias("opening_drive_mode"),
            pl.col("_drive_direction").alias("opening_drive_direction"),
        ]).drop([
            "_trade_date",
            "_opening_open",
            "_opening_close",
            "_opening_high",
            "_opening_low",
            "_opening_vol_mean",
            "_opening_return",
            "_opening_mid",
            "_drive_direction",
            "_long_raw",
            "_short_raw",
            "_continue_long",
            "_continue_short",
            "_fail_long",
            "_fail_short",
            "_long_signal",
            "_short_signal",
        ])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height
        cont = df.filter((pl.col("signal")) & (pl.col("opening_drive_mode") == "continue")).height
        fail = df.filter((pl.col("signal")) & (pl.col("opening_drive_mode") == "fail")).height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short; {} continue, {} fail)",
            self.name,
            total,
            longs,
            shorts,
            cont,
            fail,
        )
        return df
