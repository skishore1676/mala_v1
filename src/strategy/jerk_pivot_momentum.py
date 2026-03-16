"""
Jerk-Pivot Momentum Strategy

Hypothesis (from Analyst):
    The 3rd derivative of price (jerk) provides leading signals for momentum
    continuation when combined with VPOC volume confirmation. When price
    approaches a VPOC with positive velocity and acceleration, a jerk inflection
    (negative-to-positive transition) signals high-probability momentum continuation,
    while a positive-to-negative jerk signals exhaustion/reversal.

Entry Logic:
    LONG:
      - Price within vpoc_proximity_pct of VPOC (above or near it)
      - velocity_1m > 0  (price moving up)
      - accel_1m  > 0    (acceleration positive — trend strengthening)
      - jerk crosses from negative to positive (jerk_1m > 0 AND prev_jerk < 0)

    SHORT:
      - Price within vpoc_proximity_pct of VPOC (below or near it)
      - velocity_1m < 0  (price moving down)
      - accel_1m  < 0    (acceleration negative)
      - jerk crosses from positive to negative (jerk_1m < 0 AND prev_jerk > 0)

    Optional filters: volume gate, time-of-day filter.
"""

from __future__ import annotations

from datetime import time

import polars as pl
from loguru import logger

from src.config import settings
from src.strategy.base import BaseStrategy
from src.time_utils import et_time_expr


class JerkPivotMomentumStrategy(BaseStrategy):
    """
    Jerk-inflection momentum strategy anchored to rolling VPOC.
    Enters on jerk sign-change confirming velocity/acceleration alignment
    when price is in proximity of a volume point-of-control.
    """

    def __init__(
        self,
        vpoc_proximity_pct: float = 0.005,           # 0.5% from VPOC
        jerk_lookback: int = 20,                      # rolling window for jerk smoothing
        volume_multiplier: float = 1.0,               # minimum vol relative to MA
        volume_ma_period: int = settings.volume_ma_period,
        use_volume_filter: bool = True,
        use_time_filter: bool = True,
        session_start: time = time(9, 35),
        session_end: time = time(15, 30),
        strategy_label: str | None = None,
    ) -> None:
        self.vpoc_proximity_pct = vpoc_proximity_pct
        self.jerk_lookback = jerk_lookback
        self.volume_multiplier = volume_multiplier
        self.volume_ma_period = volume_ma_period
        self.use_volume_filter = use_volume_filter
        self.use_time_filter = use_time_filter
        self.session_start = session_start
        self.session_end = session_end
        self._strategy_label = strategy_label

    @property
    def name(self) -> str:
        if self._strategy_label:
            return self._strategy_label
        return (
            f"Jerk-Pivot Momentum vpoc={self.vpoc_proximity_pct:.3f}"
            f"/jl={self.jerk_lookback}"
        )

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = {
            "timestamp",
            "close",
            "velocity_1m",
            "accel_1m",
            "jerk_1m",
            "vpoc_4h",
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        vol_ma_col = f"volume_ma_{self.volume_ma_period}"

        # ── Smooth jerk over a short rolling window to reduce noise ──────
        df = df.with_columns(
            pl.col("jerk_1m")
            .rolling_mean(window_size=self.jerk_lookback)
            .alias("_jerk_smooth")
        )

        # Previous smoothed jerk (for crossover detection)
        df = df.with_columns(
            pl.col("_jerk_smooth").shift(1).alias("_prev_jerk_smooth")
        )

        # ── VPOC proximity gate ──────────────────────────────────────────
        # Price is "near" VPOC if within vpoc_proximity_pct on either side
        vpoc_dist = (pl.col("close") - pl.col("vpoc_4h")).abs() / pl.col("vpoc_4h")
        near_vpoc = (vpoc_dist <= self.vpoc_proximity_pct) & pl.col("vpoc_4h").is_not_null()

        # Price side relative to VPOC
        above_vpoc = pl.col("close") >= pl.col("vpoc_4h")
        below_vpoc = pl.col("close") <= pl.col("vpoc_4h")

        # ── Jerk inflection crossovers ────────────────────────────────────
        # Long trigger: jerk crosses from negative to positive
        jerk_cross_up = (
            (pl.col("_jerk_smooth") > 0)
            & (pl.col("_prev_jerk_smooth") <= 0)
            & pl.col("_jerk_smooth").is_not_null()
            & pl.col("_prev_jerk_smooth").is_not_null()
        )

        # Short trigger: jerk crosses from positive to negative
        jerk_cross_down = (
            (pl.col("_jerk_smooth") < 0)
            & (pl.col("_prev_jerk_smooth") >= 0)
            & pl.col("_jerk_smooth").is_not_null()
            & pl.col("_prev_jerk_smooth").is_not_null()
        )

        # ── Kinematic alignment ──────────────────────────────────────────
        long_kinematic = (pl.col("velocity_1m") > 0) & (pl.col("accel_1m") > 0)
        short_kinematic = (pl.col("velocity_1m") < 0) & (pl.col("accel_1m") < 0)

        # ── Volume gate ───────────────────────────────────────────────────
        volume_gate = (
            pl.col("volume") >= self.volume_multiplier * pl.col(vol_ma_col)
            if self.use_volume_filter
            else pl.lit(True)
        )

        # ── Time gate ─────────────────────────────────────────────────────
        if self.use_time_filter:
            time_gate = (
                (et_time_expr("timestamp") >= self.session_start)
                & (et_time_expr("timestamp") <= self.session_end)
            )
        else:
            time_gate = pl.lit(True)

        # ── Combine conditions ────────────────────────────────────────────
        long_signal = (
            near_vpoc
            & above_vpoc
            & long_kinematic
            & jerk_cross_up
            & volume_gate
            & time_gate
        )

        short_signal = (
            near_vpoc
            & below_vpoc
            & short_kinematic
            & jerk_cross_down
            & volume_gate
            & time_gate
        )

        df = df.with_columns([
            (long_signal | short_signal).fill_null(False).alias("signal"),
            pl.when(long_signal)
            .then(pl.lit("long"))
            .when(short_signal)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        ]).drop(["_jerk_smooth", "_prev_jerk_smooth"])

        total = df.filter(pl.col("signal")).height
        longs = df.filter(pl.col("signal_direction") == "long").height
        shorts = df.filter(pl.col("signal_direction") == "short").height

        logger.info(
            "Strategy '{}' generated {} signals ({} long, {} short) out of {} bars "
            "[vol_filter={}, vpoc_prox={:.3f}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_volume_filter,
            self.vpoc_proximity_pct,
        )
        return df
