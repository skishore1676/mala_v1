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
from typing import Any

import polars as pl
from loguru import logger

from src.config import settings
from src.newton.transforms import (
    acceleration_column_name,
    jerk_column_name,
    validate_periods_back,
    velocity_column_name,
)
from src.research.models import (
    ConstraintSpec,
    DomainSpec,
    GatingCondition,
    ObjectiveSpec,
    ParameterSpec,
    StrategySearchSpec,
)
from src.strategy.base import BaseStrategy, coerce_time
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
        session_start: time | str = time(9, 35),
        session_end: time | str = time(15, 30),
        kinematic_periods_back: int = 1,
        strategy_label: str | None = None,
    ) -> None:
        self.vpoc_proximity_pct = vpoc_proximity_pct
        self.jerk_lookback = jerk_lookback
        self.volume_multiplier = volume_multiplier
        self.volume_ma_period = volume_ma_period
        self.use_volume_filter = use_volume_filter
        self.use_time_filter = use_time_filter
        self.session_start = coerce_time(session_start)
        self.session_end = coerce_time(session_end)
        self.kinematic_periods_back = validate_periods_back(kinematic_periods_back)
        self._strategy_label = strategy_label

    @property
    def name(self) -> str:
        if self._strategy_label:
            return self._strategy_label
        return (
            f"Jerk-Pivot Momentum vpoc={self.vpoc_proximity_pct:.3f}"
            f"/jl={self.jerk_lookback}"
        )

    @property
    def required_features(self) -> set[str]:
        return {
            "timestamp",
            "close",
            velocity_column_name(self.kinematic_periods_back),
            acceleration_column_name(self.kinematic_periods_back),
            jerk_column_name(self.kinematic_periods_back),
            "vpoc_4h",
            "volume",
            f"volume_ma_{self.volume_ma_period}",
        }

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "vpoc_proximity_pct": [0.0015, 0.002, 0.003],
            "jerk_lookback": [8, 10, 12],
            "kinematic_periods_back": [1, 3, 5],
            "volume_multiplier": [1.0, 1.1, 1.2, 1.3],
            "use_volume_filter": [True, False],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    @property
    def search_spec(self) -> StrategySearchSpec:
        return StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="vpoc_proximity_pct",
                    type="discrete",
                    domain=DomainSpec(values=[0.0015, 0.002, 0.003]),
                    default=self.vpoc_proximity_pct,
                    prior_center=0.002,
                ),
                ParameterSpec(
                    name="jerk_lookback",
                    type="discrete",
                    domain=DomainSpec(values=[8, 10, 12]),
                    default=self.jerk_lookback,
                    prior_center=10,
                ),
                ParameterSpec(
                    name="kinematic_periods_back",
                    type="discrete",
                    domain=DomainSpec(values=[1, 3, 5]),
                    default=self.kinematic_periods_back,
                    prior_center=3,
                ),
                ParameterSpec(
                    name="use_volume_filter",
                    type="categorical",
                    domain=DomainSpec(values=[True, False]),
                    default=self.use_volume_filter,
                    prior_center=True,
                ),
                ParameterSpec(
                    name="volume_multiplier",
                    type="discrete",
                    domain=DomainSpec(values=[1.0, 1.1, 1.2, 1.3]),
                    default=self.volume_multiplier,
                    prior_center=1.3,
                ),
            ],
            constraints=ConstraintSpec(
                gating_conditions=[
                    GatingCondition(
                        parameter="volume_multiplier",
                        requires={"use_volume_filter": True},
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
            "vpoc_proximity_pct": self.vpoc_proximity_pct,
            "jerk_lookback": self.jerk_lookback,
            "volume_multiplier": self.volume_multiplier,
            "volume_ma_period": self.volume_ma_period,
            "use_volume_filter": self.use_volume_filter,
            "use_time_filter": self.use_time_filter,
            "kinematic_periods_back": self.kinematic_periods_back,
            "session_start": self.session_start.isoformat(timespec="minutes"),
            "session_end": self.session_end.isoformat(timespec="minutes"),
            "strategy_label": self._strategy_label,
        }

    def search_config(self) -> dict[str, Any]:
        config = self.strategy_config()
        if not self.use_volume_filter:
            config.pop("volume_multiplier", None)
        return config

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = self.required_features
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")

        vol_ma_col = f"volume_ma_{self.volume_ma_period}"
        velocity_col = velocity_column_name(self.kinematic_periods_back)
        accel_col = acceleration_column_name(self.kinematic_periods_back)
        jerk_col = jerk_column_name(self.kinematic_periods_back)

        # ── Smooth jerk over a short rolling window to reduce noise ──────
        df = df.with_columns(
            pl.col(jerk_col)
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
        long_kinematic = (pl.col(velocity_col) > 0) & (pl.col(accel_col) > 0)
        short_kinematic = (pl.col(velocity_col) < 0) & (pl.col(accel_col) < 0)

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
            "[vol_filter={}, vpoc_prox={:.3f}, jerk_lookback={}, kinematic_periods_back={}, "
            "volume_multiplier={:.2f}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_volume_filter,
            self.vpoc_proximity_pct,
            self.jerk_lookback,
            self.kinematic_periods_back,
            self.volume_multiplier,
        )
        return df
