"""
Opening Drive Classifier Strategy

Idea:
- Classify the first N minutes after open as an opening drive up/down.
- Trade either continuation (range expansion) or failure (mean-reverting flip).
"""

from __future__ import annotations

from datetime import time, timedelta, datetime
from typing import Any

import polars as pl
from loguru import logger

from src.newton.transforms import (
    acceleration_column_name,
    jerk_column_name,
    validate_periods_back,
)
from src.research.models import (
    ConstraintSpec,
    DomainSpec,
    GatingCondition,
    MonotonicOrdering,
    ObjectiveSpec,
    ParameterSpec,
    StrategySearchSpec,
)
from src.strategy.base import BaseStrategy, coerce_time
from src.time_utils import et_date_expr, et_time_expr


def _time_plus_minutes(base: time, minutes: int) -> time:
    anchor = datetime(2000, 1, 1, base.hour, base.minute)
    shifted = anchor + timedelta(minutes=minutes)
    return shifted.time()


class OpeningDriveClassifierStrategy(BaseStrategy):
    """Classify opening drive and trigger continuation/failure directional entries."""

    def __init__(
        self,
        market_open: time | str = time(9, 30),
        opening_window_minutes: int = 25,
        entry_start_offset_minutes: int = 25,
        entry_end_offset_minutes: int = 120,
        min_drive_return_pct: float = 0.0015,
        breakout_buffer_pct: float = 0.0,
        use_volume_filter: bool = True,
        volume_multiplier: float = 1.2,
        use_directional_mass: bool = True,
        use_jerk_confirmation: bool = True,
        use_regime_filter: bool = False,
        regime_timeframe: str = "5m",
        allow_long: bool = True,
        allow_short: bool = True,
        enable_continue: bool = True,
        enable_fail: bool = True,
        kinematic_periods_back: int = 1,
        strategy_label: str | None = None,
        require_directional_mass: bool | None = None,
    ) -> None:
        self.market_open = coerce_time(market_open)
        self.opening_window_minutes = opening_window_minutes
        self.entry_start_offset_minutes = entry_start_offset_minutes
        self.entry_end_offset_minutes = entry_end_offset_minutes
        self.min_drive_return_pct = min_drive_return_pct
        self.breakout_buffer_pct = breakout_buffer_pct
        self.use_volume_filter = use_volume_filter
        self.volume_multiplier = volume_multiplier
        self.use_directional_mass = (
            require_directional_mass
            if require_directional_mass is not None
            else use_directional_mass
        )
        self.use_jerk_confirmation = use_jerk_confirmation
        self.use_regime_filter = use_regime_filter
        self.regime_timeframe = regime_timeframe
        self.regime_col = f"impulse_regime_{self.regime_timeframe}"
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.enable_continue = enable_continue
        self.enable_fail = enable_fail
        self.kinematic_periods_back = validate_periods_back(kinematic_periods_back)
        self.strategy_label = strategy_label

    @property
    def name(self) -> str:
        return self.strategy_label or "Opening Drive Classifier"

    @property
    def required_features(self) -> set[str]:
        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            acceleration_column_name(self.kinematic_periods_back),
        }
        if self.use_volume_filter:
            required.add("volume")
        if self.use_jerk_confirmation:
            required.add(jerk_column_name(self.kinematic_periods_back))
        if self.use_directional_mass:
            required.add("directional_mass")
        if self.use_regime_filter:
            required.add(self.regime_col)
        return required

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "opening_window_minutes": [15, 20, 25, 30],
            "entry_start_offset_minutes": [20, 25, 30],
            "entry_end_offset_minutes": [90, 120],
            "min_drive_return_pct": [0.0015, 0.0020],
            "breakout_buffer_pct": [0.0, 0.0005],
            "kinematic_periods_back": [1, 3],
            "use_volume_filter": [True, False],
            "volume_multiplier": [1.2, 1.4],
            "use_directional_mass": [True, False],
            "use_jerk_confirmation": [True, False],
            "use_regime_filter": [False, True],
            "regime_timeframe": ["5m"],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    @property
    def search_spec(self) -> StrategySearchSpec:
        return StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="opening_window_minutes",
                    type="discrete",
                    domain=DomainSpec(values=[15, 20, 25, 30]),
                    default=self.opening_window_minutes,
                    prior_center=25,
                ),
                ParameterSpec(
                    name="entry_start_offset_minutes",
                    type="discrete",
                    domain=DomainSpec(values=[20, 25, 30]),
                    default=self.entry_start_offset_minutes,
                    prior_center=30,
                ),
                ParameterSpec(
                    name="entry_end_offset_minutes",
                    type="discrete",
                    domain=DomainSpec(values=[90, 120]),
                    default=self.entry_end_offset_minutes,
                    prior_center=120,
                ),
                ParameterSpec(
                    name="min_drive_return_pct",
                    type="discrete",
                    domain=DomainSpec(values=[0.0015, 0.0020]),
                    default=self.min_drive_return_pct,
                    prior_center=0.0015,
                ),
                ParameterSpec(
                    name="breakout_buffer_pct",
                    type="discrete",
                    domain=DomainSpec(values=[0.0, 0.0005]),
                    default=self.breakout_buffer_pct,
                    prior_center=0.0,
                ),
                ParameterSpec(
                    name="kinematic_periods_back",
                    type="discrete",
                    domain=DomainSpec(values=[1, 3]),
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
                    domain=DomainSpec(values=[1.2, 1.4]),
                    default=self.volume_multiplier,
                    prior_center=1.2,
                ),
                ParameterSpec(
                    name="use_directional_mass",
                    type="categorical",
                    domain=DomainSpec(values=[True, False]),
                    default=self.use_directional_mass,
                    prior_center=True,
                ),
                ParameterSpec(
                    name="use_jerk_confirmation",
                    type="categorical",
                    domain=DomainSpec(values=[True, False]),
                    default=self.use_jerk_confirmation,
                    prior_center=True,
                ),
                ParameterSpec(
                    name="use_regime_filter",
                    type="categorical",
                    domain=DomainSpec(values=[True, False]),
                    default=self.use_regime_filter,
                    prior_center=False,
                ),
                ParameterSpec(
                    name="regime_timeframe",
                    type="categorical",
                    domain=DomainSpec(values=["5m"]),
                    default=self.regime_timeframe,
                    prior_center="5m",
                ),
            ],
            constraints=ConstraintSpec(
                gating_conditions=[
                    GatingCondition(
                        parameter="volume_multiplier",
                        requires={"use_volume_filter": True},
                    ),
                    GatingCondition(
                        parameter="regime_timeframe",
                        requires={"use_regime_filter": True},
                    ),
                ],
                monotonic_ordering=[
                    MonotonicOrdering(
                        parameters=[
                            "opening_window_minutes",
                            "entry_start_offset_minutes",
                            "entry_end_offset_minutes",
                        ],
                        direction="strictly_ascending",
                    )
                ],
            ),
            objective=ObjectiveSpec(
                primary_metric="avg_test_exp_r",
                minimum_signals=20,
                tie_breakers=["pct_positive_oos_windows", "oos_signals"],
            ),
        )

    def strategy_config(self) -> dict[str, Any]:
        return {
            "market_open": self.market_open.isoformat(timespec="minutes"),
            "opening_window_minutes": self.opening_window_minutes,
            "entry_start_offset_minutes": self.entry_start_offset_minutes,
            "entry_end_offset_minutes": self.entry_end_offset_minutes,
            "min_drive_return_pct": self.min_drive_return_pct,
            "breakout_buffer_pct": self.breakout_buffer_pct,
            "use_volume_filter": self.use_volume_filter,
            "volume_multiplier": self.volume_multiplier,
            "use_directional_mass": self.use_directional_mass,
            "use_jerk_confirmation": self.use_jerk_confirmation,
            "use_regime_filter": self.use_regime_filter,
            "regime_timeframe": self.regime_timeframe,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "enable_continue": self.enable_continue,
            "enable_fail": self.enable_fail,
            "kinematic_periods_back": self.kinematic_periods_back,
            "strategy_label": self.strategy_label,
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
        accel_col = acceleration_column_name(self.kinematic_periods_back)
        jerk_col = jerk_column_name(self.kinematic_periods_back)

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
            (
                pl.col("volume")
                .filter(in_opening_window)
                .mean()
                .over("_trade_date")
                if self.use_volume_filter
                else pl.lit(None)
            ).alias("_opening_vol_mean"),
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

        volume_gate = (
            pl.col("volume") > (self.volume_multiplier * pl.col("_opening_vol_mean"))
            if self.use_volume_filter
            else pl.lit(True)
        )

        long_mass_gate = (
            pl.col("directional_mass") > 0 if self.use_directional_mass else pl.lit(True)
        )
        short_mass_gate = (
            pl.col("directional_mass") < 0 if self.use_directional_mass else pl.lit(True)
        )
        long_jerk_gate = (
            pl.col(jerk_col) > 0 if self.use_jerk_confirmation else pl.lit(True)
        )
        short_jerk_gate = (
            pl.col(jerk_col) < 0 if self.use_jerk_confirmation else pl.lit(True)
        )
        bullish_regime_gate = (
            pl.col(self.regime_col) == "bullish" if self.use_regime_filter else pl.lit(True)
        )
        bearish_regime_gate = (
            pl.col(self.regime_col) == "bearish" if self.use_regime_filter else pl.lit(True)
        )

        continue_long = (
            in_entry_window
            & (pl.col("_drive_direction") == "up")
            & (pl.col("close") >= pl.col("_opening_high") * (1.0 + self.breakout_buffer_pct))
            & (pl.col(accel_col) > 0)
            & long_jerk_gate
            & volume_gate
            & long_mass_gate
            & bullish_regime_gate
        )
        continue_short = (
            in_entry_window
            & (pl.col("_drive_direction") == "down")
            & (pl.col("close") <= pl.col("_opening_low") * (1.0 - self.breakout_buffer_pct))
            & (pl.col(accel_col) < 0)
            & short_jerk_gate
            & volume_gate
            & short_mass_gate
            & bearish_regime_gate
        )
        fail_long = (
            in_entry_window
            & (pl.col("_drive_direction") == "down")
            & (pl.col("close") > pl.col("_opening_mid"))
            & (pl.col(accel_col) > 0)
            & long_jerk_gate
            & volume_gate
            & long_mass_gate
            & bullish_regime_gate
        )
        fail_short = (
            in_entry_window
            & (pl.col("_drive_direction") == "up")
            & (pl.col("close") < pl.col("_opening_mid"))
            & (pl.col(accel_col) < 0)
            & short_jerk_gate
            & volume_gate
            & short_mass_gate
            & bearish_regime_gate
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
