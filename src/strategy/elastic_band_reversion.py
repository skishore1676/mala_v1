"""
Elastic Band Reversion Strategy

Mean-reversion hypothesis:
- If price stretches far from VPOC and short-term kinematics show exhaustion,
  expect snap-back toward value.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger

from src.newton.transforms import jerk_column_name, validate_periods_back, velocity_column_name
from src.research.models import DomainSpec, ObjectiveSpec, ParameterSpec, StrategySearchSpec
from src.strategy.base import BaseStrategy


class ElasticBandReversionStrategy(BaseStrategy):
    """Directional mean-reversion strategy around VPOC stretch extremes."""

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        z_score_window: int = 240,
        use_directional_mass: bool = True,
        use_jerk_confirmation: bool = True,
        kinematic_periods_back: int = 1,
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.z_score_window = z_score_window
        self.use_directional_mass = use_directional_mass
        self.use_jerk_confirmation = use_jerk_confirmation
        self.kinematic_periods_back = validate_periods_back(kinematic_periods_back)

    @property
    def name(self) -> str:
        dm = "+dm" if self.use_directional_mass else "-dm"
        return f"Elastic Band z={self.z_score_threshold}/w={self.z_score_window}{dm}"

    @property
    def required_features(self) -> set[str]:
        required = {
            "close",
            "vpoc_4h",
            velocity_column_name(self.kinematic_periods_back),
        }
        if self.use_jerk_confirmation:
            required.add(jerk_column_name(self.kinematic_periods_back))
        if self.use_directional_mass:
            required.add("directional_mass")
        return required

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            "z_score_threshold": [1.0, 1.25, 1.75, 2.0, 2.5, 3.0],
            "z_score_window": [120, 240, 360],
            "kinematic_periods_back": [1, 3, 5],
            "use_directional_mass": [True, False],
            "use_jerk_confirmation": [True, False],
        }

    @property
    def evaluation_mode(self) -> str:
        return "directional"

    @property
    def search_spec(self) -> StrategySearchSpec:
        return StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="z_score_threshold",
                    type="discrete",
                    domain=DomainSpec(values=[1.0, 1.25, 1.75, 2.0, 2.5, 3.0]),
                    default=self.z_score_threshold,
                    prior_center=1.25,
                ),
                ParameterSpec(
                    name="z_score_window",
                    type="discrete",
                    domain=DomainSpec(values=[120, 240, 360]),
                    default=self.z_score_window,
                    prior_center=240,
                ),
                ParameterSpec(
                    name="kinematic_periods_back",
                    type="discrete",
                    domain=DomainSpec(values=[1, 3, 5]),
                    default=self.kinematic_periods_back,
                    prior_center=3,
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
            ],
            objective=ObjectiveSpec(
                primary_metric="avg_test_exp_r",
                minimum_signals=20,
                tie_breakers=["pct_positive_oos_windows", "oos_signals"],
            ),
        )

    def strategy_config(self) -> dict[str, Any]:
        return {
            "z_score_threshold": self.z_score_threshold,
            "z_score_window": self.z_score_window,
            "kinematic_periods_back": self.kinematic_periods_back,
            "use_directional_mass": self.use_directional_mass,
            "use_jerk_confirmation": self.use_jerk_confirmation,
        }

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        required = self.required_features
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}")
        velocity_col = velocity_column_name(self.kinematic_periods_back)
        jerk_col = jerk_column_name(self.kinematic_periods_back)
        long_jerk_gate = pl.col(jerk_col) > 0 if self.use_jerk_confirmation else pl.lit(True)
        short_jerk_gate = pl.col(jerk_col) < 0 if self.use_jerk_confirmation else pl.lit(True)

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
            & (pl.col(velocity_col) < 0)
            & long_jerk_gate
            & (pl.col("directional_mass") > 0 if self.use_directional_mass else pl.lit(True))
        )

        short_signal = (
            (pl.col("_z_score") >= self.z_score_threshold)
            & (pl.col(velocity_col) > 0)
            & short_jerk_gate
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
            "[dm={}, kinematic_periods_back={}]",
            self.name,
            total,
            longs,
            shorts,
            len(df),
            self.use_directional_mass,
            self.kinematic_periods_back,
        )
        return df
