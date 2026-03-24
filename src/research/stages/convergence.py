"""Reusable convergence-stage logic."""

from __future__ import annotations

import polars as pl

from src.research.stages.candidates import candidate_identity_columns


def parse_costs(cost_grid: str) -> list[float]:
    costs = [float(x.strip()) for x in cost_grid.split(",") if x.strip()]
    if not costs:
        raise ValueError("No valid values parsed from --cost-grid")
    return costs


def cost_tag(cost_r: float) -> str:
    return f"cost{int(round(cost_r * 1000)):03d}"


def build_gate_report(
    *,
    combined: pl.DataFrame,
    cost_count: int,
    gate_min_oos_windows: int,
    gate_min_oos_signals: int,
    gate_min_pct_positive: float,
    gate_min_exp_r: float,
) -> pl.DataFrame:
    group_cols = candidate_identity_columns(combined)
    return (
        combined.group_by(group_cols)
        .agg([
            pl.len().alias("observed_cost_points"),
            pl.col("oos_windows").min().alias("min_oos_windows"),
            pl.col("oos_signals").min().alias("min_oos_signals"),
            pl.col("avg_test_exp_r").min().alias("min_avg_test_exp_r"),
            pl.col("avg_test_exp_r").mean().alias("mean_avg_test_exp_r"),
            pl.col("pct_positive_oos_windows").min().alias("min_pct_positive_oos_windows"),
            pl.col("pct_positive_oos_windows").mean().alias("mean_pct_positive_oos_windows"),
            pl.col("avg_test_confidence").mean().alias("mean_test_confidence"),
        ])
        .with_columns([
            (pl.col("observed_cost_points") == cost_count).alias("has_all_cost_points"),
            (pl.col("min_oos_windows") >= gate_min_oos_windows).alias("passes_window_gate"),
            (pl.col("min_oos_signals") >= gate_min_oos_signals).alias("passes_signal_gate"),
            (pl.col("min_pct_positive_oos_windows") >= gate_min_pct_positive).alias("passes_stability_gate"),
            (pl.col("min_avg_test_exp_r") >= gate_min_exp_r).alias("passes_exp_gate"),
        ])
        .with_columns([
            (
                pl.col("has_all_cost_points")
                & pl.col("passes_window_gate")
                & pl.col("passes_signal_gate")
                & pl.col("passes_stability_gate")
                & pl.col("passes_exp_gate")
            ).alias("passes_all_gates")
        ])
        .with_columns([
            pl.when(pl.col("passes_all_gates"))
            .then(pl.lit("promote_to_holdout"))
            .when(pl.col("has_all_cost_points") & (pl.col("min_avg_test_exp_r") > 0))
            .then(pl.lit("candidate_needs_more_stability"))
            .otherwise(pl.lit("reject_or_rework"))
            .alias("decision"),
            (
                pl.col("min_avg_test_exp_r") * 1000
                + pl.col("min_pct_positive_oos_windows") * 100
                + pl.col("min_oos_signals") / 1000
            ).alias("score"),
        ])
        .sort(["passes_all_gates", "score"], descending=[True, True])
    )
