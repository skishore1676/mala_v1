"""Helpers for preserving candidate identity and params across research stages."""

from __future__ import annotations

from typing import Any

import polars as pl

from src.strategy.base import BaseStrategy
from src.strategy.factory import build_strategy, build_strategy_by_name


_BASE_IDENTITY_COLUMNS = ("ticker", "strategy", "direction")

_NON_PARAM_COLUMNS = {
    "catalog_strategy",
    "base_strategy",
    "window_idx",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "selected_ratio",
    "evaluation_window",
    "train_signals",
    "train_confidence",
    "train_exp_r",
    "test_signals",
    "test_confidence",
    "test_exp_r",
    "effective_cost_r",
    "oos_windows",
    "oos_signals",
    "avg_test_exp_r",
    "avg_test_mfe_mae_ratio",
    "pct_positive_oos_windows",
    "avg_test_confidence",
    "observed_cost_points",
    "min_oos_windows",
    "min_oos_signals",
    "min_avg_test_exp_r",
    "mean_avg_test_exp_r",
    "min_pct_positive_oos_windows",
    "mean_pct_positive_oos_windows",
    "mean_test_confidence",
    "has_all_cost_points",
    "passes_window_gate",
    "passes_signal_gate",
    "passes_stability_gate",
    "passes_exp_gate",
    "passes_all_gates",
    "decision",
    "score",
    "cost_r",
    "cost_bps",
    "calib_signals",
    "calib_exp_r",
    "holdout_signals",
    "holdout_confidence",
    "holdout_exp_r",
    "passes_cost_gate",
    "min_holdout_signals",
    "min_holdout_exp_r",
    "mean_holdout_exp_r",
    "passes_all_cost_gates",
    "passes_holdout",
    "holdout_trades",
    "holdout_win_rate",
    "base_exp_r",
    "trades",
    "mc_prob_positive_exp",
    "mc_exp_r_mean",
    "mc_exp_r_p05",
    "mc_exp_r_p50",
    "mc_exp_r_p95",
    "mc_total_r_p05",
    "mc_total_r_p50",
    "mc_total_r_p95",
    "mc_max_dd_p50",
    "mc_max_drawdown_p50",
    "structure",
    "dte",
    "delta_plan",
    "entry_window_et",
    "profit_take",
    "risk_rule",
    "stress_profile",
    "signal_quality",
    "total_oos_signals",
    "avg_oos_exp_r",
    "med_oos_exp_r",
    "pct_positive_windows",
    "avg_confidence",
    "avg_effective_cost_r",
    "median_selected_ratio",
    "m1_score",
    "discovery_score",
    "execution_profile",
    "plateau_neighbor_count",
    "plateau_positive_neighbors",
    "plateau_positive_ratio",
    "plateau_mean_neighbor_exp_r",
    "plateau_min_neighbor_exp_r",
    "plateau_score",
}


def candidate_identity_columns(
    df: pl.DataFrame,
    *,
    extra_exclude: set[str] | None = None,
) -> list[str]:
    exclude = set(_NON_PARAM_COLUMNS)
    if extra_exclude:
        exclude.update(extra_exclude)

    identity = [column for column in _BASE_IDENTITY_COLUMNS if column in df.columns]
    extras = [
        column
        for column in df.columns
        if column not in exclude and column not in identity
    ]
    return [*identity, *extras]


def candidate_params(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in _NON_PARAM_COLUMNS
        and key not in _BASE_IDENTITY_COLUMNS
        and value is not None
    }


def build_candidate_strategy(candidate: dict[str, Any]) -> BaseStrategy:
    params = candidate_params(candidate)
    catalog_strategy = candidate.get("catalog_strategy")
    if catalog_strategy:
        return build_strategy(str(catalog_strategy), params or None)
    if params:
        try:
            return build_strategy(str(candidate["strategy"]), params or None)
        except ValueError:
            pass
    return build_strategy_by_name(str(candidate["strategy"]))
