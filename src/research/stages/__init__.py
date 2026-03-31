"""Reusable research stage logic."""

from src.research.stages.convergence import build_gate_report, cost_tag, parse_costs
from src.research.stages.execution import (
    execution_profiles_for,
    median_selected_ratio,
    option_mapping_for,
    promoted_candidates_from_holdout,
    run_execution_mapping_for_candidates,
)
from src.research.stages.holdout import (
    choose_ratio,
    eval_direction as eval_holdout_direction,
    latest_csv,
    parse_floats,
    promoted_candidates_from_gate_report,
    summarize_holdout,
    run_holdout_validation_for_candidates,
)
from src.research.stages.walk_forward import (
    Window,
    aggregate_walk_forward,
    build_windows,
    cost_r_from_bps,
    evaluate_df,
    run_walk_forward_for_strategies,
)

__all__ = [
    "Window",
    "aggregate_walk_forward",
    "build_gate_report",
    "build_windows",
    "cost_r_from_bps",
    "cost_tag",
    "choose_ratio",
    "evaluate_df",
    "eval_holdout_direction",
    "execution_profiles_for",
    "latest_csv",
    "median_selected_ratio",
    "option_mapping_for",
    "parse_costs",
    "parse_floats",
    "promoted_candidates_from_gate_report",
    "promoted_candidates_from_holdout",
    "run_execution_mapping_for_candidates",
    "run_holdout_validation_for_candidates",
    "run_walk_forward_for_strategies",
    "summarize_holdout",
]
