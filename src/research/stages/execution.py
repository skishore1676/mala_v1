"""Reusable execution-mapping stage logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig, stress_from_win_flags, stress_profile_library
from src.oracle.policies import RewardRiskWinCondition
from src.research.stages.candidates import build_candidate_strategy, candidate_identity_columns
from src.research.stages.directional import (
    canonical_directional_snapshot_windows,
    resolve_directional_metric_columns,
    resolve_evaluation_window,
)
from src.time_utils import et_date_expr


def option_mapping_for(strategy: str, direction: str, profile_name: str = "default") -> dict[str, str]:
    if profile_name == "stock_like":
        return {
            "structure": "underlying",
            "dte": "n/a",
            "delta_plan": "n/a",
            "entry_window_et": "signal bar",
            "profit_take": "rule-based exit",
            "risk_rule": "underlying stop / target",
        }
    if profile_name == "single_option":
        if direction == "long":
            return {
                "structure": "long_call",
                "dte": "7-21",
                "delta_plan": "0.35-0.55",
                "entry_window_et": "09:45-14:30",
                "profit_take": "50-90% premium",
                "risk_rule": "hard stop at -35% premium",
            }
        return {
            "structure": "long_put",
            "dte": "7-21",
            "delta_plan": "0.35-0.55",
            "entry_window_et": "09:45-14:30",
            "profit_take": "50-90% premium",
            "risk_rule": "hard stop at -35% premium",
        }
    if strategy == "Elastic Band Reversion" and direction == "short" and profile_name == "debit_spread_tight":
        return {
            "structure": "put_debit_spread",
            "dte": "7-14",
            "delta_plan": "long 0.40-0.50 / short 0.20-0.30",
            "entry_window_et": "09:45-15:00",
            "profit_take": "65-85% spread value",
            "risk_rule": "hard stop at -45% premium",
        }
    if strategy == "Elastic Band Reversion" and direction == "short":
        return {
            "structure": "put_debit_spread",
            "dte": "7-14",
            "delta_plan": "long 0.35-0.45 / short 0.15-0.25",
            "entry_window_et": "09:45-15:00",
            "profit_take": "60-80% spread value",
            "risk_rule": "hard stop at -50% premium",
        }
    if strategy == "Compression Expansion Breakout" and direction == "short":
        return {
            "structure": "put_debit_spread",
            "dte": "7-21",
            "delta_plan": "long 0.30-0.40 / short 0.12-0.22",
            "entry_window_et": "09:40-14:30",
            "profit_take": "55-75% spread value",
            "risk_rule": "hard stop at -45% premium",
        }
    if direction == "long":
        return {
            "structure": "call_debit_spread",
            "dte": "7-21",
            "delta_plan": "long 0.30-0.45 / short 0.10-0.25",
            "entry_window_et": "09:45-14:30",
            "profit_take": "50-70% spread value",
            "risk_rule": "hard stop at -45% premium",
        }
    return {
        "structure": "put_debit_spread",
        "dte": "7-21",
        "delta_plan": "long 0.30-0.45 / short 0.10-0.25",
        "entry_window_et": "09:45-14:30",
        "profit_take": "50-70% spread value",
        "risk_rule": "hard stop at -45% premium",
    }


def execution_profiles_for(strategy: str, direction: str) -> list[dict[str, str]]:
    if direction == "combined":
        return []
    return [
        {"execution_profile": "debit_spread_default", "stress_profile": "default"},
        {"execution_profile": "debit_spread_tight", "stress_profile": "debit_spread_tight"},
        {"execution_profile": "single_option", "stress_profile": "single_option"},
        {"execution_profile": "stock_like", "stress_profile": "stock_like"},
    ]


def promoted_candidates_from_holdout(holdout_summary: pl.DataFrame) -> pl.DataFrame:
    return holdout_summary.filter(pl.col("decision") == "promote_to_execution_mapping").select(
        candidate_identity_columns(holdout_summary)
    )


def median_selected_ratio(
    holdout_detail: pl.DataFrame,
    *,
    ticker: str,
    strategy: str,
    direction: str,
) -> float | None:
    ratio_candidates = (
        holdout_detail
        .filter(
            (pl.col("ticker") == ticker)
            & (pl.col("strategy") == strategy)
            & (pl.col("direction") == direction)
            & pl.col("selected_ratio").is_not_null()
        )
        .get_column("selected_ratio")
        .to_list()
    )
    if not ratio_candidates:
        return None
    return float(np.median(np.array(ratio_candidates, dtype=np.float64)))


def run_execution_mapping_for_candidates(
    *,
    promoted: pl.DataFrame,
    holdout_detail: pl.DataFrame,
    ticker_frames: dict[str, pl.DataFrame],
    metrics: MetricsCalculator,
    holdout_start,
    holdout_end,
    base_cost_r: float,
    stress_cfg: ExecutionStressConfig,
    evaluation_window: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stress_profiles = stress_profile_library(
        bootstrap_iters=stress_cfg.bootstrap_iters,
        random_seed=stress_cfg.random_seed,
    )
    resolved_window = resolve_evaluation_window(metrics, evaluation_window)
    snapshot_windows = canonical_directional_snapshot_windows(
        metrics=metrics,
        evaluation_window=resolved_window,
    )

    for candidate in promoted.iter_rows(named=True):
        ticker = candidate["ticker"]
        strategy_name = candidate["strategy"]
        direction = candidate["direction"]
        candidate_context = {
            key: value for key, value in candidate.items() if key not in {"ticker", "strategy", "direction"}
        }

        selected_ratio = median_selected_ratio(
            holdout_detail,
            ticker=ticker,
            strategy=strategy_name,
            direction=direction,
        )
        if selected_ratio is None or ticker not in ticker_frames:
            continue

        strategy = build_candidate_strategy(candidate)
        df_sig = strategy.generate_signals(ticker_frames[ticker].clone())
        df_eval = metrics.add_directional_forward_metrics(
            df_sig,
            snapshot_windows=snapshot_windows,
        )
        mfe_col, mae_col, metric_window = resolve_directional_metric_columns(
            df_eval,
            evaluation_window=resolved_window,
            metrics=metrics,
            allow_eod_fallback=False,
        )

        base = df_eval.filter(
            (et_date_expr("timestamp") >= holdout_start)
            & (et_date_expr("timestamp") <= holdout_end)
            & pl.col("signal")
            & pl.col(mfe_col).is_not_null()
            & pl.col(mae_col).is_not_null()
        )
        if direction != "combined":
            base = base.filter(pl.col("signal_direction") == direction)
        if base.is_empty():
            continue

        mfe = base[mfe_col].to_numpy()
        mae = base[mae_col].to_numpy()
        policy = RewardRiskWinCondition(ratio=selected_ratio)
        wins = policy.flags(mfe, mae)
        p = policy.confidence(mfe, mae)
        base_exp_r = policy.expectancy(mfe, mae, base_cost_r)

        for profile in execution_profiles_for(strategy_name, direction):
            stress_name = profile["stress_profile"]
            stress = stress_from_win_flags(
                win_flags=wins,
                ratio=selected_ratio,
                config=stress_profiles[stress_name],
            )
            mapping = option_mapping_for(strategy_name, direction, profile["execution_profile"])

            rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "direction": direction,
                    **candidate_context,
                    "execution_profile": profile["execution_profile"],
                    "stress_profile": stress_name,
                    "selected_ratio": selected_ratio,
                    "evaluation_window": metric_window,
                    "holdout_trades": int(len(wins)),
                    "holdout_win_rate": round(p, 4),
                    "base_exp_r": round(base_exp_r, 4),
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in stress.items()},
                    **mapping,
                }
            )

    return rows
