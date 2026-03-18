"""Reusable execution-mapping stage logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig, stress_from_win_flags
from src.strategy.factory import build_strategy_by_name
from src.time_utils import et_date_expr


def option_mapping_for(strategy: str, direction: str) -> dict[str, str]:
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


def promoted_candidates_from_holdout(holdout_summary: pl.DataFrame) -> pl.DataFrame:
    return holdout_summary.filter(pl.col("decision") == "promote_to_execution_mapping")


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for candidate in promoted.iter_rows(named=True):
        ticker = candidate["ticker"]
        strategy_name = candidate["strategy"]
        direction = candidate["direction"]

        selected_ratio = median_selected_ratio(
            holdout_detail,
            ticker=ticker,
            strategy=strategy_name,
            direction=direction,
        )
        if selected_ratio is None or ticker not in ticker_frames:
            continue

        strategy = build_strategy_by_name(strategy_name)
        df_sig = strategy.generate_signals(ticker_frames[ticker].clone())
        df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

        base = df_eval.filter(
            (et_date_expr("timestamp") >= holdout_start)
            & (et_date_expr("timestamp") <= holdout_end)
            & pl.col("signal")
            & pl.col("forward_mfe_eod").is_not_null()
            & pl.col("forward_mae_eod").is_not_null()
        )
        if direction != "combined":
            base = base.filter(pl.col("signal_direction") == direction)
        if base.is_empty():
            continue

        mfe = base["forward_mfe_eod"].to_numpy()
        mae = base["forward_mae_eod"].to_numpy()
        wins = mfe >= (selected_ratio * mae)
        p = float(np.mean(wins))
        base_exp_r = p * selected_ratio - (1.0 - p) - base_cost_r

        stress = stress_from_win_flags(
            win_flags=wins,
            ratio=selected_ratio,
            config=stress_cfg,
        )
        mapping = option_mapping_for(strategy_name, direction)

        rows.append(
            {
                "ticker": ticker,
                "strategy": strategy_name,
                "direction": direction,
                "selected_ratio": selected_ratio,
                "holdout_trades": int(len(wins)),
                "holdout_win_rate": round(p, 4),
                "base_exp_r": round(base_exp_r, 4),
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in stress.items()},
                **mapping,
            }
        )

    return rows
