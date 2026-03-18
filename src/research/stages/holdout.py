"""Reusable holdout-validation stage logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.oracle.policies import RewardRiskWinCondition
from src.research.stages.walk_forward import cost_r_from_bps
from src.strategy.factory import build_strategy_by_name
from src.time_utils import et_date_expr


def parse_floats(csv_like: str) -> list[float]:
    values = [float(x.strip()) for x in csv_like.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Could not parse numeric values from: {csv_like}")
    return values


def latest_csv(out_dir: Path, prefix: str, exclude_substrings: tuple[str, ...] = ()) -> Path:
    candidates = sorted(out_dir.glob(f"{prefix}_*.csv"))
    filtered = [
        candidate for candidate in candidates
        if not any(substr in candidate.name for substr in exclude_substrings)
    ]
    if not filtered:
        raise FileNotFoundError(f"No files found for {prefix}_*.csv in {out_dir}")
    return filtered[-1]


def eval_direction(df_eval: pl.DataFrame, direction: str, ratio: float, cost_bps: float) -> dict[str, float | int | None]:
    base = df_eval.filter(pl.col("signal")).drop_nulls(
        subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"]
    )
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    if base.is_empty():
        return {"signals": 0, "confidence": None, "exp_r": None}

    mfe = base["forward_mfe_eod"].to_numpy()
    mae = base["forward_mae_eod"].to_numpy()
    policy = RewardRiskWinCondition(ratio=ratio)
    p = policy.confidence(mfe, mae)

    avg_price = float(base["close"].mean())
    avg_mae_dollars = float(np.mean(mae))
    cost_r = cost_r_from_bps(cost_bps, avg_mae_dollars, avg_price)

    exp_r = policy.expectancy(mfe, mae, cost_r)
    return {"signals": int(len(mfe)), "confidence": round(p, 4), "exp_r": round(exp_r, 4)}


def choose_ratio(
    *,
    calib_df: pl.DataFrame,
    direction: str,
    ratios: list[float],
    cost_bps: float,
    min_calib_signals: int,
) -> tuple[float | None, dict[str, float | int | None]]:
    best_ratio = None
    best_exp = -1e9
    best_stats: dict[str, float | int | None] = {"signals": 0, "confidence": None, "exp_r": None}
    for ratio in ratios:
        stats = eval_direction(calib_df, direction, ratio, cost_bps)
        if stats["exp_r"] is None or int(stats["signals"]) < min_calib_signals:
            continue
        exp_r = float(stats["exp_r"])
        if exp_r > best_exp:
            best_exp = exp_r
            best_ratio = ratio
            best_stats = stats
    return best_ratio, best_stats


def promoted_candidates_from_gate_report(gate_df: pl.DataFrame) -> pl.DataFrame:
    return gate_df.filter(pl.col("decision") == "promote_to_holdout").select(["ticker", "strategy", "direction"])


def run_holdout_validation_for_candidates(
    *,
    promoted: pl.DataFrame,
    ticker_frames: dict[str, pl.DataFrame],
    metrics: MetricsCalculator,
    start_date,
    calibration_end,
    holdout_start,
    holdout_end,
    ratios: list[float],
    costs: list[float],
    min_calibration_signals: int,
    min_holdout_signals: int,
) -> list[dict[str, object]]:
    detail_rows: list[dict[str, object]] = []

    for candidate in promoted.iter_rows(named=True):
        ticker = candidate["ticker"]
        strategy_name = candidate["strategy"]
        direction = candidate["direction"]
        if ticker not in ticker_frames:
            continue

        strategy = build_strategy_by_name(strategy_name)
        df_sig = strategy.generate_signals(ticker_frames[ticker].clone())
        df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

        calib_df = df_eval.filter(
            (et_date_expr("timestamp") >= start_date)
            & (et_date_expr("timestamp") <= calibration_end)
        )
        holdout_df = df_eval.filter(
            (et_date_expr("timestamp") >= holdout_start)
            & (et_date_expr("timestamp") <= holdout_end)
        )

        for cost_bps in costs:
            selected_ratio, calib_stats = choose_ratio(
                calib_df=calib_df,
                direction=direction,
                ratios=ratios,
                cost_bps=cost_bps,
                min_calib_signals=min_calibration_signals,
            )
            if selected_ratio is None:
                detail_rows.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy_name,
                        "direction": direction,
                        "cost_bps": cost_bps,
                        "selected_ratio": None,
                        "calib_signals": 0,
                        "calib_exp_r": None,
                        "holdout_signals": 0,
                        "holdout_confidence": None,
                        "holdout_exp_r": None,
                        "passes_cost_gate": False,
                    }
                )
                continue

            holdout_stats = eval_direction(holdout_df, direction, selected_ratio, cost_bps)
            holdout_signals = int(holdout_stats["signals"])
            holdout_exp = holdout_stats["exp_r"]
            passes_cost = (
                holdout_exp is not None
                and holdout_signals >= min_holdout_signals
                and float(holdout_exp) >= 0.0
            )
            detail_rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "direction": direction,
                    "cost_bps": cost_bps,
                    "selected_ratio": selected_ratio,
                    "calib_signals": int(calib_stats["signals"]),
                    "calib_exp_r": calib_stats["exp_r"],
                    "holdout_signals": holdout_signals,
                    "holdout_confidence": holdout_stats["confidence"],
                    "holdout_exp_r": holdout_exp,
                    "passes_cost_gate": passes_cost,
                }
            )

    return detail_rows


def summarize_holdout(detail_df: pl.DataFrame, cost_count: int) -> pl.DataFrame:
    return (
        detail_df.group_by(["ticker", "strategy", "direction"])
        .agg([
            pl.len().alias("observed_cost_points"),
            pl.col("holdout_signals").min().alias("min_holdout_signals"),
            pl.col("holdout_exp_r").min().alias("min_holdout_exp_r"),
            pl.col("holdout_exp_r").mean().alias("mean_holdout_exp_r"),
            pl.col("passes_cost_gate").all().alias("passes_all_cost_gates"),
        ])
        .with_columns([
            (
                (pl.col("observed_cost_points") == cost_count)
                & pl.col("passes_all_cost_gates")
            ).alias("passes_holdout")
        ])
        .with_columns([
            pl.when(pl.col("passes_holdout"))
            .then(pl.lit("promote_to_execution_mapping"))
            .otherwise(pl.lit("fail_holdout_or_need_rework"))
            .alias("decision")
        ])
        .sort(["passes_holdout", "min_holdout_exp_r"], descending=[True, True])
    )
