"""Reusable walk-forward research stage logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from src.oracle.metrics import MetricsCalculator
from src.oracle.policies import RewardRiskWinCondition
from src.research.stages.directional import (
    canonical_directional_snapshot_windows,
    resolve_directional_metric_columns,
    resolve_evaluation_window,
)
from src.time_utils import et_date_expr


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def build_windows(start: date, end: date, train_months: int, test_months: int) -> list[Window]:
    windows: list[Window] = []
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + relativedelta(months=train_months) - relativedelta(days=1)
        test_start = train_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)
        if test_end > end:
            break
        windows.append(Window(train_start, train_end, test_start, test_end))
        cursor = cursor + relativedelta(months=test_months)
    return windows


def eval_ratio(mfe: np.ndarray, mae: np.ndarray, ratio: float, cost_r: float) -> tuple[float, float]:
    policy = RewardRiskWinCondition(ratio=ratio)
    p = policy.confidence(mfe, mae)
    exp_r = policy.expectancy(mfe, mae, cost_r)
    return p, exp_r


def cost_r_from_bps(cost_bps: float, avg_mae_dollars: float, avg_entry_price: float) -> float:
    """Convert basis-point transaction cost to R units: cost_dollars / avg_mae."""
    if avg_mae_dollars is None or avg_mae_dollars <= 0 or avg_entry_price <= 0:
        return 0.05
    return (avg_entry_price * cost_bps / 10_000.0) / avg_mae_dollars


def evaluate_df(
    df_eval: pl.DataFrame,
    direction: str,
    ratio: float,
    cost_r: float | None = None,
    cost_bps: float | None = None,
    evaluation_window: int | None = None,
) -> dict[str, float | int | None]:
    mfe_col, mae_col, resolved_window = resolve_directional_metric_columns(
        df_eval,
        evaluation_window=evaluation_window,
        allow_eod_fallback=evaluation_window is None,
    )
    base = df_eval.filter(pl.col("signal")).drop_nulls(
        subset=[mfe_col, mae_col, "signal_direction"]
    )
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    if base.is_empty():
        return {
            "signals": 0,
            "confidence": None,
            "exp_r": None,
            "avg_mfe_mae_ratio": None,
            "effective_cost_r": None,
            "evaluation_window": resolved_window,
        }

    mfe = base[mfe_col].to_numpy()
    mae = base[mae_col].to_numpy()
    valid_ratio_mask = np.isfinite(mfe) & np.isfinite(mae) & (mae > 0)
    avg_mfe_mae_ratio = (
        round(float(np.mean(mfe[valid_ratio_mask] / mae[valid_ratio_mask])), 4)
        if np.any(valid_ratio_mask)
        else None
    )

    if cost_bps is not None:
        avg_mae_d = float(np.mean(mae))
        avg_entry = float(base["close"].mean()) if "close" in base.columns else 0.0
        effective_cost_r = cost_r_from_bps(cost_bps, avg_mae_d, avg_entry)
    else:
        effective_cost_r = cost_r or 0.05

    p, exp_r = eval_ratio(mfe, mae, ratio, effective_cost_r)
    return {
        "signals": len(mfe),
        "confidence": round(p, 4),
        "exp_r": round(exp_r, 4),
        "avg_mfe_mae_ratio": avg_mfe_mae_ratio,
        "effective_cost_r": round(effective_cost_r, 5),
        "evaluation_window": resolved_window,
    }


def run_walk_forward_for_strategies(
    *,
    ticker: str,
    df: pl.DataFrame,
    strategies: Iterable,
    windows: list[Window],
    ratios: list[float],
    metrics: MetricsCalculator,
    min_signals: int,
    cost_r: float | None = None,
    cost_bps: float | None = None,
    evaluation_window: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    resolved_window = resolve_evaluation_window(metrics, evaluation_window)
    snapshot_windows = canonical_directional_snapshot_windows(
        metrics=metrics,
        evaluation_window=resolved_window,
    )

    for strategy in strategies:
        df_sig = strategy.generate_signals(df.clone())
        df_eval_all = metrics.add_directional_forward_metrics(
            df_sig,
            snapshot_windows=snapshot_windows,
        )

        for w_idx, window in enumerate(windows, start=1):
            train_df = df_eval_all.filter(
                (et_date_expr("timestamp") >= window.train_start)
                & (et_date_expr("timestamp") <= window.train_end)
            )
            test_df = df_eval_all.filter(
                (et_date_expr("timestamp") >= window.test_start)
                & (et_date_expr("timestamp") <= window.test_end)
            )

            for direction in ("combined", "long", "short"):
                best_ratio = None
                best_train_exp = -1e9
                best_train_conf = None
                best_train_n = 0

                for ratio in ratios:
                    train_stats = evaluate_df(
                        train_df,
                        direction,
                        ratio,
                        cost_r=cost_r,
                        cost_bps=cost_bps,
                        evaluation_window=resolved_window,
                    )
                    n = int(train_stats["signals"])
                    if n < min_signals or train_stats["exp_r"] is None:
                        continue
                    exp_r = float(train_stats["exp_r"])
                    if exp_r > best_train_exp:
                        best_train_exp = exp_r
                        best_ratio = ratio
                        best_train_conf = train_stats["confidence"]
                        best_train_n = n

                if best_ratio is None:
                    continue

                test_stats = evaluate_df(
                    test_df,
                    direction,
                    best_ratio,
                    cost_r=cost_r,
                    cost_bps=cost_bps,
                    evaluation_window=resolved_window,
                )
                test_n = int(test_stats["signals"])
                if test_n < min_signals or test_stats["exp_r"] is None:
                    continue

                rows.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy.name,
                        "direction": direction,
                        "window_idx": w_idx,
                        "train_start": window.train_start.isoformat(),
                        "train_end": window.train_end.isoformat(),
                        "test_start": window.test_start.isoformat(),
                        "test_end": window.test_end.isoformat(),
                        "selected_ratio": best_ratio,
                        "evaluation_window": resolved_window,
                        "train_signals": best_train_n,
                        "train_confidence": best_train_conf,
                        "train_exp_r": round(best_train_exp, 4),
                        "test_signals": test_n,
                        "test_confidence": test_stats["confidence"],
                        "test_exp_r": test_stats["exp_r"],
                        "test_avg_mfe_mae_ratio": test_stats.get("avg_mfe_mae_ratio"),
                        "effective_cost_r": test_stats.get("effective_cost_r"),
                    }
                )

    return rows


def aggregate_walk_forward(rows: list[dict[str, object]]) -> pl.DataFrame:
    out_df = pl.DataFrame(rows)
    agg_exprs = [
        pl.len().alias("oos_windows"),
        pl.col("test_signals").sum().alias("oos_signals"),
        pl.col("test_exp_r").drop_nans().mean().alias("avg_test_exp_r"),
        (pl.col("test_exp_r").drop_nans() > 0).mean().alias("pct_positive_oos_windows"),
        pl.col("test_confidence").drop_nans().mean().alias("avg_test_confidence"),
    ]
    if "test_avg_mfe_mae_ratio" in out_df.columns:
        agg_exprs.append(
            pl.col("test_avg_mfe_mae_ratio").drop_nans().mean().alias("avg_test_mfe_mae_ratio")
        )
    return (
        out_df.group_by(["ticker", "strategy", "direction"])
        .agg(agg_exprs)
        .sort(["pct_positive_oos_windows", "avg_test_exp_r"], descending=[True, True])
    )
