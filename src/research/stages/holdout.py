"""Reusable holdout-validation stage logic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.research.stages.candidates import build_candidate_strategy, candidate_identity_columns
from src.research.stages.directional import (
    canonical_directional_snapshot_windows,
    resolve_directional_metric_columns,
    resolve_evaluation_window,
)
from src.research.stages.walk_forward import evaluate_df
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


def eval_direction(
    df_eval: pl.DataFrame,
    direction: str,
    ratio: float,
    cost_bps: float,
    evaluation_window: int | None = None,
) -> dict[str, float | int | None]:
    return evaluate_df(
        df_eval,
        direction,
        ratio,
        cost_bps=cost_bps,
        evaluation_window=evaluation_window,
    )


def choose_ratio(
    *,
    calib_df: pl.DataFrame,
    direction: str,
    ratios: list[float],
    cost_bps: float,
    min_calib_signals: int,
    evaluation_window: int | None = None,
) -> tuple[float | None, dict[str, float | int | None]]:
    best_ratio = None
    best_exp = -1e9
    best_stats: dict[str, float | int | None] = {"signals": 0, "confidence": None, "exp_r": None}
    for ratio in ratios:
        stats = eval_direction(
            calib_df,
            direction,
            ratio,
            cost_bps,
            evaluation_window=evaluation_window,
        )
        if stats["exp_r"] is None or int(stats["signals"]) < min_calib_signals:
            continue
        exp_r = float(stats["exp_r"])
        if exp_r > best_exp:
            best_exp = exp_r
            best_ratio = ratio
            best_stats = stats
    return best_ratio, best_stats


def promoted_candidates_from_gate_report(gate_df: pl.DataFrame) -> pl.DataFrame:
    return gate_df.filter(pl.col("decision") == "promote_to_holdout").select(
        candidate_identity_columns(gate_df)
    )


def extract_signal_trade_rows(
    df_eval: pl.DataFrame,
    *,
    direction: str,
    evaluation_window: int | None,
) -> list[dict[str, object]]:
    """Return one dict per realized signal row in ``df_eval``.

    Used by ``run_holdout_validation_for_candidates`` when
    ``retain_trade_rows=True`` so that downstream per-regime slicing
    (``src.research.catalog_regime_performance``) has the raw trade
    shape to bucket by market regime. The shape is deliberately
    minimal — just trade_date, signal_direction, mfe, mae — because
    that is all the slicer needs. Callers that want richer context
    (entry price, hold time, etc.) should take a second pass.

    Signal rows are already filtered to rows where ``signal == True``
    and where ``mfe/mae/signal_direction`` are non-null, matching the
    filter ``evaluate_df`` applies internally.
    """
    mfe_col, mae_col, _ = resolve_directional_metric_columns(
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
        return []

    # Extract only the columns we need. Keep polars' native date
    # conversion so callers get datetime.date objects ready for the
    # market_regime classifier.
    trade_date_expr = et_date_expr("timestamp").alias("trade_date")
    slim = base.select(
        [
            trade_date_expr,
            pl.col("signal_direction"),
            pl.col(mfe_col).alias("mfe"),
            pl.col(mae_col).alias("mae"),
        ]
    )
    return slim.to_dicts()


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
    evaluation_window: int | None = None,
    trade_rows_sink: list[dict[str, object]] | None = None,
    catalog_key_fn: Callable[[dict[str, object]], str] | None = None,
) -> list[dict[str, object]]:
    """Run holdout validation over promoted candidates.

    Optional hooks for per-regime performance slicing (W2.1 in the
    trade_lab architecture) without changing the existing return
    shape:

      * ``trade_rows_sink``: if provided, realized signal rows from the
        out-of-sample ``holdout_df`` are appended here. Each row is a
        dict ready to feed into
        ``src.research.catalog_regime_performance.TradeRow``.
      * ``catalog_key_fn``: called with each candidate row dict to
        derive the ``catalog_key`` attached to the trade rows. If
        ``None`` (the default), a placeholder key ``"{ticker}::{strategy}::{direction}"``
        is used — fine for unit tests and standalone callers but not
        for the real nightly pipeline, which should pass in a
        callable that builds the same catalog_key Mala's playbook
        catalog uses (``playbooks._build_catalog_key``).

    Both hooks are no-ops when ``trade_rows_sink`` is None, so
    existing callers are unaffected.
    """
    detail_rows: list[dict[str, object]] = []
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
        if ticker not in ticker_frames:
            continue

        strategy = build_candidate_strategy(candidate)
        df_sig = strategy.generate_signals(ticker_frames[ticker].clone())
        df_eval = metrics.add_directional_forward_metrics(
            df_sig,
            snapshot_windows=snapshot_windows,
        )

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
                evaluation_window=resolved_window,
            )
            if selected_ratio is None:
                detail_rows.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy_name,
                        "direction": direction,
                        **candidate_context,
                        "cost_bps": cost_bps,
                        "evaluation_window": resolved_window,
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

            holdout_stats = eval_direction(
                holdout_df,
                direction,
                selected_ratio,
                cost_bps,
                evaluation_window=resolved_window,
            )
            holdout_signals = int(holdout_stats["signals"])
            holdout_exp = holdout_stats["exp_r"]
            passes_cost = (
                holdout_exp is not None
                and holdout_signals >= min_holdout_signals
                and float(holdout_exp) >= 0.0
            )

            if trade_rows_sink is not None:
                # Emit one row per realized out-of-sample signal for
                # per-regime slicing downstream. We only do this once
                # per (candidate, direction) because per-cost iteration
                # shares the same signal set.
                if cost_bps == costs[0]:
                    if catalog_key_fn is not None:
                        key = catalog_key_fn(candidate)
                    else:
                        key = f"{ticker}::{strategy_name}::{direction}"
                    raw_rows = extract_signal_trade_rows(
                        holdout_df,
                        direction=direction,
                        evaluation_window=resolved_window,
                    )
                    for raw in raw_rows:
                        trade_rows_sink.append(
                            {
                                "catalog_key": key,
                                "ticker": ticker,
                                "strategy": strategy_name,
                                "direction": direction,
                                "trade_date": raw["trade_date"],
                                "mfe": float(raw["mfe"]),
                                "mae": float(raw["mae"]),
                            }
                        )
            detail_rows.append(
                {
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "direction": direction,
                    **candidate_context,
                    "cost_bps": cost_bps,
                    "evaluation_window": resolved_window,
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
    group_cols = candidate_identity_columns(detail_df)
    return (
        detail_df.group_by(group_cols)
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
