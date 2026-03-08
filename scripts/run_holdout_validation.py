#!/usr/bin/env python3
"""
M5 Holdout validator for promoted candidates.

Flow:
1. Load promoted candidates from convergence gate report.
2. Refit reward:risk ratio on calibration period only.
3. Evaluate holdout period only across friction assumptions.
4. Output pass/fail promotion summary.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.regime_router import RegimeRouterStrategy
from src.config import settings
from src.time_utils import et_date_expr


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run holdout validation for promoted candidates")
    parser.add_argument("--gate-report", default="", help="Path to convergence_gate_report CSV. Defaults to latest.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--calibration-end", type=date.fromisoformat, default=date(2025, 11, 30))
    parser.add_argument("--holdout-start", type=date.fromisoformat, default=date(2025, 12, 1))
    parser.add_argument("--holdout-end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--cost-grid", default="0.05,0.08,0.12")
    parser.add_argument("--min-calibration-signals", type=int, default=200)
    parser.add_argument("--min-holdout-signals", type=int, default=500)
    parser.add_argument("--out-dir", default="data/results")
    return parser.parse_args()


def parse_floats(csv_like: str) -> list[float]:
    values = [float(x.strip()) for x in csv_like.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Could not parse numeric values from: {csv_like}")
    return values


def latest_gate_report(out_dir: Path) -> Path:
    candidates = sorted(out_dir.glob("convergence_gate_report_*.csv"))
    if not candidates:
        raise FileNotFoundError("No convergence_gate_report_*.csv found in data/results")
    return candidates[-1]


def build_strategy(strategy_name: str):
    if strategy_name == "Elastic Band Reversion":
        return ElasticBandReversionStrategy(
            z_score_threshold=2.0,
            z_score_window=240,
        )
    if strategy_name == "Kinematic Ladder":
        return KinematicLadderStrategy(
            regime_window=30,
            accel_window=10,
            volume_multiplier=1.05,
            volume_ma_period=settings.volume_ma_period,
            use_time_filter=True,
        )
    if strategy_name == "Compression Expansion Breakout":
        return CompressionBreakoutStrategy(
            compression_window=20,
            breakout_lookback=20,
            compression_factor=0.85,
            volume_ma_period=settings.volume_ma_period,
            volume_multiplier=1.15,
            use_time_filter=True,
        )
    if strategy_name == "Regime Router (Kinematic + Compression)":
        return RegimeRouterStrategy(
            vol_short_window=20,
            vol_long_window=60,
            trend_vel_window=30,
            trend_vol_ratio=1.0,
            compression_vol_ratio=0.9,
            trend_velocity_floor=0.015,
        )
    raise ValueError(f"Unsupported strategy in holdout validator: {strategy_name}")


def eval_direction(df_eval: pl.DataFrame, direction: str, ratio: float, cost_r: float) -> dict:
    base = df_eval.filter(pl.col("signal")).drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"])
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    if base.is_empty():
        return {"signals": 0, "confidence": None, "exp_r": None}

    mfe = base["forward_mfe_eod"].to_numpy()
    mae = base["forward_mae_eod"].to_numpy()
    wins = mfe >= (ratio * mae)
    p = float(np.mean(wins))
    exp_r = p * ratio - (1.0 - p) - cost_r
    return {"signals": int(len(mfe)), "confidence": round(p, 4), "exp_r": round(exp_r, 4)}


def choose_ratio(
    calib_df: pl.DataFrame,
    direction: str,
    ratios: list[float],
    cost_r: float,
    min_calib_signals: int,
) -> tuple[float | None, dict]:
    best_ratio = None
    best_exp = -1e9
    best_stats: dict = {"signals": 0, "confidence": None, "exp_r": None}
    for ratio in ratios:
        stats = eval_direction(calib_df, direction, ratio, cost_r)
        if stats["exp_r"] is None or int(stats["signals"]) < min_calib_signals:
            continue
        exp_r = float(stats["exp_r"])
        if exp_r > best_exp:
            best_exp = exp_r
            best_ratio = ratio
            best_stats = stats
    return best_ratio, best_stats


def print_summary_table(df: pl.DataFrame) -> None:
    table = Table(title="M5 Holdout Summary", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("Decision")
    table.add_column("Costs", justify="right")
    table.add_column("Min Holdout Exp(R)", justify="right")
    table.add_column("Min Holdout N", justify="right")

    for row in df.iter_rows(named=True):
        table.add_row(
            str(row["ticker"]),
            str(row["strategy"]),
            str(row["direction"]),
            str(row["decision"]),
            str(int(row["observed_cost_points"])),
            f"{float(row['min_holdout_exp_r']):+.3f}",
            str(int(row["min_holdout_signals"])),
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    ratios = parse_floats(args.ratios)
    costs = parse_floats(args.cost_grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_report_path = Path(args.gate_report) if args.gate_report else latest_gate_report(out_dir)
    gate_df = pl.read_csv(gate_report_path)
    promoted = gate_df.filter(pl.col("decision") == "promote_to_holdout").select(["ticker", "strategy", "direction"])

    if promoted.is_empty():
        console.print("[red]No promoted candidates found in gate report.[/]")
        return

    console.rule("[bold cyan]M5 Holdout Validation[/]")
    console.print(
        f"Gate report: {gate_report_path}\n"
        f"Calibration: {args.start} -> {args.calibration_end} | "
        f"Holdout: {args.holdout_start} -> {args.holdout_end}\n"
        f"Ratios={ratios} | Costs={costs} | "
        f"Min calib signals={args.min_calibration_signals} | Min holdout signals={args.min_holdout_signals}"
    )

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()

    # cache enriched data by ticker
    ticker_frames: dict[str, pl.DataFrame] = {}
    for ticker in promoted["ticker"].unique().to_list():
        raw = storage.load_bars(ticker, args.start, args.holdout_end)
        if raw.is_empty():
            continue
        ticker_frames[ticker] = physics.enrich(raw)

    detail_rows: list[dict] = []

    for c in promoted.iter_rows(named=True):
        ticker = c["ticker"]
        strategy_name = c["strategy"]
        direction = c["direction"]
        if ticker not in ticker_frames:
            continue

        strategy = build_strategy(strategy_name)
        df_sig = strategy.generate_signals(ticker_frames[ticker].clone())
        df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

        calib_df = df_eval.filter(
            (et_date_expr("timestamp") >= args.start)
            & (et_date_expr("timestamp") <= args.calibration_end)
        )
        holdout_df = df_eval.filter(
            (et_date_expr("timestamp") >= args.holdout_start)
            & (et_date_expr("timestamp") <= args.holdout_end)
        )

        for cost_r in costs:
            selected_ratio, calib_stats = choose_ratio(
                calib_df=calib_df,
                direction=direction,
                ratios=ratios,
                cost_r=cost_r,
                min_calib_signals=args.min_calibration_signals,
            )
            if selected_ratio is None:
                detail_rows.append({
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "direction": direction,
                    "cost_r": cost_r,
                    "selected_ratio": None,
                    "calib_signals": 0,
                    "calib_exp_r": None,
                    "holdout_signals": 0,
                    "holdout_confidence": None,
                    "holdout_exp_r": None,
                    "passes_cost_gate": False,
                })
                continue

            holdout_stats = eval_direction(holdout_df, direction, selected_ratio, cost_r)
            holdout_signals = int(holdout_stats["signals"])
            holdout_exp = holdout_stats["exp_r"]
            passes_cost = (
                holdout_exp is not None
                and holdout_signals >= args.min_holdout_signals
                and float(holdout_exp) >= 0.0
            )
            detail_rows.append({
                "ticker": ticker,
                "strategy": strategy_name,
                "direction": direction,
                "cost_r": cost_r,
                "selected_ratio": selected_ratio,
                "calib_signals": int(calib_stats["signals"]),
                "calib_exp_r": calib_stats["exp_r"],
                "holdout_signals": holdout_signals,
                "holdout_confidence": holdout_stats["confidence"],
                "holdout_exp_r": holdout_exp,
                "passes_cost_gate": passes_cost,
            })

    if not detail_rows:
        console.print("[red]No holdout rows generated.[/]")
        return

    detail_df = pl.DataFrame(detail_rows)
    cost_count = len(costs)
    summary_df = (
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

    print_summary_table(summary_df)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detail_path = out_dir / f"holdout_validation_detail_{stamp}.csv"
    summary_path = out_dir / f"holdout_validation_summary_{stamp}.csv"
    detail_df.write_csv(detail_path)
    summary_df.write_csv(summary_path)

    console.print(f"\nSaved holdout detail -> [green]{detail_path}[/]")
    console.print(f"Saved holdout summary -> [green]{summary_path}[/]")


if __name__ == "__main__":
    main()
