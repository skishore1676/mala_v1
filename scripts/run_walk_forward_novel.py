#!/usr/bin/env python3
"""
Walk-forward evaluation for novel directional strategies.

Approach:
- Split timeline into rolling train/test windows.
- For each strategy+ticker, choose best ratio on TRAIN by expectancy.
- Evaluate chosen ratio on next TEST window (out-of-sample).

Usage:
  python scripts/run_walk_forward_novel.py
  python scripts/run_walk_forward_novel.py --train-months 6 --test-months 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.config import settings
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.results_db import ResultsDB
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.regime_router import RegimeRouterStrategy
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy
from src.time_utils import et_date_expr


console = Console()


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for novel strategies")
    parser.add_argument("--tickers", nargs="+", default=settings.default_tickers)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--cost-r", type=float, default=0.05)
    parser.add_argument("--min-signals", type=int, default=20)
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filenames (for multi-run comparisons).",
    )
    return parser.parse_args()


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
    wins = mfe >= (ratio * mae)
    p = float(np.mean(wins)) if len(wins) else 0.0
    exp_r = p * ratio - (1.0 - p) - cost_r
    return p, exp_r


def evaluate_df(df_eval: pl.DataFrame, direction: str, ratio: float, cost_r: float) -> dict:
    base = df_eval.filter(pl.col("signal")).drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"])
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    if base.is_empty():
        return {"signals": 0, "confidence": None, "exp_r": None}

    mfe = base["forward_mfe_eod"].to_numpy()
    mae = base["forward_mae_eod"].to_numpy()

    p, exp_r = eval_ratio(mfe, mae, ratio, cost_r)
    return {
        "signals": len(mfe),
        "confidence": round(p, 4),
        "exp_r": round(exp_r, 4),
    }


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    windows = build_windows(args.start, args.end, args.train_months, args.test_months)

    if not windows:
        console.print("[red]No valid walk-forward windows for chosen date range/params.[/]")
        return

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()

    strategies = [
        ElasticBandReversionStrategy(
            z_score_threshold=2.0,
            z_score_window=240,
        ),
        KinematicLadderStrategy(
            regime_window=30,
            accel_window=10,
            volume_multiplier=1.05,
            volume_ma_period=settings.volume_ma_period,
            use_time_filter=True,
        ),
        CompressionBreakoutStrategy(
            compression_window=20,
            breakout_lookback=20,
            compression_factor=0.85,
            volume_ma_period=settings.volume_ma_period,
            volume_multiplier=1.15,
            use_time_filter=True,
        ),
        RegimeRouterStrategy(
            vol_short_window=20,
            vol_long_window=60,
            trend_vel_window=30,
            trend_vol_ratio=1.0,
            compression_vol_ratio=0.9,
            trend_velocity_floor=0.015,
        ),
        OpeningDriveClassifierStrategy(
            opening_window_minutes=25,
            entry_start_offset_minutes=25,
            entry_end_offset_minutes=120,
            min_drive_return_pct=0.0015,
            volume_multiplier=1.2,
        ),
        OpeningDriveClassifierStrategy(
            opening_window_minutes=25,
            entry_start_offset_minutes=25,
            entry_end_offset_minutes=120,
            min_drive_return_pct=0.0020,
            breakout_buffer_pct=0.0005,
            volume_multiplier=1.4,
            allow_long=False,
            allow_short=True,
            enable_continue=True,
            enable_fail=False,
            strategy_label="Opening Drive v2 (Short Continue)",
        ),
    ]

    rows: list[dict] = []

    console.rule("[bold cyan]Walk-Forward Evaluation[/]")
    console.print(
        f"Tickers: {args.tickers} | Range: {args.start} -> {args.end} | "
        f"Train/Test months: {args.train_months}/{args.test_months} | Ratios: {ratios} | cost_r={args.cost_r}"
    )

    for ticker in args.tickers:
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            continue
        df = physics.enrich(df)

        for strategy in strategies:
            df_sig = strategy.generate_signals(df.clone())
            df_eval_all = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

            for w_idx, w in enumerate(windows, start=1):
                train_df = df_eval_all.filter(
                    (et_date_expr("timestamp") >= w.train_start)
                    & (et_date_expr("timestamp") <= w.train_end)
                )
                test_df = df_eval_all.filter(
                    (et_date_expr("timestamp") >= w.test_start)
                    & (et_date_expr("timestamp") <= w.test_end)
                )

                for direction in ("combined", "long", "short"):
                    # Pick ratio by TRAIN expectancy
                    best_ratio = None
                    best_train_exp = -1e9
                    best_train_conf = None
                    best_train_n = 0

                    for ratio in ratios:
                        train_stats = evaluate_df(train_df, direction, ratio, args.cost_r)
                        n = int(train_stats["signals"])
                        if n < args.min_signals or train_stats["exp_r"] is None:
                            continue
                        exp_r = float(train_stats["exp_r"])
                        if exp_r > best_train_exp:
                            best_train_exp = exp_r
                            best_ratio = ratio
                            best_train_conf = train_stats["confidence"]
                            best_train_n = n

                    if best_ratio is None:
                        continue

                    test_stats = evaluate_df(test_df, direction, best_ratio, args.cost_r)
                    test_n = int(test_stats["signals"])
                    if test_n < args.min_signals or test_stats["exp_r"] is None:
                        continue

                    rows.append({
                        "ticker": ticker,
                        "strategy": strategy.name,
                        "direction": direction,
                        "window_idx": w_idx,
                        "train_start": w.train_start.isoformat(),
                        "train_end": w.train_end.isoformat(),
                        "test_start": w.test_start.isoformat(),
                        "test_end": w.test_end.isoformat(),
                        "selected_ratio": best_ratio,
                        "train_signals": best_train_n,
                        "train_confidence": best_train_conf,
                        "train_exp_r": round(best_train_exp, 4),
                        "test_signals": test_n,
                        "test_confidence": test_stats["confidence"],
                        "test_exp_r": test_stats["exp_r"],
                    })

    if not rows:
        console.print("[red]No walk-forward rows generated.[/]")
        return

    out_df = pl.DataFrame(rows)

    agg = (
        out_df.group_by(["ticker", "strategy", "direction"])
        .agg([
            pl.len().alias("oos_windows"),
            pl.col("test_signals").sum().alias("oos_signals"),
            pl.col("test_exp_r").mean().alias("avg_test_exp_r"),
            (pl.col("test_exp_r") > 0).mean().alias("pct_positive_oos_windows"),
            pl.col("test_confidence").mean().alias("avg_test_confidence"),
        ])
        .sort(["pct_positive_oos_windows", "avg_test_exp_r"], descending=[True, True])
    )

    table = Table(title="Walk-Forward OOS Summary", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("Win." , justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Avg OOS Exp(R)", justify="right")
    table.add_column("% OOS > 0", justify="right")

    for r in agg.iter_rows(named=True):
        table.add_row(
            str(r["ticker"]),
            str(r["strategy"]),
            str(r["direction"]),
            str(r["oos_windows"]),
            str(r["oos_signals"]),
            f"{float(r['avg_test_exp_r']):+.3f}",
            f"{float(r['pct_positive_oos_windows']):.1%}",
        )

    console.print(table)

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    safe_tag = args.tag.strip().replace(" ", "_")
    suffix = f"_{safe_tag}" if safe_tag else ""
    detail_path = out_dir / f"walk_forward_novel_detail_{stamp}{suffix}.csv"
    agg_path = out_dir / f"walk_forward_novel_summary_{stamp}{suffix}.csv"
    out_df.write_csv(detail_path)
    agg.write_csv(agg_path)

    console.print(f"\nSaved walk-forward detail -> [green]{detail_path}[/]")
    console.print(f"Saved walk-forward summary -> [green]{agg_path}[/]")

    db = ResultsDB()
    run_id = db.start_run(
        script="run_walk_forward_novel.py",
        params={
            "tickers": args.tickers,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "train_months": args.train_months,
            "test_months": args.test_months,
            "ratios": ratios,
            "cost_r": args.cost_r,
            "min_signals": args.min_signals,
            "tag": args.tag,
        },
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_walk_forward_novel.py",
        artifact_type="walk_forward_novel_detail",
        source_path=str(detail_path),
        df=out_df,
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_walk_forward_novel.py",
        artifact_type="walk_forward_novel_summary",
        source_path=str(agg_path),
        df=agg,
    )
    db.finish_run(run_id)
    console.print(f"Saved DB rows -> [green]{db.db_path}[/] (run_id={run_id})")


if __name__ == "__main__":
    main()
