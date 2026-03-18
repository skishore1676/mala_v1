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
from datetime import date
from pathlib import Path
import sys

import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.config import settings
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.results_db import ResultsDB
from src.research.registry import ResearchRegistry
from src.research.stages import aggregate_walk_forward, build_windows, run_walk_forward_for_strategies
from src.strategy.base import required_feature_union
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy
from src.strategy.jerk_pivot_momentum import JerkPivotMomentumStrategy


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for novel strategies")
    parser.add_argument("--tickers", nargs="+", default=settings.default_tickers)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--cost-r", type=float, default=None,
                        help="Fixed cost in R units (legacy). Superseded by --cost-bps.")
    parser.add_argument("--cost-bps", type=float, default=8.0,
                        help="Transaction cost in basis points of entry price (relative, ticker-aware). Default 8 bps.")
    parser.add_argument("--min-signals", type=int, default=20)
    parser.add_argument(
        "--strategy-source",
        default="legacy",
        choices=["legacy", "tracked", "validation"],
        help="Select strategies from the legacy hardcoded list, tracked registry, or validation set.",
    )
    parser.add_argument(
        "--strategy-names",
        nargs="+",
        default=None,
        help="Optional subset of strategy display names to run for the selected source.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filenames (for multi-run comparisons).",
    )
    return parser.parse_args()


def _legacy_strategies() -> list:
    """Current production list preserved during phased migration."""
    return [
        ElasticBandReversionStrategy(z_score_threshold=1.25, z_score_window=360),
        ElasticBandReversionStrategy(z_score_threshold=1.0,  z_score_window=240),
        ElasticBandReversionStrategy(z_score_threshold=1.75, z_score_window=120),
        ElasticBandReversionStrategy(z_score_threshold=2.0,  z_score_window=240),
        ElasticBandReversionStrategy(z_score_threshold=2.5,  z_score_window=240),
        ElasticBandReversionStrategy(z_score_threshold=2.5,  z_score_window=360),
        ElasticBandReversionStrategy(z_score_threshold=3.0,  z_score_window=240),
        KinematicLadderStrategy(regime_window=20, accel_window=8,  use_volume_filter=False),
        KinematicLadderStrategy(regime_window=30, accel_window=8,  use_volume_filter=False),
        KinematicLadderStrategy(regime_window=20, accel_window=12, use_volume_filter=False),
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
        JerkPivotMomentumStrategy(
            vpoc_proximity_pct=0.003,
            jerk_lookback=10,
            volume_multiplier=1.1,
            use_volume_filter=True,
            strategy_label="Jerk-Pivot Momentum (tight)",
        ),
    ]


def _select_strategies(strategy_source: str, strategy_names: list[str] | None) -> list:
    if strategy_source == "legacy":
        strategies = _legacy_strategies()
    else:
        registry = ResearchRegistry()
        if strategy_source == "tracked":
            strategies = registry.build_tracked_strategies()
        else:
            strategies = registry.build_validation_strategies()

    if strategy_names:
        wanted = set(strategy_names)
        strategies = [strategy for strategy in strategies if strategy.name in wanted]
    return strategies


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    windows = build_windows(args.start, args.end, args.train_months, args.test_months)

    # Determine friction mode: relative bps wins if --cost-r not explicitly passed
    use_relative = args.cost_r is None
    cost_bps_val = args.cost_bps if use_relative else None
    fixed_cost_r = args.cost_r if not use_relative else None

    if not windows:
        console.print("[red]No valid walk-forward windows for chosen date range/params.[/]")
        return

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()
    strategies = _select_strategies(args.strategy_source, args.strategy_names)

    if not strategies:
        console.print("[red]No strategies selected for the chosen source/filter.[/]")
        return

    rows: list[dict] = []
    needed_features = required_feature_union(strategies)

    console.rule("[bold cyan]Walk-Forward Evaluation[/]")
    friction_desc = (
        f"relative {cost_bps_val} bps" if use_relative else f"fixed cost_r={fixed_cost_r}"
    )
    console.print(
        f"Tickers: {args.tickers} | Range: {args.start} -> {args.end} | "
        f"Train/Test months: {args.train_months}/{args.test_months} | Ratios: {ratios} | "
        f"Friction: {friction_desc} | Strategy source: {args.strategy_source} "
        f"({len(strategies)} strategies)"
    )

    for ticker in args.tickers:
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            continue
        df = physics.enrich_for_features(df, needed_features)
        rows.extend(
            run_walk_forward_for_strategies(
                ticker=ticker,
                df=df,
                strategies=strategies,
                windows=windows,
                ratios=ratios,
                metrics=metrics,
                min_signals=args.min_signals,
                cost_r=fixed_cost_r,
                cost_bps=cost_bps_val,
            )
        )

    if not rows:
        console.print("[red]No walk-forward rows generated.[/]")
        return

    out_df = pl.DataFrame(rows)
    agg = aggregate_walk_forward(rows)

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
            f"{float(r['avg_test_exp_r']):+.3f}" if r["avg_test_exp_r"] is not None else "NaN",
            f"{float(r['pct_positive_oos_windows']):.1%}" if r["pct_positive_oos_windows"] is not None else "NaN",
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
            "cost_r": fixed_cost_r,
            "cost_bps": cost_bps_val,
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
