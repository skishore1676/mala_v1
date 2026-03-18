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

import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.results_db import ResultsDB
from src.research.stages import (
    latest_csv,
    parse_floats,
    promoted_candidates_from_gate_report,
    run_holdout_validation_for_candidates,
    summarize_holdout,
)


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
            f"{float(row['min_holdout_exp_r']):+.3f}" if row["min_holdout_exp_r"] is not None else "NaN",
            str(int(row["min_holdout_signals"])),
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    ratios = parse_floats(args.ratios)
    costs = parse_floats(args.cost_grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_report_path = Path(args.gate_report) if args.gate_report else latest_csv(
        out_dir,
        "convergence_gate_report",
        exclude_substrings=("relaxed",),
    )
    gate_df = pl.read_csv(gate_report_path)
    promoted = promoted_candidates_from_gate_report(gate_df)

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

    detail_rows = run_holdout_validation_for_candidates(
        promoted=promoted,
        ticker_frames=ticker_frames,
        metrics=metrics,
        start_date=args.start,
        calibration_end=args.calibration_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        ratios=ratios,
        costs=costs,
        min_calibration_signals=args.min_calibration_signals,
        min_holdout_signals=args.min_holdout_signals,
    )

    if not detail_rows:
        console.print("[red]No holdout rows generated.[/]")
        return

    detail_df = pl.DataFrame(detail_rows)
    summary_df = summarize_holdout(detail_df, cost_count=len(costs))

    print_summary_table(summary_df)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detail_path = out_dir / f"holdout_validation_detail_{stamp}.csv"
    summary_path = out_dir / f"holdout_validation_summary_{stamp}.csv"
    detail_df.write_csv(detail_path)
    summary_df.write_csv(summary_path)

    console.print(f"\nSaved holdout detail -> [green]{detail_path}[/]")
    console.print(f"Saved holdout summary -> [green]{summary_path}[/]")

    db = ResultsDB()
    run_id = db.start_run(
        script="run_holdout_validation.py",
        params={
            "gate_report": str(gate_report_path),
            "start": args.start.isoformat(),
            "calibration_end": args.calibration_end.isoformat(),
            "holdout_start": args.holdout_start.isoformat(),
            "holdout_end": args.holdout_end.isoformat(),
            "ratios": ratios,
            "costs": costs,
            "min_calibration_signals": args.min_calibration_signals,
            "min_holdout_signals": args.min_holdout_signals,
        },
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_holdout_validation.py",
        artifact_type="holdout_validation_detail",
        source_path=str(detail_path),
        df=detail_df,
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_holdout_validation.py",
        artifact_type="holdout_validation_summary",
        source_path=str(summary_path),
        df=summary_df,
    )
    db.finish_run(run_id)
    console.print(f"Saved DB rows -> [green]{db.db_path}[/] (run_id={run_id})")


if __name__ == "__main__":
    main()
