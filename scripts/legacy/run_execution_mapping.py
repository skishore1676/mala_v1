#!/usr/bin/env python3
"""
M6 execution mapping for holdout-promoted candidates.

Builds practical option-structure mappings and execution-stressed expectancy
using Monte Carlo perturbations.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.oracle.results_db import ResultsDB
from src.research.stages import (
    latest_csv,
    run_execution_mapping_for_candidates,
    promoted_candidates_from_holdout,
)
from src.strategy.base import required_feature_union
from src.strategy.factory import build_strategy_by_name


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M6 execution mapping on holdout-promoted candidates.")
    parser.add_argument("--holdout-summary", default="", help="Path to holdout_validation_summary CSV.")
    parser.add_argument("--holdout-detail", default="", help="Path to holdout_validation_detail CSV.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--holdout-start", type=date.fromisoformat, default=date(2025, 12, 1))
    parser.add_argument("--holdout-end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--base-cost-r", type=float, default=0.08)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--out-dir", default="data/results")
    return parser.parse_args()

def print_summary(df: pl.DataFrame) -> None:
    table = Table(title="M6 Execution Mapping Summary", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("Ratio", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Base Exp(R)", justify="right")
    table.add_column("MC Exp p05/p50/p95", justify="right")
    table.add_column("P(Exp>0)", justify="right")
    table.add_column("Structure")
    for r in df.iter_rows(named=True):
        table.add_row(
            str(r["ticker"]),
            str(r["strategy"]),
            str(r["direction"]),
            f"{float(r['selected_ratio']):.2f}",
            str(int(r["holdout_trades"])),
            f"{float(r['base_exp_r']):+.3f}",
            f"{float(r['mc_exp_r_p05']):+.3f}/{float(r['mc_exp_r_p50']):+.3f}/{float(r['mc_exp_r_p95']):+.3f}",
            f"{float(r['mc_prob_positive_exp']):.1%}",
            str(r["structure"]),
        )
    console.print(table)


def write_md(path: Path, df: pl.DataFrame, source_summary: Path, source_detail: Path) -> None:
    lines: list[str] = []
    lines.append(f"# M6 Execution Mapping ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("## Source Artifacts")
    lines.append(f"- Holdout summary: `{source_summary}`")
    lines.append(f"- Holdout detail: `{source_detail}`")
    lines.append("")
    lines.append("## Candidate Mapping")
    lines.append("| ticker | strategy | direction | selected_ratio | holdout_trades | base_exp_r | mc_exp_r_p05 | mc_exp_r_p50 | mc_exp_r_p95 | mc_prob_positive_exp | structure | dte | delta_plan | entry_window_et | profit_take | risk_rule |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|")
    for row in df.iter_rows(named=True):
        lines.append(
            f"| {row['ticker']} | {row['strategy']} | {row['direction']} | "
            f"{float(row['selected_ratio']):.2f} | {int(row['holdout_trades'])} | "
            f"{float(row['base_exp_r']):.4f} | {float(row['mc_exp_r_p05']):.4f} | "
            f"{float(row['mc_exp_r_p50']):.4f} | {float(row['mc_exp_r_p95']):.4f} | "
            f"{float(row['mc_prob_positive_exp']):.4f} | {row['structure']} | {row['dte']} | "
            f"{row['delta_plan']} | {row['entry_window_et']} | {row['profit_take']} | {row['risk_rule']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_summary_path = Path(args.holdout_summary) if args.holdout_summary else latest_csv(out_dir, "holdout_validation_summary")
    holdout_detail_path = Path(args.holdout_detail) if args.holdout_detail else latest_csv(out_dir, "holdout_validation_detail")

    holdout_summary = pl.read_csv(holdout_summary_path)
    holdout_detail = pl.read_csv(holdout_detail_path)
    promoted = promoted_candidates_from_holdout(holdout_summary)
    if promoted.is_empty():
        console.print("[red]No candidates promoted to execution mapping in holdout summary.[/]")
        return

    console.rule("[bold cyan]M6 Execution Mapping[/]")
    console.print(
        f"Holdout source: {holdout_summary_path.name} | Detail source: {holdout_detail_path.name}\n"
        f"Window: {args.holdout_start} -> {args.holdout_end} | Base cost_r={args.base_cost_r} | "
        f"Monte Carlo iters={args.bootstrap_iters}"
    )

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()
    stress_cfg = ExecutionStressConfig(bootstrap_iters=args.bootstrap_iters)
    candidate_strategies = [
        build_strategy_by_name(name) for name in promoted["strategy"].unique().to_list()
    ]
    needed_features = required_feature_union(candidate_strategies)

    ticker_frames: dict[str, pl.DataFrame] = {}
    for ticker in promoted["ticker"].unique().to_list():
        raw = storage.load_bars(ticker, args.start, args.holdout_end)
        if raw.is_empty():
            continue
        ticker_frames[ticker] = physics.enrich_for_features(raw, needed_features)

    rows = run_execution_mapping_for_candidates(
        promoted=promoted,
        holdout_detail=holdout_detail,
        ticker_frames=ticker_frames,
        metrics=metrics,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        base_cost_r=args.base_cost_r,
        stress_cfg=stress_cfg,
    )

    if not rows:
        console.print("[red]No execution mapping rows generated.[/]")
        return

    result_df = pl.DataFrame(rows).sort("mc_exp_r_p50", descending=True)
    print_summary(result_df)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = out_dir / f"execution_mapping_summary_{stamp}.csv"
    md_path = out_dir / f"execution_mapping_summary_{stamp}.md"
    result_df.write_csv(summary_path)
    write_md(md_path, result_df, holdout_summary_path, holdout_detail_path)

    db = ResultsDB()
    run_id = db.start_run(
        script="run_execution_mapping.py",
        params={
            "holdout_summary": str(holdout_summary_path),
            "holdout_detail": str(holdout_detail_path),
            "holdout_start": args.holdout_start.isoformat(),
            "holdout_end": args.holdout_end.isoformat(),
            "base_cost_r": args.base_cost_r,
            "bootstrap_iters": args.bootstrap_iters,
        },
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_execution_mapping.py",
        artifact_type="execution_mapping_summary",
        source_path=str(summary_path),
        df=result_df,
    )
    db.finish_run(run_id)

    console.print(f"\nSaved execution mapping CSV -> [green]{summary_path}[/]")
    console.print(f"Saved execution mapping MD -> [green]{md_path}[/]")
    console.print(f"Saved DB rows -> [green]{db.db_path}[/] (run_id={run_id})")


if __name__ == "__main__":
    main()
