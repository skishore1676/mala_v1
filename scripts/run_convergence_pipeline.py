#!/usr/bin/env python3
"""
Strategy convergence pipeline with measurable gates.

What it does:
1. Runs walk-forward evaluation across a friction grid (cost_r values).
2. Aggregates candidate performance across all cost points.
3. Applies promotion gates and produces a ranked shortlist.

Usage:
  ./.venv/bin/python scripts/run_convergence_pipeline.py
  ./.venv/bin/python scripts/run_convergence_pipeline.py --cost-grid 0.05,0.08,0.12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "META", "AMD", "PLTR", "AAPL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategy convergence gates over walk-forward outputs.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--walkforward-min-signals", type=int, default=20)
    parser.add_argument("--cost-grid", default="0.05,0.08,0.12")

    # Promotion gates.
    parser.add_argument("--gate-min-oos-windows", type=int, default=6)
    parser.add_argument("--gate-min-oos-signals", type=int, default=3000)
    parser.add_argument("--gate-min-pct-positive", type=float, default=0.67)
    parser.add_argument("--gate-min-exp-r", type=float, default=0.0)

    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--out-dir", default="data/results")
    return parser.parse_args()


def parse_costs(cost_grid: str) -> list[float]:
    costs = [float(x.strip()) for x in cost_grid.split(",") if x.strip()]
    if not costs:
        raise ValueError("No valid values parsed from --cost-grid")
    return costs


def cost_tag(cost_r: float) -> str:
    # 0.05 -> "cost050", 0.12 -> "cost120"
    return f"cost{int(round(cost_r * 1000)):03d}"


def run_walkforward(args: argparse.Namespace, cost_r: float) -> Path:
    tag = cost_tag(cost_r)
    cmd = [
        args.python_bin,
        "scripts/run_walk_forward_novel.py",
        "--tickers",
        *args.tickers,
        "--start",
        args.start.isoformat(),
        "--end",
        args.end.isoformat(),
        "--train-months",
        str(args.train_months),
        "--test-months",
        str(args.test_months),
        "--ratios",
        args.ratios,
        "--cost-r",
        str(cost_r),
        "--min-signals",
        str(args.walkforward_min_signals),
        "--tag",
        tag,
    ]
    console.print(f"[cyan]Running walk-forward for cost_r={cost_r:.3f} ({tag})[/]")
    subprocess.run(cmd, check=True)

    out_dir = Path(args.out_dir)
    stamp = date.today().isoformat()
    summary_path = out_dir / f"walk_forward_novel_summary_{stamp}_{tag}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary not found: {summary_path}")
    return summary_path


def build_gate_report(
    combined: pl.DataFrame,
    cost_count: int,
    args: argparse.Namespace,
) -> pl.DataFrame:
    report = (
        combined.group_by(["ticker", "strategy", "direction"])
        .agg([
            pl.len().alias("observed_cost_points"),
            pl.col("oos_windows").min().alias("min_oos_windows"),
            pl.col("oos_signals").min().alias("min_oos_signals"),
            pl.col("avg_test_exp_r").min().alias("min_avg_test_exp_r"),
            pl.col("avg_test_exp_r").mean().alias("mean_avg_test_exp_r"),
            pl.col("pct_positive_oos_windows").min().alias("min_pct_positive_oos_windows"),
            pl.col("pct_positive_oos_windows").mean().alias("mean_pct_positive_oos_windows"),
            pl.col("avg_test_confidence").mean().alias("mean_test_confidence"),
        ])
        .with_columns([
            (pl.col("observed_cost_points") == cost_count).alias("has_all_cost_points"),
            (pl.col("min_oos_windows") >= args.gate_min_oos_windows).alias("passes_window_gate"),
            (pl.col("min_oos_signals") >= args.gate_min_oos_signals).alias("passes_signal_gate"),
            (pl.col("min_pct_positive_oos_windows") >= args.gate_min_pct_positive).alias("passes_stability_gate"),
            (pl.col("min_avg_test_exp_r") >= args.gate_min_exp_r).alias("passes_exp_gate"),
        ])
        .with_columns([
            (
                pl.col("has_all_cost_points")
                & pl.col("passes_window_gate")
                & pl.col("passes_signal_gate")
                & pl.col("passes_stability_gate")
                & pl.col("passes_exp_gate")
            ).alias("passes_all_gates")
        ])
        .with_columns([
            pl.when(pl.col("passes_all_gates"))
            .then(pl.lit("promote_to_holdout"))
            .when(pl.col("has_all_cost_points") & (pl.col("min_avg_test_exp_r") > 0))
            .then(pl.lit("candidate_needs_more_stability"))
            .otherwise(pl.lit("reject_or_rework"))
            .alias("decision"),
            (
                pl.col("min_avg_test_exp_r") * 1000
                + pl.col("min_pct_positive_oos_windows") * 100
                + pl.col("min_oos_signals") / 1000
            ).alias("score"),
        ])
        .sort(["passes_all_gates", "score"], descending=[True, True])
    )
    return report


def print_shortlist(report: pl.DataFrame, top_n: int) -> None:
    shortlist = report.filter(pl.col("passes_all_gates")).head(top_n)
    if shortlist.is_empty():
        shortlist = report.head(top_n)

    table = Table(title="Convergence Shortlist", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("Decision")
    table.add_column("Min Exp(R)", justify="right")
    table.add_column("Min %OOS>0", justify="right")
    table.add_column("Min Signals", justify="right")
    table.add_column("Score", justify="right")

    for row in shortlist.iter_rows(named=True):
        table.add_row(
            str(row["ticker"]),
            str(row["strategy"]),
            str(row["direction"]),
            str(row["decision"]),
            f"{float(row['min_avg_test_exp_r']):+.3f}",
            f"{float(row['min_pct_positive_oos_windows']):.1%}",
            str(int(row["min_oos_signals"])),
            f"{float(row['score']):.2f}",
        )
    console.print(table)


def write_shortlist_md(
    path: Path,
    report: pl.DataFrame,
    costs: list[float],
    args: argparse.Namespace,
) -> None:
    passed = report.filter(pl.col("passes_all_gates"))
    preview = passed if not passed.is_empty() else report.head(args.top_n)

    lines: list[str] = []
    lines.append(f"# Convergence Shortlist ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("## Run Setup")
    lines.append(f"- Tickers: `{', '.join(args.tickers)}`")
    lines.append(f"- Range: `{args.start}` to `{args.end}`")
    lines.append(f"- Train/Test months: `{args.train_months}/{args.test_months}`")
    lines.append(f"- Ratios: `{args.ratios}`")
    lines.append(f"- Cost grid: `{', '.join(f'{c:.3f}' for c in costs)}`")
    lines.append("")
    lines.append("## Promotion Gates")
    lines.append(f"- `oos_windows >= {args.gate_min_oos_windows}`")
    lines.append(f"- `oos_signals >= {args.gate_min_oos_signals}`")
    lines.append(f"- `pct_positive_oos_windows >= {args.gate_min_pct_positive:.2f}`")
    lines.append(f"- `min_avg_test_exp_r >= {args.gate_min_exp_r:.3f}` across all costs")
    lines.append("")
    lines.append(f"- Candidates passing all gates: **{passed.height}**")
    lines.append("")
    lines.append("## Ranked Candidates")
    lines.append("| ticker | strategy | direction | decision | min_avg_test_exp_r | min_pct_positive_oos_windows | min_oos_signals | score |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for row in preview.iter_rows(named=True):
        lines.append(
            f"| {row['ticker']} | {row['strategy']} | {row['direction']} | {row['decision']} | "
            f"{float(row['min_avg_test_exp_r']):.4f} | {float(row['min_pct_positive_oos_windows']):.4f} | "
            f"{int(row['min_oos_signals'])} | {float(row['score']):.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `promote_to_holdout`: ready for untouched holdout validation.")
    lines.append("- `candidate_needs_more_stability`: positive but failed one or more stability gates.")
    lines.append("- `reject_or_rework`: fails robustness requirements.")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    costs = parse_costs(args.cost_grid)

    console.rule("[bold green]Strategy Convergence Pipeline[/]")
    console.print(
        f"Tickers={args.tickers} | Range={args.start}->{args.end} | "
        f"Cost grid={costs} | Gates: windows>={args.gate_min_oos_windows}, "
        f"signals>={args.gate_min_oos_signals}, pct>={args.gate_min_pct_positive}, exp>={args.gate_min_exp_r}"
    )

    summary_frames: list[pl.DataFrame] = []
    for cost_r in costs:
        summary_path = run_walkforward(args, cost_r)
        df = pl.read_csv(summary_path).with_columns(pl.lit(cost_r).alias("cost_r"))
        summary_frames.append(df)

    combined = pl.concat(summary_frames)
    report = build_gate_report(combined, cost_count=len(costs), args=args)

    print_shortlist(report, args.top_n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    combined_path = out_dir / f"convergence_cost_summary_{ts}.csv"
    report_path = out_dir / f"convergence_gate_report_{ts}.csv"
    shortlist_path = out_dir / f"convergence_shortlist_{ts}.md"

    combined.write_csv(combined_path)
    report.write_csv(report_path)
    write_shortlist_md(shortlist_path, report, costs, args)

    console.print(f"\nSaved cost summary -> [green]{combined_path}[/]")
    console.print(f"Saved gate report -> [green]{report_path}[/]")
    console.print(f"Saved shortlist -> [green]{shortlist_path}[/]")


if __name__ == "__main__":
    main()
