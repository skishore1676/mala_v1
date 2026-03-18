#!/usr/bin/env python3
"""
Elastic Band Reversion — Z-Score Parameter Grid Search

Sweeps z_score_threshold × z_score_window in walk-forward OOS mode to find
the efficient frontier between signal count and OOS expectancy.

Friction modes:
  --cost-r FLOAT     : Fixed cost in reward:risk units (legacy, uniform across tickers)
  --cost-bps FLOAT   : Cost in basis points of entry price (relative, ticker-aware)
                       Converted to R units per-window per-direction using avg MAE.

Usage:
  uv run python scripts/legacy/run_elastic_grid.py --tickers SPY AAPL META IWM QQQ TSLA
  uv run python scripts/legacy/run_elastic_grid.py --cost-bps 8 --tickers AAPL META

Output:
  - data/results/elastic_grid_detail_<stamp>.csv  (all window×direction rows)
  - data/results/elastic_grid_heatmap_<stamp>.csv (aggregated per param cell)
  - data/results/elastic_grid_report_<stamp>.md   (human-readable frontier)
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

from src.oracle.results_db import ResultsDB
from src.research import ResearchOrchestrator, ResearchStage


console = Console()

# ── Parameter grid ────────────────────────────────────────────────────────────

Z_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
Z_WINDOWS    = [60, 120, 180, 240, 360]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elastic Band z-score grid search")
    p.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "META", "IWM", "QQQ", "TSLA"])
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    p.add_argument("--end",   type=date.fromisoformat, default=date(2026, 2, 28))
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months",  type=int, default=3)
    p.add_argument("--ratios",  default="1.0,1.25,1.5,2.0",
                   help="Reward:risk ratios tried on train to pick best.")
    p.add_argument("--cost-r",   type=float, default=None,
                   help="Fixed cost in R units (legacy). Mutually exclusive with --cost-bps.")
    p.add_argument("--cost-bps", type=float, default=8.0,
                   help="Cost in basis points of entry price (relative, per-ticker). Default 8 bps.")
    p.add_argument("--min-signals", type=int, default=20,
                   help="Min signals per OOS window to count the window.")
    p.add_argument("--min-oos-signals-total", type=int, default=100,
                   help="Min total OOS signals across all windows for a cell to be considered valid.")
    p.add_argument("--out-dir", default="data/results")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    use_relative = args.cost_r is None
    cost_bps = args.cost_bps if use_relative else None
    fixed_cost_r = args.cost_r if not use_relative else None

    console.rule("[bold green]Elastic Band — Z-Score Grid Search[/]")
    console.print(
        f"Tickers: {args.tickers}\n"
        f"Range: {args.start} → {args.end} | "
        f"Train/Test: {args.train_months}m/{args.test_months}m | "
        f"{'Relative friction: ' + str(cost_bps) + ' bps' if use_relative else 'Fixed cost_r: ' + str(fixed_cost_r)}\n"
        f"Grid: {len(Z_THRESHOLDS)} thresholds × {len(Z_WINDOWS)} windows = "
        f"{len(Z_THRESHOLDS) * len(Z_WINDOWS)} param combos per ticker"
    )

    orchestrator = ResearchOrchestrator()
    sweep = orchestrator.run_action(
        ResearchStage.M1_DISCOVERY,
        "parameter_sweep",
        strategy_name="Elastic Band Reversion",
        parameter_space={
            "z_score_threshold": Z_THRESHOLDS,
            "z_score_window": Z_WINDOWS,
        },
        max_configs=len(Z_THRESHOLDS) * len(Z_WINDOWS),
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        ratios=ratios,
        min_signals=args.min_signals,
        cost_r=fixed_cost_r,
        cost_bps=cost_bps,
        min_total_signals=args.min_oos_signals_total,
    )
    detail_df = sweep.artifacts["detail"]
    heatmap = sweep.artifacts["aggregate"]

    if detail_df.is_empty() or heatmap.is_empty():
        console.print("[red]No rows generated. Check that tickers have cached data.[/]")
        return

    # ── Print heatmap table (combined direction, best configs per ticker) ────
    console.rule("[bold green]Grid Search Results — Combined Direction[/]")
    for ticker in args.tickers:
        sub = (
            heatmap
            .filter((pl.col("ticker") == ticker) & (pl.col("direction") == "combined"))
            .sort("avg_oos_exp_r", descending=True)
        )
        if sub.is_empty():
            continue

        table = Table(
            title=f"{ticker} | Elastic Band | Combined — OOS Expectancy Grid",
            show_lines=True,
        )
        table.add_column("Z-Thresh", justify="right")
        table.add_column("Z-Window", justify="right")
        table.add_column("OOS Wins", justify="right")
        table.add_column("Signals",  justify="right")
        table.add_column("Avg E[R]", justify="right")
        table.add_column("% E>0",   justify="right")
        table.add_column("Avg Conf", justify="right")
        table.add_column("Eff Cost", justify="right")
        table.add_column("Quality",  justify="right")

        for row in sub.iter_rows(named=True):
            exp_str = f"{float(row['avg_oos_exp_r']):+.3f}"
            table.add_row(
                str(row["z_score_threshold"]),
                str(row["z_score_window"]),
                str(row["oos_windows"]),
                str(int(row["total_oos_signals"])),
                exp_str,
                f"{float(row['pct_positive_windows']):.0%}",
                f"{float(row['avg_confidence']):.2%}",
                f"{float(row['avg_effective_cost_r']):.4f}",
                str(row["signal_quality"]),
            )
        console.print(table)

    # ── Write artifacts ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    detail_path  = out_dir / f"elastic_grid_detail_{stamp}.csv"
    heatmap_path = out_dir / f"elastic_grid_heatmap_{stamp}.csv"
    report_path  = out_dir / f"elastic_grid_report_{stamp}.md"

    detail_df.write_csv(detail_path)
    heatmap.write_csv(heatmap_path)
    _write_report(report_path, heatmap, args, stamp)

    console.print(f"\nSaved detail   → [green]{detail_path}[/]")
    console.print(f"Saved heatmap  → [green]{heatmap_path}[/]")
    console.print(f"Saved report   → [green]{report_path}[/]")

    # ── Store in results DB ──────────────────────────────────────────────────
    db = ResultsDB()
    run_id = db.start_run(
        script="run_elastic_grid.py",
        params={
            "tickers": args.tickers,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "train_months": args.train_months,
            "test_months": args.test_months,
            "ratios": ratios,
            "cost_bps": cost_bps,
            "fixed_cost_r": fixed_cost_r,
            "z_thresholds": Z_THRESHOLDS,
            "z_windows": Z_WINDOWS,
        },
    )
    db.ingest_dataframe(
        run_id=run_id,
        script="run_elastic_grid.py",
        artifact_type="elastic_grid_heatmap",
        source_path=str(heatmap_path),
        df=heatmap,
    )
    db.finish_run(run_id)
    console.print(f"Saved DB rows  → [green]{db.db_path}[/] (run_id={run_id})")


def _write_report(path: Path, heatmap: pl.DataFrame, args, stamp: str) -> None:
    lines = [
        f"# Elastic Band Z-Score Grid Report",
        f"",
        f"**Generated:** {stamp}  ",
        f"**Tickers:** {', '.join(args.tickers)}  ",
        f"**Range:** {args.start} → {args.end}  ",
        f"**Walk-forward:** {args.train_months}m train / {args.test_months}m test  ",
        f"**Friction:** {'relative ' + str(args.cost_bps) + ' bps' if args.cost_r is None else 'fixed cost_r=' + str(args.cost_r)}  ",
        f"**Min OOS signals (validity):** {args.min_oos_signals_total}  ",
        f"",
        f"---",
        f"",
        f"## Efficient Frontier — Top 10 Valid Configs per Ticker (Combined Direction)",
        f"",
        f"| Ticker | Z-Thresh | Z-Window | OOS Windows | OOS Signals | Avg E[R] | % E>0 | Avg Conf | Eff Cost |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    top = (
        heatmap
        .filter(
            (pl.col("direction") == "combined")
            & (pl.col("signal_quality") == "valid")
        )
        .sort("avg_oos_exp_r", descending=True)
    )

    for row in top.head(10).iter_rows(named=True):
        lines.append(
            f"| {row['ticker']} | {row['z_score_threshold']} | {row['z_score_window']} "
            f"| {row['oos_windows']} | {int(row['total_oos_signals'])} "
            f"| {float(row['avg_oos_exp_r']):+.4f} "
            f"| {float(row['pct_positive_windows']):.0%} "
            f"| {float(row['avg_confidence']):.2%} "
            f"| {float(row['avg_effective_cost_r']):.4f} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Low-N Configs (below {args.min_oos_signals_total} total OOS signals)",
        f"",
        f"> These cells are excluded from the frontier — sample too small to trust.",
        f"",
    ]

    low_n = (
        heatmap
        .filter(
            (pl.col("direction") == "combined")
            & (pl.col("signal_quality") == "low_n")
        )
        .sort("total_oos_signals", descending=True)
    )

    if not low_n.is_empty():
        lines.append(f"| Ticker | Z-Thresh | Z-Window | OOS Signals |")
        lines.append(f"|---|---:|---:|---:|")
        for row in low_n.iter_rows(named=True):
            lines.append(
                f"| {row['ticker']} | {row['z_score_threshold']} | {row['z_score_window']} "
                f"| {int(row['total_oos_signals'])} |"
            )
    else:
        lines.append("> All valid — no low-N cells.")

    lines += [
        f"",
        f"---",
        f"",
        f"## Per-Direction Breakdown (Short Side)",
        f"",
        f"| Ticker | Z-Thresh | Z-Window | OOS Signals | Avg E[R] | % E>0 |",
        f"|---|---:|---:|---:|---:|---:|",
    ]

    short_top = (
        heatmap
        .filter(
            (pl.col("direction") == "short")
            & (pl.col("signal_quality") == "valid")
        )
        .sort("avg_oos_exp_r", descending=True)
        .head(10)
    )
    for row in short_top.iter_rows(named=True):
        lines.append(
            f"| {row['ticker']} | {row['z_score_threshold']} | {row['z_score_window']} "
            f"| {int(row['total_oos_signals'])} "
            f"| {float(row['avg_oos_exp_r']):+.4f} "
            f"| {float(row['pct_positive_windows']):.0%} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Notes",
        f"",
        f"- `Avg E[R]`: Average OOS expectancy in reward:risk units across walk-forward windows.",
        f"- `% E>0`: Fraction of OOS windows with positive expectancy (stability indicator).",
        f"- `Eff Cost`: Effective `cost_r` used — relative to avg MAE of the ticker (if `--cost-bps` mode).",
        f"- A config should show **both** positive `Avg E[R]` and high `% E>0` to be considered.",
        f"- `low_n` configs are excluded from the frontier (not meaningful without ≥{args.min_oos_signals_total} OOS signals).",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
