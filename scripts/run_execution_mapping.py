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

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig, stress_from_win_flags
from src.oracle.results_db import ResultsDB
from src.strategy.factory import build_strategy_by_name
from src.time_utils import et_date_expr


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


def latest_file(out_dir: Path, prefix: str) -> Path:
    files = sorted(out_dir.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found for {prefix}_*.csv in {out_dir}")
    return files[-1]


def option_mapping_for(strategy: str, direction: str) -> dict[str, str]:
    if strategy == "Elastic Band Reversion" and direction == "short":
        return {
            "structure": "put_debit_spread",
            "dte": "7-14",
            "delta_plan": "long 0.35-0.45 / short 0.15-0.25",
            "entry_window_et": "09:45-15:00",
            "profit_take": "60-80% spread value",
            "risk_rule": "hard stop at -50% premium",
        }
    if strategy == "Compression Expansion Breakout" and direction == "short":
        return {
            "structure": "put_debit_spread",
            "dte": "7-21",
            "delta_plan": "long 0.30-0.40 / short 0.12-0.22",
            "entry_window_et": "09:40-14:30",
            "profit_take": "55-75% spread value",
            "risk_rule": "hard stop at -45% premium",
        }
    if direction == "long":
        return {
            "structure": "call_debit_spread",
            "dte": "7-21",
            "delta_plan": "long 0.30-0.45 / short 0.10-0.25",
            "entry_window_et": "09:45-14:30",
            "profit_take": "50-70% spread value",
            "risk_rule": "hard stop at -45% premium",
        }
    return {
        "structure": "put_debit_spread",
        "dte": "7-21",
        "delta_plan": "long 0.30-0.45 / short 0.10-0.25",
        "entry_window_et": "09:45-14:30",
        "profit_take": "50-70% spread value",
        "risk_rule": "hard stop at -45% premium",
    }


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

    holdout_summary_path = Path(args.holdout_summary) if args.holdout_summary else latest_file(out_dir, "holdout_validation_summary")
    holdout_detail_path = Path(args.holdout_detail) if args.holdout_detail else latest_file(out_dir, "holdout_validation_detail")

    holdout_summary = pl.read_csv(holdout_summary_path)
    holdout_detail = pl.read_csv(holdout_detail_path)
    promoted = holdout_summary.filter(pl.col("decision") == "promote_to_execution_mapping")
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

    rows: list[dict] = []

    for cand in promoted.iter_rows(named=True):
        ticker = cand["ticker"]
        strategy_name = cand["strategy"]
        direction = cand["direction"]

        ratio_candidates = (
            holdout_detail
            .filter(
                (pl.col("ticker") == ticker)
                & (pl.col("strategy") == strategy_name)
                & (pl.col("direction") == direction)
                & pl.col("selected_ratio").is_not_null()
            )
            .get_column("selected_ratio")
            .to_list()
        )
        if not ratio_candidates:
            continue
        selected_ratio = float(np.median(np.array(ratio_candidates, dtype=np.float64)))

        df_raw = storage.load_bars(ticker, args.start, args.holdout_end)
        if df_raw.is_empty():
            continue
        df = physics.enrich(df_raw)
        strategy = build_strategy_by_name(strategy_name)
        df_sig = strategy.generate_signals(df)
        df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

        base = (
            df_eval.filter(
                (et_date_expr("timestamp") >= args.holdout_start)
                & (et_date_expr("timestamp") <= args.holdout_end)
                & pl.col("signal")
                & pl.col("forward_mfe_eod").is_not_null()
                & pl.col("forward_mae_eod").is_not_null()
            )
        )
        if direction != "combined":
            base = base.filter(pl.col("signal_direction") == direction)
        if base.is_empty():
            continue

        mfe = base["forward_mfe_eod"].to_numpy()
        mae = base["forward_mae_eod"].to_numpy()
        wins = mfe >= (selected_ratio * mae)
        p = float(np.mean(wins))
        base_exp_r = p * selected_ratio - (1.0 - p) - args.base_cost_r

        stress = stress_from_win_flags(
            win_flags=wins,
            ratio=selected_ratio,
            config=stress_cfg,
        )
        mapping = option_mapping_for(strategy_name, direction)

        row = {
            "ticker": ticker,
            "strategy": strategy_name,
            "direction": direction,
            "selected_ratio": selected_ratio,
            "holdout_trades": int(len(wins)),
            "holdout_win_rate": round(p, 4),
            "base_exp_r": round(base_exp_r, 4),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in stress.items()},
            **mapping,
        }
        rows.append(row)

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

