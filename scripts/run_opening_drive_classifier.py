#!/usr/bin/env python3
"""
Run Opening Drive Classifier strategy and report directional expectancy.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.config import settings
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.results_db import ResultsDB
from src.strategy.opening_drive_classifier import OpeningDriveClassifierStrategy


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Opening Drive Classifier")
    parser.add_argument("--tickers", nargs="+", default=settings.default_tickers)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=365 * settings.lookback_years),
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
    )
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--cost-r", type=float, default=0.05)
    parser.add_argument("--opening-window-minutes", type=int, default=25)
    parser.add_argument("--entry-start-offset-minutes", type=int, default=25)
    parser.add_argument("--entry-end-offset-minutes", type=int, default=120)
    parser.add_argument("--min-drive-return-pct", type=float, default=0.0015)
    parser.add_argument("--breakout-buffer-pct", type=float, default=0.0)
    parser.add_argument("--volume-multiplier", type=float, default=1.2)
    parser.add_argument("--allow-long", action="store_true", default=False)
    parser.add_argument("--allow-short", action="store_true", default=False)
    parser.add_argument("--enable-continue", action="store_true", default=False)
    parser.add_argument("--enable-fail", action="store_true", default=False)
    parser.add_argument("--strategy-label", default="")
    return parser.parse_args()


def eval_ratio_grid(
    mfe: np.ndarray,
    mae: np.ndarray,
    ratios: list[float],
    bootstrap_iters: int,
    cost_r: float,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    n = len(mfe)
    if n == 0:
        return rows
    for ratio in ratios:
        wins = mfe >= (ratio * mae)
        p_hat = float(np.mean(wins))
        p_be = float((1.0 + cost_r) / (1.0 + ratio))
        exp_r = p_hat * ratio - (1.0 - p_hat) - cost_r
        p_boot = rng.binomial(n=n, p=p_hat, size=bootstrap_iters) / n
        exp_boot = p_boot * ratio - (1.0 - p_boot) - cost_r
        rows.append({
            "ratio": ratio,
            "signals": n,
            "confidence": round(p_hat, 4),
            "breakeven_confidence": round(p_be, 4),
            "edge_vs_breakeven": round(p_hat - p_be, 4),
            "exp_r": round(float(exp_r), 4),
            "exp_r_p05": round(float(np.quantile(exp_boot, 0.05)), 4),
            "exp_r_p95": round(float(np.quantile(exp_boot, 0.95)), 4),
            "prob_positive_exp": round(float(np.mean(exp_boot > 0.0)), 4),
        })
    return rows


def print_summary_table(ticker: str, summary: pl.DataFrame) -> None:
    table = Table(title=f"{ticker} | Opening Drive Classifier", show_lines=True)
    table.add_column("Direction")
    table.add_column("Signals", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Confidence(2:1)", justify="right")
    table.add_column("Avg MFE/MAE", justify="right")
    for row in summary.iter_rows(named=True):
        table.add_row(
            str(row.get("direction")),
            str(row.get("total_signals")),
            str(row.get("wins")),
            f"{float(row.get('confidence_2to1', 0.0)):.2%}",
            str(row.get("avg_mfe_mae_ratio")),
        )
    console.print(table)


def print_mode_table(ticker: str, mode_df: pl.DataFrame) -> None:
    if mode_df.is_empty():
        return
    table = Table(title=f"{ticker} | Opening Drive Modes", show_lines=True)
    table.add_column("Mode")
    table.add_column("Direction")
    table.add_column("Signals", justify="right")
    table.add_column("Avg MFE", justify="right")
    table.add_column("Avg MAE", justify="right")
    table.add_column("MFE/MAE", justify="right")
    for row in mode_df.iter_rows(named=True):
        table.add_row(
            str(row["opening_drive_mode"]),
            str(row["signal_direction"]),
            str(int(row["signals"])),
            f"{float(row['avg_mfe']):.4f}",
            f"{float(row['avg_mae']):.4f}",
            f"{float(row['avg_mfe_mae_ratio']):.3f}" if row["avg_mfe_mae_ratio"] is not None else "n/a",
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()
    rng = np.random.default_rng(17)

    allow_long = args.allow_long or (not args.allow_long and not args.allow_short)
    allow_short = args.allow_short or (not args.allow_long and not args.allow_short)
    enable_continue = args.enable_continue or (not args.enable_continue and not args.enable_fail)
    enable_fail = args.enable_fail or (not args.enable_continue and not args.enable_fail)

    strategy = OpeningDriveClassifierStrategy(
        opening_window_minutes=args.opening_window_minutes,
        entry_start_offset_minutes=args.entry_start_offset_minutes,
        entry_end_offset_minutes=args.entry_end_offset_minutes,
        min_drive_return_pct=args.min_drive_return_pct,
        breakout_buffer_pct=args.breakout_buffer_pct,
        volume_multiplier=args.volume_multiplier,
        allow_long=allow_long,
        allow_short=allow_short,
        enable_continue=enable_continue,
        enable_fail=enable_fail,
        strategy_label=args.strategy_label if args.strategy_label else None,
    )

    summary_rows: list[dict] = []
    robustness_rows: list[dict] = []
    mode_rows: list[dict] = []

    console.rule("[bold green]Opening Drive Classifier Backtest[/]")
    console.print(
        f"Tickers: {args.tickers} | Range: {args.start} -> {args.end} | "
        f"Ratios: {ratios} | Bootstrap: {args.bootstrap_iters} | cost_r={args.cost_r}"
    )

    for ticker in args.tickers:
        console.rule(f"[bold cyan]{ticker}")
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            console.print(f"[yellow]No data for {ticker}, skipping.[/]")
            continue

        df = physics.enrich(df)
        df_sig = strategy.generate_signals(df)
        if df_sig.filter(pl.col("signal")).is_empty():
            console.print("[yellow]No signals.[/]")
            continue

        df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))
        summary = metrics.summarise_directional_signals(df_eval)
        if summary.is_empty():
            console.print("[yellow]No valid directional metrics.[/]")
            continue

        print_summary_table(ticker, summary)

        for row in summary.iter_rows(named=True):
            out = dict(row)
            out["ticker"] = ticker
            out["strategy"] = strategy.name
            summary_rows.append(out)

        mode_df = (
            df_eval.filter(pl.col("signal"))
            .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod"])
            .group_by(["opening_drive_mode", "signal_direction"])
            .agg([
                pl.len().alias("signals"),
                pl.col("forward_mfe_eod").mean().alias("avg_mfe"),
                pl.col("forward_mae_eod").mean().alias("avg_mae"),
            ])
            .with_columns(
                pl.when(pl.col("avg_mae") > 0)
                .then(pl.col("avg_mfe") / pl.col("avg_mae"))
                .otherwise(pl.lit(None))
                .alias("avg_mfe_mae_ratio")
            )
            .sort(["opening_drive_mode", "signal_direction"])
        )
        print_mode_table(ticker, mode_df)

        for row in mode_df.iter_rows(named=True):
            out = dict(row)
            out["ticker"] = ticker
            out["strategy"] = strategy.name
            mode_rows.append(out)

        base = (
            df_eval.filter(pl.col("signal"))
            .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod"])
            .select(["forward_mfe_eod", "forward_mae_eod"])
        )
        ratio_rows = eval_ratio_grid(
            mfe=base["forward_mfe_eod"].to_numpy(),
            mae=base["forward_mae_eod"].to_numpy(),
            ratios=ratios,
            bootstrap_iters=args.bootstrap_iters,
            cost_r=args.cost_r,
            rng=rng,
        )
        for row in ratio_rows:
            out = dict(row)
            out["ticker"] = ticker
            out["strategy"] = strategy.name
            robustness_rows.append(out)

    if not summary_rows:
        console.print("[red]No results generated.[/]")
        return

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_df = pl.DataFrame(summary_rows)
    robustness_df = pl.DataFrame(robustness_rows) if robustness_rows else pl.DataFrame()
    mode_df = pl.DataFrame(mode_rows) if mode_rows else pl.DataFrame()

    summary_path = out_dir / f"opening_drive_summary_{stamp}.csv"
    robustness_path = out_dir / f"opening_drive_robustness_{stamp}.csv"
    mode_path = out_dir / f"opening_drive_mode_summary_{stamp}.csv"
    summary_df.write_csv(summary_path)
    if not robustness_df.is_empty():
        robustness_df.write_csv(robustness_path)
    if not mode_df.is_empty():
        mode_df.write_csv(mode_path)

    params = {
        "tickers": args.tickers,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "ratios": ratios,
        "bootstrap_iters": args.bootstrap_iters,
        "cost_r": args.cost_r,
        "opening_window_minutes": args.opening_window_minutes,
        "entry_start_offset_minutes": args.entry_start_offset_minutes,
        "entry_end_offset_minutes": args.entry_end_offset_minutes,
        "min_drive_return_pct": args.min_drive_return_pct,
        "breakout_buffer_pct": args.breakout_buffer_pct,
        "volume_multiplier": args.volume_multiplier,
        "allow_long": allow_long,
        "allow_short": allow_short,
        "enable_continue": enable_continue,
        "enable_fail": enable_fail,
        "strategy_label": args.strategy_label,
    }

    db = ResultsDB()
    run_id = db.start_run(script="run_opening_drive_classifier.py", params=params)
    db.ingest_dataframe(
        run_id=run_id,
        script="run_opening_drive_classifier.py",
        artifact_type="opening_drive_summary",
        source_path=str(summary_path),
        df=summary_df,
    )
    if not robustness_df.is_empty():
        db.ingest_dataframe(
            run_id=run_id,
            script="run_opening_drive_classifier.py",
            artifact_type="opening_drive_robustness",
            source_path=str(robustness_path),
            df=robustness_df,
        )
    if not mode_df.is_empty():
        db.ingest_dataframe(
            run_id=run_id,
            script="run_opening_drive_classifier.py",
            artifact_type="opening_drive_mode_summary",
            source_path=str(mode_path),
            df=mode_df,
        )
    db.finish_run(run_id)

    console.print(f"\nSaved summary -> [green]{summary_path}[/]")
    if not robustness_df.is_empty():
        console.print(f"Saved robustness -> [green]{robustness_path}[/]")
    if not mode_df.is_empty():
        console.print(f"Saved mode summary -> [green]{mode_path}[/]")
    console.print(f"Saved DB rows -> [green]{db.db_path}[/] (run_id={run_id})")


if __name__ == "__main__":
    main()
