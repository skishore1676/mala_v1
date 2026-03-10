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
  uv run python scripts/run_elastic_grid.py --tickers SPY AAPL META IWM QQQ TSLA
  uv run python scripts/run_elastic_grid.py --cost-bps 8 --tickers AAPL META

Output:
  - data/results/elastic_grid_detail_<stamp>.csv  (all window×direction rows)
  - data/results/elastic_grid_heatmap_<stamp>.csv (aggregated per param cell)
  - data/results/elastic_grid_report_<stamp>.md   (human-readable frontier)
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from itertools import product
from pathlib import Path
import sys

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.results_db import ResultsDB
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.time_utils import et_date_expr


console = Console()

# ── Parameter grid ────────────────────────────────────────────────────────────

Z_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
Z_WINDOWS    = [60, 120, 180, 240, 360]


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def build_windows(start: date, end: date, train_months: int, test_months: int):
    from dataclasses import dataclass

    @dataclass
    class Window:
        train_start: date
        train_end: date
        test_start: date
        test_end: date

    windows = []
    cursor = start
    while True:
        ts = cursor
        te = ts + relativedelta(months=train_months) - relativedelta(days=1)
        vs = te + relativedelta(days=1)
        ve = vs + relativedelta(months=test_months) - relativedelta(days=1)
        if ve > end:
            break
        windows.append(Window(ts, te, vs, ve))
        cursor += relativedelta(months=test_months)
    return windows


def cost_r_from_bps(cost_bps: float, avg_mae_dollars: float, entry_price: float) -> float:
    """
    Convert a basis-point transaction cost to reward:risk units.

    cost_dollars = entry_price * cost_bps / 10_000
    cost_r       = cost_dollars / avg_mae_dollars

    If avg_mae_dollars is tiny / zero we fall back to a small fixed value.
    """
    if avg_mae_dollars is None or avg_mae_dollars <= 0:
        return 0.05  # safe fallback
    cost_dollars = entry_price * cost_bps / 10_000.0
    return cost_dollars / avg_mae_dollars


def eval_direction(
    df: pl.DataFrame,
    direction: str,
    ratio: float,
    cost_r: float,
) -> dict:
    """Evaluate a single direction subset at a given ratio and cost_r."""
    base = df.filter(pl.col("signal")).drop_nulls(
        subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"]
    )
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    n = len(base)
    if n == 0:
        return {"signals": 0, "confidence": None, "exp_r": None, "avg_mae": None, "avg_entry": None}

    mfe = base["forward_mfe_eod"].to_numpy()
    mae = base["forward_mae_eod"].to_numpy()

    # avg entry price for relative cost conversion
    avg_entry = float(base["close"].mean()) if "close" in base.columns else 0.0
    avg_mae = float(np.mean(mae))

    wins = mfe >= (ratio * mae)
    p = float(np.mean(wins))
    exp_r = p * ratio - (1.0 - p) - cost_r
    return {
        "signals": n,
        "confidence": round(p, 4),
        "exp_r": round(exp_r, 4),
        "avg_mae": round(avg_mae, 6),
        "avg_entry": round(avg_entry, 4),
    }


def pick_best_ratio(
    train_df: pl.DataFrame,
    direction: str,
    ratios: list[float],
    cost_r: float,
    min_signals: int,
) -> tuple[float | None, float | None]:
    """Choose the ratio with highest train expectancy."""
    best_ratio, best_exp = None, -1e9
    for ratio in ratios:
        stats = eval_direction(train_df, direction, ratio, cost_r)
        if stats["signals"] < min_signals or stats["exp_r"] is None:
            continue
        if float(stats["exp_r"]) > best_exp:
            best_exp = float(stats["exp_r"])
            best_ratio = ratio
    return best_ratio, (best_exp if best_ratio is not None else None)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    windows = build_windows(args.start, args.end, args.train_months, args.test_months)

    if not windows:
        console.print("[red]No walk-forward windows for the given date range.[/]")
        return

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

    storage = LocalStorage()
    physics  = PhysicsEngine()
    metrics  = MetricsCalculator()

    grid_params = list(product(Z_THRESHOLDS, Z_WINDOWS))
    all_detail_rows: list[dict] = []

    for ticker in args.tickers:
        console.rule(f"[cyan]{ticker}[/]")
        df_raw = storage.load_bars(ticker, args.start, args.end)
        if df_raw.is_empty():
            console.print(f"[yellow]  No cached data for {ticker}, skipping.[/]")
            continue
        df_raw = physics.enrich(df_raw)

        for (z_thresh, z_win) in grid_params:
            label = f"z={z_thresh:.2f}/w={z_win}"
            strat = ElasticBandReversionStrategy(
                z_score_threshold=z_thresh,
                z_score_window=z_win,
            )

            df_sig = strat.generate_signals(df_raw.clone())
            df_eval_all = metrics.add_directional_forward_metrics(
                df_sig, snapshot_windows=(30, 60)
            )

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
                    # Determine cost_r for this tick/direction/window
                    if use_relative:
                        # Estimate avg_mae and avg_entry from test set for realistic cost
                        base_for_cost = test_df.filter(pl.col("signal")).drop_nulls(
                            subset=["forward_mae_eod", "signal_direction"]
                        )
                        if direction != "combined":
                            base_for_cost = base_for_cost.filter(
                                pl.col("signal_direction") == direction
                            )
                        if len(base_for_cost) > 0 and "close" in base_for_cost.columns:
                            avg_mae_d = float(base_for_cost["forward_mae_eod"].mean() or 0)
                            avg_entry_d = float(base_for_cost["close"].mean() or 0)
                        else:
                            avg_mae_d, avg_entry_d = 0.0, 0.0
                        effective_cost_r = cost_r_from_bps(cost_bps, avg_mae_d, avg_entry_d)
                    else:
                        effective_cost_r = fixed_cost_r
                        avg_mae_d, avg_entry_d = None, None

                    # Pick ratio on train
                    best_ratio, best_train_exp = pick_best_ratio(
                        train_df, direction, ratios, effective_cost_r, args.min_signals
                    )
                    if best_ratio is None:
                        continue

                    # Evaluate on OOS
                    test_stats = eval_direction(test_df, direction, best_ratio, effective_cost_r)
                    if test_stats["signals"] < args.min_signals or test_stats["exp_r"] is None:
                        continue

                    all_detail_rows.append({
                        "ticker": ticker,
                        "z_score_threshold": z_thresh,
                        "z_score_window": z_win,
                        "direction": direction,
                        "window_idx": w_idx,
                        "train_start": w.train_start.isoformat(),
                        "train_end": w.train_end.isoformat(),
                        "test_start": w.test_start.isoformat(),
                        "test_end": w.test_end.isoformat(),
                        "selected_ratio": best_ratio,
                        "train_exp_r": round(float(best_train_exp), 4),
                        "test_signals": int(test_stats["signals"]),
                        "test_confidence": test_stats["confidence"],
                        "test_exp_r": test_stats["exp_r"],
                        "effective_cost_r": round(effective_cost_r, 5),
                        "avg_mae_dollars": round(avg_mae_d, 5) if avg_mae_d is not None else None,
                        "avg_entry_price": round(avg_entry_d, 4) if avg_entry_d is not None else None,
                    })

        console.print(f"  {ticker}: {len([r for r in all_detail_rows if r['ticker'] == ticker])} detail rows generated")

    if not all_detail_rows:
        console.print("[red]No rows generated. Check that tickers have cached data.[/]")
        return

    detail_df = pl.DataFrame(all_detail_rows)

    # ── Aggregate into heatmap ───────────────────────────────────────────────
    heatmap = (
        detail_df
        .group_by(["ticker", "z_score_threshold", "z_score_window", "direction"])
        .agg([
            pl.len().alias("oos_windows"),
            pl.col("test_signals").sum().alias("total_oos_signals"),
            pl.col("test_exp_r").mean().alias("avg_oos_exp_r"),
            pl.col("test_exp_r").median().alias("med_oos_exp_r"),
            (pl.col("test_exp_r") > 0).mean().alias("pct_positive_windows"),
            pl.col("test_confidence").mean().alias("avg_confidence"),
            pl.col("effective_cost_r").mean().alias("avg_effective_cost_r"),
        ])
        .with_columns(
            pl.when(pl.col("total_oos_signals") >= args.min_oos_signals_total)
            .then(pl.lit("valid"))
            .otherwise(pl.lit("low_n"))
            .alias("signal_quality")
        )
        .sort(["ticker", "direction", "avg_oos_exp_r"], descending=[False, False, True])
    )

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
