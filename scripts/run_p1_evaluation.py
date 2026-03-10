#!/usr/bin/env python3
"""
P1 Evaluation Script — Volume Filter Ablation + KL/CBK Grid Search

Three experiments in one run:

1. Kinematic Ladder OOS grid (regime_window × accel_window) with volume ON vs OFF
2. Compression Breakout OOS grid (compression_window × compression_factor) with volume ON vs OFF
3. Elastic Band directional-mass ablation: with vs without use_directional_mass
   at the best params found in the P0 grid search

Output per experiment:
  - Heatmap CSV: param combos × (vol_on | vol_off) avg OOS E[R]
  - Ablation comparison table
  - Combined markdown report

Usage:
  uv run python scripts/run_p1_evaluation.py
  uv run python scripts/run_p1_evaluation.py --tickers SPY AAPL META IWM QQQ TSLA
  uv run python scripts/run_p1_evaluation.py --skip-kl --skip-cbk   # only elastic ablation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sys
from itertools import product

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
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.time_utils import et_date_expr


console = Console()


# ── Parameter grids ───────────────────────────────────────────────────────────

KL_REGIME_WINDOWS = [20, 30, 45]
KL_ACCEL_WINDOWS  = [8, 12, 20]

CBK_COMP_WINDOWS    = [15, 20, 30]
CBK_COMP_FACTORS    = [0.70, 0.80, 0.90]
CBK_BREAKOUT_LOOKS  = [15, 20]

# Best Elastic Band params from P0 grid search (per-ticker)
ELASTIC_ABLATION_CONFIGS = [
    # (ticker, z_threshold, z_window)
    ("AAPL", 2.5, 240),
    ("META", 1.25, 360),
    ("TSLA", 1.75, 120),
    ("QQQ", 2.5, 360),
    ("IWM", 2.0, 240),
    ("SPY", 2.0, 240),   # known weak, included for completeness
]


# ── Shared helpers ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1: volume ablation + KL/CBK grid search")
    p.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "META", "IWM", "QQQ", "TSLA"])
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    p.add_argument("--end",   type=date.fromisoformat, default=date(2026, 2, 28))
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months",  type=int, default=3)
    p.add_argument("--ratios",  default="1.0,1.25,1.5,2.0")
    p.add_argument("--cost-bps", type=float, default=8.0)
    p.add_argument("--min-signals", type=int, default=20)
    p.add_argument("--min-oos-signals-total", type=int, default=100)
    p.add_argument("--skip-kl",      action="store_true", help="Skip Kinematic Ladder grid")
    p.add_argument("--skip-cbk",     action="store_true", help="Skip Compression Breakout grid")
    p.add_argument("--skip-elastic", action="store_true", help="Skip Elastic Band ablation")
    p.add_argument("--out-dir", default="data/results")
    return p.parse_args()


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def build_windows(start: date, end: date, train_months: int, test_months: int) -> list[Window]:
    windows: list[Window] = []
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


def cost_r_from_bps(cost_bps: float, avg_mae_d: float, avg_entry: float) -> float:
    if avg_mae_d <= 0 or avg_entry <= 0:
        return 0.05
    return (avg_entry * cost_bps / 10_000.0) / avg_mae_d


def eval_window_direction(
    df_eval: pl.DataFrame,
    direction: str,
    ratios: list[float],
    cost_bps: float,
    min_signals: int,
) -> dict | None:
    """Pick best ratio on df_eval treating it as a test window. Returns None if too few signals."""
    base = df_eval.filter(pl.col("signal")).drop_nulls(
        subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"]
    )
    if direction != "combined":
        base = base.filter(pl.col("signal_direction") == direction)

    n = len(base)
    if n < min_signals:
        return None

    mfe = base["forward_mfe_eod"].to_numpy()
    mae = base["forward_mae_eod"].to_numpy()
    avg_entry = float(base["close"].mean()) if "close" in base.columns else 0.0
    avg_mae_d = float(np.mean(mae))
    effective_cost_r = cost_r_from_bps(cost_bps, avg_mae_d, avg_entry)

    best_exp, best_ratio = -1e9, ratios[0]
    for ratio in ratios:
        wins = mfe >= ratio * mae
        p = float(np.mean(wins))
        exp_r = p * ratio - (1.0 - p) - effective_cost_r
        if exp_r > best_exp:
            best_exp, best_ratio = exp_r, ratio

    wins = mfe >= best_ratio * mae
    p = float(np.mean(wins))
    exp_r = p * best_ratio - (1.0 - p) - effective_cost_r
    return {
        "signals": n,
        "exp_r": round(exp_r, 4),
        "confidence": round(p, 4),
        "effective_cost_r": round(effective_cost_r, 5),
        "ratio": best_ratio,
    }


def walk_forward_strategy(
    df_raw: pl.DataFrame,
    strategy,
    windows: list[Window],
    ratios: list[float],
    cost_bps: float,
    min_signals: int,
    metrics: MetricsCalculator,
) -> list[dict]:
    """Run walk-forward OOS evaluation. Returns rows of per-window per-direction results."""
    df_sig = strategy.generate_signals(df_raw.clone())
    df_eval_all = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))

    rows = []
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
            # Train to pick ratio
            train_res = eval_window_direction(train_df, direction, ratios, cost_bps, min_signals)
            if train_res is None:
                continue
            # Test OOS
            test_res = eval_window_direction(test_df, direction, [train_res["ratio"]], cost_bps, min_signals)
            if test_res is None:
                continue

            rows.append({
                "window_idx": w_idx,
                "train_start": w.train_start.isoformat(),
                "test_start": w.test_start.isoformat(),
                "direction": direction,
                "test_signals": test_res["signals"],
                "test_exp_r": test_res["exp_r"],
                "test_confidence": test_res["confidence"],
                "effective_cost_r": test_res["effective_cost_r"],
            })
    return rows


def aggregate_oos(rows: list[dict], min_total_signals: int) -> dict:
    """Aggregate walk-forward rows to per-direction summary."""
    if not rows:
        return {}
    df = pl.DataFrame(rows)
    result = {}
    for direction in ("combined", "long", "short"):
        sub = df.filter(pl.col("direction") == direction)
        if sub.is_empty():
            continue
        total_sig = int(sub["test_signals"].sum())
        avg_exp = float(sub["test_exp_r"].mean())
        pct_pos = float((sub["test_exp_r"] > 0).mean())
        quality = "valid" if total_sig >= min_total_signals else "low_n"
        result[direction] = {
            "oos_windows": len(sub),
            "total_oos_signals": total_sig,
            "avg_oos_exp_r": round(avg_exp, 4),
            "pct_positive_windows": round(pct_pos, 4),
            "signal_quality": quality,
        }
    return result


def print_ablation_comparison(
    strategy_name: str,
    ticker: str,
    param_label: str,
    on_result: dict,
    off_result: dict,
    direction: str = "combined",
) -> None:
    on = on_result.get(direction, {})
    off = off_result.get(direction, {})
    if not on and not off:
        return

    table = Table(title=f"{ticker} | {strategy_name} | Vol Filter Ablation ({direction})", show_lines=True)
    table.add_column("Config")
    table.add_column("OOS Windows", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Avg E[R]", justify="right")
    table.add_column("% E>0", justify="right")
    table.add_column("Quality")

    for label, res in [("volume ON (base)", on), ("volume OFF (ablation)", off)]:
        if res:
            table.add_row(
                label,
                str(res.get("oos_windows", "—")),
                str(res.get("total_oos_signals", "—")),
                f"{res.get('avg_oos_exp_r', 0.0):+.3f}",
                f"{res.get('pct_positive_windows', 0.0):.0%}",
                str(res.get("signal_quality", "—")),
            )
        else:
            table.add_row(label, "—", "—", "—", "—", "—")

    console.print(table)


# ── Experiment 1: Kinematic Ladder ────────────────────────────────────────────


def run_kl_experiment(
    args, tickers_data: dict, windows: list[Window], ratios: list[float], metrics: MetricsCalculator
) -> list[dict]:
    console.rule("[bold cyan]Experiment 1: Kinematic Ladder Grid (vol ON + OFF)[/]")
    kl_params = list(product(KL_REGIME_WINDOWS, KL_ACCEL_WINDOWS))
    console.print(f"Grid: {len(kl_params)} param combos × 2 (vol on/off) per ticker")

    all_rows = []

    for ticker, df_raw in tickers_data.items():
        for (rw, aw) in kl_params:
            for vol_on in (True, False):
                strat = KinematicLadderStrategy(
                    regime_window=rw,
                    accel_window=aw,
                    use_volume_filter=vol_on,
                    use_time_filter=True,
                )
                label = f"rw={rw},aw={aw},vol={'on' if vol_on else 'off'}"
                rows = walk_forward_strategy(df_raw, strat, windows, ratios, args.cost_bps, args.min_signals, metrics)
                agg = aggregate_oos(rows, args.min_oos_signals_total)
                for direction, stats in agg.items():
                    all_rows.append({
                        "ticker": ticker,
                        "regime_window": rw,
                        "accel_window": aw,
                        "use_volume_filter": vol_on,
                        "direction": direction,
                        **stats,
                    })

        # Print ablation comparison for best default param (rw=30, aw=10)
        for direction in ("combined", "short"):
            on_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["regime_window"] == 30 and r["accel_window"] == 10
                    and r["use_volume_filter"] and r["direction"] == direction
                ),
                {}
            )
            off_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["regime_window"] == 30 and r["accel_window"] == 10
                    and not r["use_volume_filter"] and r["direction"] == direction
                ),
                {}
            )
            print_ablation_comparison(
                "Kinematic Ladder", ticker, "rw=30,aw=10",
                {direction: on_agg}, {direction: off_agg}, direction
            )

    return all_rows


# ── Experiment 2: Compression Breakout ────────────────────────────────────────


def run_cbk_experiment(
    args, tickers_data: dict, windows: list[Window], ratios: list[float], metrics: MetricsCalculator
) -> list[dict]:
    console.rule("[bold cyan]Experiment 2: Compression Breakout Grid (vol ON + OFF)[/]")
    cbk_params = list(product(CBK_COMP_WINDOWS, CBK_COMP_FACTORS, CBK_BREAKOUT_LOOKS))
    console.print(f"Grid: {len(cbk_params)} param combos × 2 (vol on/off) per ticker")

    all_rows = []

    for ticker, df_raw in tickers_data.items():
        for (cw, cf, bl) in cbk_params:
            for vol_on in (True, False):
                strat = CompressionBreakoutStrategy(
                    compression_window=cw,
                    compression_factor=cf,
                    breakout_lookback=bl,
                    use_volume_filter=vol_on,
                    use_time_filter=True,
                )
                rows = walk_forward_strategy(df_raw, strat, windows, ratios, args.cost_bps, args.min_signals, metrics)
                agg = aggregate_oos(rows, args.min_oos_signals_total)
                for direction, stats in agg.items():
                    all_rows.append({
                        "ticker": ticker,
                        "compression_window": cw,
                        "compression_factor": cf,
                        "breakout_lookback": bl,
                        "use_volume_filter": vol_on,
                        "direction": direction,
                        **stats,
                    })

        # Print ablation for default config (cw=20, cf=0.80, bl=20)
        for direction in ("combined", "short"):
            on_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["compression_window"] == 20
                    and r["compression_factor"] == 0.80 and r["breakout_lookback"] == 20
                    and r["use_volume_filter"] and r["direction"] == direction
                ),
                {}
            )
            off_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["compression_window"] == 20
                    and r["compression_factor"] == 0.80 and r["breakout_lookback"] == 20
                    and not r["use_volume_filter"] and r["direction"] == direction
                ),
                {}
            )
            print_ablation_comparison(
                "Compression Breakout", ticker, "cw=20,cf=0.80,bl=20",
                {direction: on_agg}, {direction: off_agg}, direction
            )

    return all_rows


# ── Experiment 3: Elastic Band directional-mass ablation ──────────────────────


def run_elastic_ablation(
    args, tickers_data: dict, windows: list[Window], ratios: list[float], metrics: MetricsCalculator
) -> list[dict]:
    console.rule("[bold cyan]Experiment 3: Elastic Band Directional-Mass Ablation[/]")
    console.print(f"Testing {len(ELASTIC_ABLATION_CONFIGS)} (ticker, z-thresh, z-window) configs with dm ON vs OFF")

    all_rows = []

    for (ticker, z_thresh, z_win) in ELASTIC_ABLATION_CONFIGS:
        if ticker not in tickers_data:
            continue
        df_raw = tickers_data[ticker]

        for dm_on in (True, False):
            strat = ElasticBandReversionStrategy(
                z_score_threshold=z_thresh,
                z_score_window=z_win,
                use_directional_mass=dm_on,
            )
            rows = walk_forward_strategy(df_raw, strat, windows, ratios, args.cost_bps, args.min_signals, metrics)
            agg = aggregate_oos(rows, args.min_oos_signals_total)
            for direction, stats in agg.items():
                all_rows.append({
                    "ticker": ticker,
                    "z_score_threshold": z_thresh,
                    "z_score_window": z_win,
                    "use_directional_mass": dm_on,
                    "direction": direction,
                    **stats,
                })

        # Print ablation comparison
        label = f"z={z_thresh}/w={z_win}"
        for direction in ("combined", "short"):
            on_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["z_score_threshold"] == z_thresh
                    and r["z_score_window"] == z_win and r["use_directional_mass"]
                    and r["direction"] == direction
                ),
                {}
            )
            off_agg = next(
                (
                    {k: v for k, v in r.items() if k in ("oos_windows","total_oos_signals","avg_oos_exp_r","pct_positive_windows","signal_quality")}
                    for r in all_rows
                    if r["ticker"] == ticker and r["z_score_threshold"] == z_thresh
                    and r["z_score_window"] == z_win and not r["use_directional_mass"]
                    and r["direction"] == direction
                ),
                {}
            )
            print_ablation_comparison(
                "Elastic Band (DM ablation)", ticker, label,
                {direction: on_agg}, {direction: off_agg}, direction
            )

    return all_rows


# ── Report writing ─────────────────────────────────────────────────────────────


def write_p1_report(
    path: Path,
    kl_rows: list[dict],
    cbk_rows: list[dict],
    elastic_rows: list[dict],
    args,
    stamp: str,
) -> None:
    lines = [
        "# P1 Evaluation Report — Volume Ablation + KL/CBK Grid",
        "",
        f"**Generated:** {stamp}  ",
        f"**Tickers:** {', '.join(args.tickers)}  ",
        f"**Range:** {args.start} → {args.end}  ",
        f"**Walk-forward:** {args.train_months}m train / {args.test_months}m test  ",
        f"**Friction:** relative {args.cost_bps} bps  ",
        "",
        "---",
        "",
    ]

    def add_section(title: str, rows: list[dict], vol_key: str, param_keys: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("> No data — experiment was skipped.")
            lines.append("")
            return

        df = pl.DataFrame(rows).filter(pl.col("signal_quality") == "valid")
        if df.is_empty():
            lines.append("> No valid cells (all below min OOS signal threshold).")
            lines.append("")
            return

        # Best configs overall (combined direction, vol ON)
        best = (
            df.filter((pl.col("direction") == "combined") & pl.col(vol_key))
            .sort("avg_oos_exp_r", descending=True)
            .head(10)
        )
        # Best configs with vol OFF (ablation)
        best_off = (
            df.filter((pl.col("direction") == "combined") & ~pl.col(vol_key))
            .sort("avg_oos_exp_r", descending=True)
            .head(10)
        )

        col_header = " | ".join(["Ticker"] + param_keys + ["OOS Wins", "Signals", "Avg E[R]", "% E>0"])
        col_sep = "|---" * (len(param_keys) + 5) + "|"
        lines.append(f"### Volume Filter ON — Combined Direction")
        lines.append(f"| {col_header} |")
        lines.append(col_sep)
        for row in best.iter_rows(named=True):
            vals = [str(row[k]) for k in param_keys]
            lines.append(
                f"| {row['ticker']} | {' | '.join(vals)} | {row['oos_windows']} "
                f"| {int(row['total_oos_signals'])} "
                f"| {float(row['avg_oos_exp_r']):+.4f} "
                f"| {float(row['pct_positive_windows']):.0%} |"
            )

        lines.append("")
        lines.append(f"### Volume Filter OFF (Ablation) — Combined Direction")
        lines.append(f"| {col_header} |")
        lines.append(col_sep)
        for row in best_off.iter_rows(named=True):
            vals = [str(row[k]) for k in param_keys]
            lines.append(
                f"| {row['ticker']} | {' | '.join(vals)} | {row['oos_windows']} "
                f"| {int(row['total_oos_signals'])} "
                f"| {float(row['avg_oos_exp_r']):+.4f} "
                f"| {float(row['pct_positive_windows']):.0%} |"
            )
        lines.append("")

    add_section("Kinematic Ladder", kl_rows, "use_volume_filter", ["regime_window", "accel_window"])
    lines.append("---")
    lines.append("")
    add_section("Compression Breakout", cbk_rows, "use_volume_filter", ["compression_window", "compression_factor", "breakout_lookback"])
    lines.append("---")
    lines.append("")
    add_section("Elastic Band — Directional Mass Ablation", elastic_rows, "use_directional_mass", ["z_score_threshold", "z_score_window"])

    path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    windows = build_windows(args.start, args.end, args.train_months, args.test_months)

    if not windows:
        console.print("[red]No walk-forward windows for the given date range.[/]")
        return

    console.rule("[bold green]P1 Evaluation — Volume Ablation + KL/CBK Grid Search[/]")
    console.print(
        f"Tickers: {args.tickers} | Range: {args.start}→{args.end} | "
        f"Walk-forward: {args.train_months}m/{args.test_months}m | "
        f"Friction: {args.cost_bps} bps\n"
        f"Experiments: KL={'skip' if args.skip_kl else 'run'}, "
        f"CBK={'skip' if args.skip_cbk else 'run'}, "
        f"Elastic={'skip' if args.skip_elastic else 'run'}"
    )

    # Load and enrich all ticker data once (shared across experiments)
    storage = LocalStorage()
    physics  = PhysicsEngine()
    metrics  = MetricsCalculator()

    tickers_data: dict[str, pl.DataFrame] = {}
    for ticker in args.tickers:
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            console.print(f"[yellow]No data for {ticker}, skipping[/]")
            continue
        tickers_data[ticker] = physics.enrich(df)

    if not tickers_data:
        console.print("[red]No ticker data loaded. Exiting.[/]")
        return

    kl_rows, cbk_rows, elastic_rows = [], [], []

    if not args.skip_kl:
        kl_rows = run_kl_experiment(args, tickers_data, windows, ratios, metrics)
    if not args.skip_cbk:
        cbk_rows = run_cbk_experiment(args, tickers_data, windows, ratios, metrics)
    if not args.skip_elastic:
        elastic_rows = run_elastic_ablation(args, tickers_data, windows, ratios, metrics)

    # Write artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    kl_path      = out_dir / f"p1_kl_grid_{stamp}.csv"
    cbk_path     = out_dir / f"p1_cbk_grid_{stamp}.csv"
    elastic_path = out_dir / f"p1_elastic_ablation_{stamp}.csv"
    report_path  = out_dir / f"p1_report_{stamp}.md"

    if kl_rows:
        pl.DataFrame(kl_rows).write_csv(kl_path)
        console.print(f"Saved KL grid    → [green]{kl_path}[/]")
    if cbk_rows:
        pl.DataFrame(cbk_rows).write_csv(cbk_path)
        console.print(f"Saved CBK grid   → [green]{cbk_path}[/]")
    if elastic_rows:
        pl.DataFrame(elastic_rows).write_csv(elastic_path)
        console.print(f"Saved Elastic    → [green]{elastic_path}[/]")

    write_p1_report(report_path, kl_rows, cbk_rows, elastic_rows, args, stamp)
    console.print(f"Saved report     → [green]{report_path}[/]")

    db = ResultsDB()
    run_id = db.start_run(
        script="run_p1_evaluation.py",
        params={
            "tickers": args.tickers,
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "cost_bps": args.cost_bps,
            "skip_kl": args.skip_kl,
            "skip_cbk": args.skip_cbk,
            "skip_elastic": args.skip_elastic,
        },
    )
    for df_rows, artifact_type in [
        (kl_rows, "p1_kl_grid"),
        (cbk_rows, "p1_cbk_grid"),
        (elastic_rows, "p1_elastic_ablation"),
    ]:
        if df_rows:
            db.ingest_dataframe(
                run_id=run_id,
                script="run_p1_evaluation.py",
                artifact_type=artifact_type,
                source_path=str(out_dir),
                df=pl.DataFrame(df_rows),
            )
    db.finish_run(run_id)
    console.print(f"Saved DB rows    → [green]{db.db_path}[/] (run_id={run_id})")


if __name__ == "__main__":
    main()
