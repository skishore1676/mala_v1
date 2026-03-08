#!/usr/bin/env python3
"""
Run novel directional strategy ideas and compare edge metrics.

Usage:
    python scripts/run_novel_ideas.py
    python scripts/run_novel_ideas.py --tickers SPY QQQ IWM --start 2025-01-01 --end 2026-02-28
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.config import settings
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy
from src.strategy.regime_router import RegimeRouterStrategy

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest novel directional strategy ideas")
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
    parser.add_argument(
        "--ratios",
        default="1.0,1.25,1.5,2.0",
        help="Comma-separated reward:risk ratios for robustness checks.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Monte Carlo bootstrap iterations for expectancy robustness.",
    )
    parser.add_argument(
        "--cost-r",
        type=float,
        default=0.05,
        help="Per-trade friction in R units (slippage+fees+decay proxy).",
    )
    return parser.parse_args()


def _print_summary_table(ticker: str, strategy_name: str, summary: pl.DataFrame) -> None:
    table = Table(title=f"{ticker} | {strategy_name}", show_lines=True)
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


def _evaluate_ratio_grid(
    *,
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


def _print_ratio_table(ticker: str, strategy_name: str, rows: list[dict]) -> None:
    if not rows:
        return
    table = Table(title=f"{ticker} | {strategy_name} | Monte Carlo Robustness", show_lines=True)
    table.add_column("R:R", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("BE Conf", justify="right")
    table.add_column("P(E>0)", justify="right")
    table.add_column("Exp(R)", justify="right")

    for row in rows:
        table.add_row(
            f"{float(row['ratio']):.2f}",
            str(row["signals"]),
            f"{float(row['confidence']):.2%}",
            f"{float(row['breakeven_confidence']):.2%}",
            f"{float(row['prob_positive_exp']):.1%}",
            f"{float(row['exp_r']):+.3f}",
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()
    rng = np.random.default_rng(7)

    strategies = [
        ElasticBandReversionStrategy(
            stretch_pct=0.002,
            volume_multiplier=1.15,
            volume_ma_period=settings.volume_ma_period,
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
    ]

    results_rows: list[dict] = []
    robustness_rows: list[dict] = []

    console.rule("[bold green]Novel Strategy Backtest[/]")
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

        for strategy in strategies:
            df_sig = strategy.generate_signals(df.clone())
            signal_count = df_sig.filter(pl.col("signal")).height
            if signal_count == 0:
                console.print(f"[yellow]{strategy.name}: no signals[/]")
                continue

            df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))
            summary = metrics.summarise_directional_signals(df_eval)

            if summary.is_empty():
                console.print(f"[yellow]{strategy.name}: no valid directional metrics[/]")
                continue

            _print_summary_table(ticker, strategy.name, summary)

            for row in summary.iter_rows(named=True):
                row_out = dict(row)
                row_out["ticker"] = ticker
                row_out["strategy"] = strategy.name
                results_rows.append(row_out)

            base = (
                df_eval.filter(pl.col("signal"))
                .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod"])
                .select(["forward_mfe_eod", "forward_mae_eod"])
            )
            ratio_rows = _evaluate_ratio_grid(
                mfe=base["forward_mfe_eod"].to_numpy(),
                mae=base["forward_mae_eod"].to_numpy(),
                ratios=ratios,
                bootstrap_iters=args.bootstrap_iters,
                cost_r=args.cost_r,
                rng=rng,
            )
            _print_ratio_table(ticker, strategy.name, ratio_rows)

            for row in ratio_rows:
                row_out = dict(row)
                row_out["ticker"] = ticker
                row_out["strategy"] = strategy.name
                robustness_rows.append(row_out)

    if not results_rows:
        console.print("[red]No results generated.[/]")
        return

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_df = pl.DataFrame(results_rows)
    csv_path = out_dir / f"novel_strategy_summary_{stamp}.csv"
    json_path = out_dir / f"novel_strategy_summary_{stamp}.json"
    robustness_csv_path = out_dir / f"novel_strategy_robustness_{stamp}.csv"

    results_df.write_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(results_rows, f, indent=2, default=str)
    if robustness_rows:
        pl.DataFrame(robustness_rows).write_csv(robustness_csv_path)

    console.print(f"\nSaved summary CSV -> [green]{csv_path}[/]")
    console.print(f"Saved summary JSON -> [green]{json_path}[/]")
    if robustness_rows:
        console.print(f"Saved robustness CSV -> [green]{robustness_csv_path}[/]")


if __name__ == "__main__":
    main()
