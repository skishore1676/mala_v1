#!/usr/bin/env python3
"""
Measurement Sensitivity + Monte Carlo Wrapper

Compares multiple reward:risk thresholds (e.g., 1:1, 1.5:1, 2:1)
using directional forward metrics and bootstrap confidence intervals.

Usage:
  python scripts/run_measurement_sensitivity.py
  python scripts/run_measurement_sensitivity.py --ratios 1.0,1.5,2.0 --bootstrap-iters 5000
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

# Ensure src import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.config import settings
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.strategy.elastic_band_reversion import ElasticBandReversionStrategy
from src.strategy.kinematic_ladder import KinematicLadderStrategy


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reward:risk sensitivity and Monte Carlo robustness checks"
    )
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
        help="Comma-separated reward:risk ratios to evaluate",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=3000,
        help="Bootstrap iterations for confidence distribution",
    )
    parser.add_argument(
        "--cost-r",
        type=float,
        default=0.05,
        help="Per-trade friction in R units (slippage+fees+decay proxy)",
    )
    return parser.parse_args()


def _evaluate_group(
    *,
    mfe: np.ndarray,
    mae: np.ndarray,
    ratio: float,
    n_boot: int,
    cost_r: float,
    rng: np.random.Generator,
) -> dict:
    n = len(mfe)
    if n == 0:
        return {
            "signals": 0,
            "confidence": None,
            "breakeven_confidence": None,
            "edge_vs_breakeven": None,
            "exp_r": None,
            "exp_r_p05": None,
            "exp_r_p95": None,
            "prob_positive_exp": None,
        }

    wins = mfe >= (ratio * mae)
    p_hat = float(np.mean(wins))

    # If payoff is +ratio R for winners and -1 R for losers:
    # E[R] = p*ratio - (1-p) - cost_r
    # Break-even p = (1 + cost_r) / (1 + ratio)
    p_be = float((1.0 + cost_r) / (1.0 + ratio))

    exp_r = p_hat * ratio - (1.0 - p_hat) - cost_r

    # Bootstrap distribution of win probability (Bernoulli approximation)
    p_boot = rng.binomial(n=n, p=p_hat, size=n_boot) / n
    exp_boot = p_boot * ratio - (1.0 - p_boot) - cost_r

    return {
        "signals": n,
        "confidence": round(p_hat, 4),
        "breakeven_confidence": round(p_be, 4),
        "edge_vs_breakeven": round(p_hat - p_be, 4),
        "exp_r": round(float(exp_r), 4),
        "exp_r_p05": round(float(np.quantile(exp_boot, 0.05)), 4),
        "exp_r_p95": round(float(np.quantile(exp_boot, 0.95)), 4),
        "prob_positive_exp": round(float(np.mean(exp_boot > 0.0)), 4),
    }


def _print_focus_table(df: pl.DataFrame) -> None:
    # Show strongest rows by probability of positive expectancy.
    top = (
        df.filter(pl.col("signals") >= 25)
        .sort(["prob_positive_exp", "edge_vs_breakeven"], descending=[True, True])
        .head(15)
    )

    table = Table(title="Top Conditions (min 25 signals)", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Dir")
    table.add_column("R:R", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("BE Conf", justify="right")
    table.add_column("P(E>0)", justify="right")
    table.add_column("Exp(R)", justify="right")

    for r in top.iter_rows(named=True):
        table.add_row(
            str(r["ticker"]),
            str(r["strategy"]),
            str(r["direction"]),
            f"{float(r['ratio']):.2f}",
            str(r["signals"]),
            f"{float(r['confidence']):.2%}",
            f"{float(r['breakeven_confidence']):.2%}",
            f"{float(r['prob_positive_exp']):.1%}",
            f"{float(r['exp_r']):+.3f}",
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
            z_score_threshold=2.0,
            z_score_window=240,
        ),
        KinematicLadderStrategy(
            regime_window=30,
            accel_window=10,
            volume_multiplier=1.05,
            volume_ma_period=settings.volume_ma_period,
            use_time_filter=True,
        ),
    ]

    console.rule("[bold cyan]Measurement Sensitivity + Monte Carlo[/]")
    console.print(
        f"Tickers: {args.tickers} | Range: {args.start} -> {args.end} | "
        f"Ratios: {ratios} | Bootstrap iters: {args.bootstrap_iters} | cost_r={args.cost_r}"
    )

    rows: list[dict] = []

    for ticker in args.tickers:
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            continue
        df = physics.enrich(df)

        for strategy in strategies:
            df_sig = strategy.generate_signals(df.clone())
            if df_sig.filter(pl.col("signal")).is_empty():
                continue

            df_eval = metrics.add_directional_forward_metrics(df_sig, snapshot_windows=(30, 60))
            base = (
                df_eval.filter(pl.col("signal"))
                .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"])
                .select(["signal_direction", "forward_mfe_eod", "forward_mae_eod"])
            )
            if base.is_empty():
                continue

            for direction in ("combined", "long", "short"):
                if direction == "combined":
                    subset = base
                else:
                    subset = base.filter(pl.col("signal_direction") == direction)

                if subset.is_empty():
                    continue

                mfe = subset["forward_mfe_eod"].to_numpy()
                mae = subset["forward_mae_eod"].to_numpy()

                for ratio in ratios:
                    stats = _evaluate_group(
                        mfe=mfe,
                        mae=mae,
                        ratio=ratio,
                        n_boot=args.bootstrap_iters,
                        cost_r=args.cost_r,
                        rng=rng,
                    )
                    rows.append({
                        "ticker": ticker,
                        "strategy": strategy.name,
                        "direction": direction,
                        "ratio": ratio,
                        **stats,
                    })

    if not rows:
        console.print("[red]No measurement rows generated.[/]")
        return

    out_df = pl.DataFrame(rows)
    _print_focus_table(out_df)

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = out_dir / f"measurement_sensitivity_{ts}.csv"
    out_df.write_csv(csv_path)

    console.print(f"\nSaved measurement sensitivity -> [green]{csv_path}[/]")


if __name__ == "__main__":
    main()
