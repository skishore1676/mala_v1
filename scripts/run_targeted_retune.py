#!/usr/bin/env python3
"""Targeted retuning sweep for ET-session strategies.

Focuses on:
- Kinematic Ladder
- Compression Expansion Breakout

Scores each param set using friction-adjusted expectancy robustness.
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
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
from src.strategy.kinematic_ladder import KinematicLadderStrategy
from src.strategy.compression_breakout import CompressionBreakoutStrategy


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted strategy retuning")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    p.add_argument("--ratio", type=float, default=1.5)
    p.add_argument("--cost-r", type=float, default=0.05)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--min-signals", type=int, default=50)
    return p.parse_args()


def _score_group(mfe: np.ndarray, mae: np.ndarray, ratio: float, cost_r: float, n_boot: int, rng: np.random.Generator) -> dict:
    n = len(mfe)
    if n == 0:
        return {"signals": 0, "confidence": None, "exp_r": None, "prob_pos_exp": None}

    wins = mfe >= ratio * mae
    p = float(np.mean(wins))
    exp_r = p * ratio - (1.0 - p) - cost_r

    p_boot = rng.binomial(n=n, p=p, size=n_boot) / n
    exp_boot = p_boot * ratio - (1.0 - p_boot) - cost_r

    return {
        "signals": n,
        "confidence": round(p, 4),
        "exp_r": round(exp_r, 4),
        "prob_pos_exp": round(float(np.mean(exp_boot > 0)), 4),
    }


def _evaluate(df_eval: pl.DataFrame, ratio: float, cost_r: float, n_boot: int, min_signals: int, rng: np.random.Generator) -> dict:
    base = (
        df_eval.filter(pl.col("signal"))
        .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"])
    )

    rows = []
    for direction in ("combined", "long", "short"):
        subset = base if direction == "combined" else base.filter(pl.col("signal_direction") == direction)
        mfe = subset["forward_mfe_eod"].to_numpy() if len(subset) else np.array([])
        mae = subset["forward_mae_eod"].to_numpy() if len(subset) else np.array([])
        stats = _score_group(mfe, mae, ratio, cost_r, n_boot, rng)
        if stats["signals"] >= min_signals:
            rows.append({"direction": direction, **stats})

    # objective: combined expectancy, then prob positive, then signal count
    combined = next((r for r in rows if r["direction"] == "combined"), None)
    if combined is None:
        return {"rows": rows, "objective": -1e9}

    objective = float(combined["exp_r"]) * 1000 + float(combined["prob_pos_exp"]) * 100 + float(combined["signals"]) / 1000
    return {"rows": rows, "objective": objective}


def main() -> None:
    args = parse_args()

    storage = LocalStorage()
    physics = PhysicsEngine()
    metrics = MetricsCalculator()
    rng = np.random.default_rng(7)

    # Param grids
    kinematic_grid = list(product(
        [20, 35],               # regime_window
        [8, 12],                # accel_window
        [1.00, 1.10],           # volume_multiplier
        [(9, 35, 15, 30), (9, 45, 15, 20)],
    ))

    compression_grid = list(product(
        [15, 25],               # compression_window
        [15, 25],               # breakout_lookback
        [0.80, 0.95],           # compression_factor
        [1.05, 1.20],           # volume_multiplier
        [(9, 40, 15, 30), (9, 50, 15, 20)],
    ))

    rows: list[dict] = []

    console.rule("[bold cyan]Targeted Retune Sweep (ET-corrected)[/]")
    console.print(f"Tickers={args.tickers} range={args.start}->{args.end} ratio={args.ratio} cost_r={args.cost_r}")

    for ticker in args.tickers:
        console.print(
            f"Running {ticker}: {len(kinematic_grid)} kinematic + "
            f"{len(compression_grid)} compression configs"
        )
        df = storage.load_bars(ticker, args.start, args.end)
        if df.is_empty():
            continue
        df = physics.enrich(df)

        # Kinematic
        for regime_window, accel_window, vol_mult, sess in kinematic_grid:
            sh, sm, eh, em = sess
            strat = KinematicLadderStrategy(
                regime_window=regime_window,
                accel_window=accel_window,
                volume_multiplier=vol_mult,
                use_time_filter=True,
                session_start=__import__('datetime').time(sh, sm),
                session_end=__import__('datetime').time(eh, em),
            )
            df_eval = metrics.add_directional_forward_metrics(strat.generate_signals(df.clone()), snapshot_windows=(30, 60))
            scored = _evaluate(df_eval, args.ratio, args.cost_r, args.bootstrap_iters, args.min_signals, rng)
            for r in scored["rows"]:
                rows.append({
                    "ticker": ticker,
                    "strategy": "Kinematic Ladder",
                    "params": f"rw={regime_window},aw={accel_window},vol={vol_mult},sess={sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                    "direction": r["direction"],
                    "signals": r["signals"],
                    "confidence": r["confidence"],
                    "exp_r": r["exp_r"],
                    "prob_pos_exp": r["prob_pos_exp"],
                    "objective": scored["objective"],
                })

        # Compression
        for cw, bl, cf, vol_mult, sess in compression_grid:
            sh, sm, eh, em = sess
            strat = CompressionBreakoutStrategy(
                compression_window=cw,
                breakout_lookback=bl,
                compression_factor=cf,
                volume_multiplier=vol_mult,
                use_time_filter=True,
                session_start=__import__('datetime').time(sh, sm),
                session_end=__import__('datetime').time(eh, em),
            )
            df_eval = metrics.add_directional_forward_metrics(strat.generate_signals(df.clone()), snapshot_windows=(30, 60))
            scored = _evaluate(df_eval, args.ratio, args.cost_r, args.bootstrap_iters, args.min_signals, rng)
            for r in scored["rows"]:
                rows.append({
                    "ticker": ticker,
                    "strategy": "Compression Expansion Breakout",
                    "params": f"cw={cw},bl={bl},cf={cf},vol={vol_mult},sess={sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                    "direction": r["direction"],
                    "signals": r["signals"],
                    "confidence": r["confidence"],
                    "exp_r": r["exp_r"],
                    "prob_pos_exp": r["prob_pos_exp"],
                    "objective": scored["objective"],
                })

    if not rows:
        console.print("[red]No rows generated[/]")
        return

    out = pl.DataFrame(rows)

    # ranking on combined direction only
    ranked = (
        out.filter(pl.col("direction") == "combined")
        .sort(["exp_r", "prob_pos_exp", "signals"], descending=[True, True, True])
    )

    table = Table(title="Top Retuned Configs (Combined)", show_lines=True)
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Exp(R)", justify="right")
    table.add_column("P(E>0)", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Params")

    for r in ranked.head(20).iter_rows(named=True):
        table.add_row(
            str(r["ticker"]),
            str(r["strategy"]),
            f"{float(r['exp_r']):+.3f}",
            f"{float(r['prob_pos_exp']):.1%}",
            str(r["signals"]),
            str(r["params"]),
        )

    console.print(table)

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = out_dir / f"retune_sweep_full_{stamp}.csv"
    top_path = out_dir / f"retune_sweep_top_{stamp}.csv"

    out.write_csv(full_path)
    ranked.head(100).write_csv(top_path)

    console.print(f"\nSaved full sweep -> [green]{full_path}[/]")
    console.print(f"Saved top configs -> [green]{top_path}[/]")


if __name__ == "__main__":
    main()
