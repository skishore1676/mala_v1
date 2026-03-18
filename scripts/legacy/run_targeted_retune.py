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

import polars as pl
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.research import ResearchOrchestrator
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


def main() -> None:
    args = parse_args()
    orchestrator = ResearchOrchestrator()

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

    candidates: list[dict] = []

    console.rule("[bold cyan]Targeted Retune Sweep (ET-corrected)[/]")
    console.print(f"Tickers={args.tickers} range={args.start}->{args.end} ratio={args.ratio} cost_r={args.cost_r}")

    for ticker in args.tickers:
        console.print(
            f"Running {ticker}: {len(kinematic_grid)} kinematic + "
            f"{len(compression_grid)} compression configs"
        )
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
            candidates.append({
                "strategy": strat,
                "strategy_name": "Kinematic Ladder",
                "params_label": f"rw={regime_window},aw={accel_window},vol={vol_mult},sess={sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                "metadata": {
                    "candidate_family": "kinematic_ladder",
                    "regime_window": regime_window,
                    "accel_window": accel_window,
                    "volume_multiplier": vol_mult,
                    "session": f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                },
            })

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
            candidates.append({
                "strategy": strat,
                "strategy_name": "Compression Expansion Breakout",
                "params_label": f"cw={cw},bl={bl},cf={cf},vol={vol_mult},sess={sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                "metadata": {
                    "candidate_family": "compression_breakout",
                    "compression_window": cw,
                    "breakout_lookback": bl,
                    "compression_factor": cf,
                    "volume_multiplier": vol_mult,
                    "session": f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}",
                },
            })

    result = orchestrator.toolbox().retune_search(
        candidates=candidates,
        ratio=args.ratio,
        cost_r=args.cost_r,
        bootstrap_iters=args.bootstrap_iters,
        min_signals=args.min_signals,
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
    )
    out = result.artifacts["detail"]
    ranked = result.artifacts["ranked"]

    if out.is_empty():
        console.print("[red]No rows generated[/]")
        return

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
