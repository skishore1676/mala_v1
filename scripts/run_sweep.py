#!/usr/bin/env python3
"""
Parameter Sweep – Opening Bell Edition

Run 12 experiments with varying strategy configurations, all restricted
to the first 35 minutes of the trading day (9:30–10:04 ET).

Usage:
    python scripts/run_sweep.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.reporting import ExperimentReporter
from src.config import DATA_DIR
from src.time_utils import et_time_expr

console = Console()


# ── Experiment Configuration ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """One parameter combination to test."""
    name: str
    description: str
    # Physics
    ema_periods: List[int] = field(default_factory=lambda: [4, 8, 12])
    vpoc_lookback: int = 240
    volume_ma_period: int = 20
    # Strategy filters
    require_positive_velocity: bool = True
    require_positive_acceleration: bool = False
    require_positive_jerk: bool = False
    require_ema_stack: bool = True
    require_price_above_vpoc: bool = True
    vpoc_distance_pct: float = 0.0  # min % distance from VPOC (0 = just above)
    volume_above_ma: bool = True
    volume_multiplier: float = 1.0  # volume > X * volume_ma
    # Oracle
    forward_window: int = 15
    win_ratio: float = 2.0  # MFE > X * MAE = win


# ── Define the 12 experiments ────────────────────────────────────────────────

EXPERIMENTS: List[ExperimentConfig] = [
    # ── Baseline ─────────────────────────────────────────────────────────
    ExperimentConfig(
        name="01_Baseline",
        description="Opening bell: EMA 4/8/12 stack, 15-min window, 2:1 win",
    ),

    # ── Physics filters ──────────────────────────────────────────────────
    ExperimentConfig(
        name="02_Accel_Filter",
        description="Opening bell + acceleration > 0 (momentum building)",
        require_positive_acceleration=True,
    ),
    ExperimentConfig(
        name="03_Accel_Jerk_Filter",
        description="Opening bell + accel > 0 AND jerk > 0 (early move)",
        require_positive_acceleration=True,
        require_positive_jerk=True,
    ),

    # ── VPOC distance ────────────────────────────────────────────────────
    ExperimentConfig(
        name="04_VPOC_Escape_01pct",
        description="Opening bell + >0.1% above VPOC (escape velocity)",
        vpoc_distance_pct=0.001,
    ),
    ExperimentConfig(
        name="05_VPOC_Escape_Accel",
        description="Opening bell + 0.1% VPOC + accel > 0",
        vpoc_distance_pct=0.001,
        require_positive_acceleration=True,
    ),

    # ── Volume filters ───────────────────────────────────────────────────
    ExperimentConfig(
        name="06_High_Volume_1_5x",
        description="Opening bell + volume 1.5x MA (strong conviction)",
        volume_multiplier=1.5,
    ),
    ExperimentConfig(
        name="07_High_Volume_2x_Accel",
        description="Opening bell + 2x volume + accel > 0 (high force)",
        volume_multiplier=2.0,
        require_positive_acceleration=True,
    ),

    # ── Forward window variations ────────────────────────────────────────
    ExperimentConfig(
        name="08_30min_Window",
        description="Opening bell + 30-min fwd window (inertia)",
        forward_window=30,
        require_positive_acceleration=True,
    ),
    ExperimentConfig(
        name="09_60min_Window",
        description="Opening bell + 60-min fwd window (hour test)",
        forward_window=60,
        require_positive_acceleration=True,
    ),

    # ── Win ratio relaxation ─────────────────────────────────────────────
    ExperimentConfig(
        name="10_Relaxed_Win_1_5x",
        description="Opening bell + relaxed win: MFE > 1.5x MAE",
        win_ratio=1.5,
        require_positive_acceleration=True,
    ),

    # ── Kitchen sink ─────────────────────────────────────────────────────
    ExperimentConfig(
        name="11_Full_Physics",
        description="Opening bell + all physics + 0.1% VPOC + 1.5x vol + 30min",
        require_positive_acceleration=True,
        require_positive_jerk=True,
        vpoc_distance_pct=0.001,
        volume_multiplier=1.5,
        forward_window=30,
    ),
    ExperimentConfig(
        name="12_Full_Physics_Relaxed",
        description="Opening bell + kitchen sink + relaxed 1.5:1 win",
        require_positive_acceleration=True,
        require_positive_jerk=True,
        vpoc_distance_pct=0.001,
        volume_multiplier=1.5,
        forward_window=30,
        win_ratio=1.5,
    ),
]


# ── Custom Signal Generator ──────────────────────────────────────────────────

# Opening bell window: 9:30 AM – 10:04 AM ET
# Stored timestamps are UTC; convert to ET before applying opening-window filters.
OPENING_START = time(9, 30)
OPENING_END = time(10, 5)


def generate_signals(df: pl.DataFrame, cfg: ExperimentConfig) -> pl.DataFrame:
    """Apply configurable logic gates based on experiment config.

    All signals are restricted to the opening bell window (9:30–10:04 ET).
    """
    gates: List[pl.Expr] = []

    # ── Time-of-day gate: first 35 minutes of market open ────────────────
    gates.append(
        (et_time_expr("timestamp") >= OPENING_START)
        & (et_time_expr("timestamp") < OPENING_END)
    )

    ema_cols = [f"ema_{p}" for p in sorted(cfg.ema_periods)]

    # Gate: EMA stack alignment
    if cfg.require_ema_stack:
        for i in range(len(ema_cols) - 1):
            gates.append(pl.col(ema_cols[i]) > pl.col(ema_cols[i + 1]))

    # Gate: Price above VPOC
    if cfg.require_price_above_vpoc:
        if cfg.vpoc_distance_pct > 0:
            gates.append(
                (pl.col("close") - pl.col("vpoc_4h")) / pl.col("vpoc_4h")
                > cfg.vpoc_distance_pct
            )
        else:
            gates.append(pl.col("close") > pl.col("vpoc_4h"))

    # Gate: Volume above MA
    vol_ma_col = f"volume_ma_{cfg.volume_ma_period}"
    if cfg.volume_above_ma:
        gates.append(
            pl.col("volume") > cfg.volume_multiplier * pl.col(vol_ma_col)
        )

    # Gate: Positive velocity
    if cfg.require_positive_velocity:
        gates.append(pl.col("velocity_1m") > 0)

    # Gate: Positive acceleration
    if cfg.require_positive_acceleration:
        gates.append(pl.col("accel_1m") > 0)

    # Gate: Positive jerk
    if cfg.require_positive_jerk:
        gates.append(pl.col("jerk_1m") > 0)

    # Combine all gates with AND
    combined = gates[0]
    for g in gates[1:]:
        combined = combined & g

    return df.with_columns(combined.alias("signal"))


# ── Custom Win Calculation ───────────────────────────────────────────────────

def add_custom_forward_metrics(
    df: pl.DataFrame,
    forward_window: int,
    win_ratio: float,
) -> pl.DataFrame:
    """Add forward MFE/MAE with custom win ratio."""
    import numpy as np

    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    for i in range(n - forward_window):
        future_high = high[i + 1: i + 1 + forward_window]
        future_low = low[i + 1: i + 1 + forward_window]
        entry = close[i]
        mfe[i] = float(np.max(future_high) - entry)
        mae[i] = float(entry - np.min(future_low))

    mfe_col = f"forward_mfe_{forward_window}"
    mae_col = f"forward_mae_{forward_window}"

    df = df.with_columns([
        pl.Series(mfe_col, mfe),
        pl.Series(mae_col, mae),
    ])
    df = df.with_columns(
        (pl.col(mfe_col) > win_ratio * pl.col(mae_col)).alias("win")
    )
    return df


# ── Run a Single Experiment ──────────────────────────────────────────────────

def run_experiment(
    df_enriched: pl.DataFrame,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """Run one experiment and return results dict."""
    # Generate signals
    df = generate_signals(df_enriched.clone(), cfg)
    signal_count = df.filter(pl.col("signal")).height

    if signal_count == 0:
        return {
            "name": cfg.name,
            "description": cfg.description,
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "confidence": 0.0,
            "avg_mfe": None,
            "avg_mae": None,
            "median_mfe": None,
            "median_mae": None,
            "signal_rate_pct": 0.0,
        }

    # Add forward metrics with custom window and win ratio
    df = add_custom_forward_metrics(df, cfg.forward_window, cfg.win_ratio)

    mfe_col = f"forward_mfe_{cfg.forward_window}"
    mae_col = f"forward_mae_{cfg.forward_window}"

    signals = df.filter(pl.col("signal")).drop_nulls(subset=[mfe_col, mae_col])
    total = signals.height
    wins = signals.filter(pl.col("win")).height if total > 0 else 0
    confidence = wins / total if total > 0 else 0.0

    import math

    def safe_float(val):
        if val is None:
            return None
        v = float(val)
        return round(v, 4) if not math.isnan(v) else None

    return {
        "name": cfg.name,
        "description": cfg.description,
        "total_signals": total,
        "wins": wins,
        "losses": total - wins,
        "confidence": round(confidence, 4),
        "avg_mfe": safe_float(signals[mfe_col].mean()),
        "avg_mae": safe_float(signals[mae_col].mean()),
        "median_mfe": safe_float(signals[mfe_col].median()),
        "median_mae": safe_float(signals[mae_col].median()),
        "signal_rate_pct": round(signal_count / len(df_enriched) * 100, 2),
        "forward_window": cfg.forward_window,
        "win_ratio": cfg.win_ratio,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    ticker = "SPY"
    start = date(2025, 2, 12)
    end = date(2026, 2, 12)

    console.print("[bold green]Opening Bell Sweep[/] – Loading cached data …")

    # Load cached data
    storage = LocalStorage()
    df = storage.load_bars(ticker, start, end)
    if df.is_empty():
        console.print("[red]No cached data found. Run main.py first.[/]")
        return

    console.print(f"  Loaded [cyan]{len(df):,}[/] bars for {ticker}")

    # Enrich once — all experiments share the same physics
    physics = PhysicsEngine()
    df_enriched = physics.enrich(df)
    console.print(f"  Enriched with [cyan]{len(df_enriched.columns)}[/] columns\n")

    # Run all experiments
    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(EXPERIMENTS, 1):
        console.print(
            f"  [{i:2d}/{len(EXPERIMENTS)}] {cfg.name}: {cfg.description}"
        )
        result = run_experiment(df_enriched, cfg)
        results.append(result)
        conf_str = f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
        console.print(
            f"         → Signals: {result['total_signals']:,}  "
            f"Confidence: {conf_str}"
        )

    # Print comparison table
    console.print()
    table = Table(
        title=f"Parameter Sweep Results – {ticker} ({start} → {end})",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Experiment", style="bold", max_width=30)
    table.add_column("Signals", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Med MFE", justify="right")
    table.add_column("Med MAE", justify="right")
    table.add_column("MFE/MAE", justify="right", style="cyan")
    table.add_column("Sig Rate", justify="right", style="dim")

    for i, r in enumerate(results, 1):
        med_mfe = r.get("median_mfe")
        med_mae = r.get("median_mae")
        ratio = ""
        if med_mfe and med_mae and med_mae > 0:
            ratio = f"{med_mfe / med_mae:.2f}x"

        table.add_row(
            str(i),
            r["name"],
            f"{r['total_signals']:,}",
            str(r["wins"]),
            f"{r['confidence']:.2%}" if r["confidence"] else "N/A",
            f"${med_mfe:.4f}" if med_mfe else "N/A",
            f"${med_mae:.4f}" if med_mae else "N/A",
            ratio,
            f"{r.get('signal_rate_pct', 0):.1f}%",
        )

    console.print(table)

    # Save results to JSON
    output_path = DATA_DIR / "results" / "opening_bell_sweep_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "ticker": ticker,
            "date_range": {"start": str(start), "end": str(end)},
            "total_bars": len(df_enriched),
            "experiments": results,
            "configs": [asdict(c) for c in EXPERIMENTS],
        }, f, indent=2, default=str)

    console.print(f"\n  Results saved → [green]{output_path}[/]")
    console.print("[bold green]✓ Sweep complete.[/]")


if __name__ == "__main__":
    main()
