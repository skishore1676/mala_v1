#!/usr/bin/env python3
"""Run a first-pass multi-symbol research cycle for Market Impulse."""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.research.loop_export import LoopArtifactExporter
from src.research.models import ResearchStage
from src.research.reporting import ResearchJournal
from src.research.run_storage import append_strategy_index, create_run_dir
from src.research.stages import (
    aggregate_walk_forward,
    build_gate_report,
    build_windows,
    promoted_candidates_from_gate_report,
    promoted_candidates_from_holdout,
    run_execution_mapping_for_candidates,
    run_holdout_validation_for_candidates,
    run_walk_forward_for_strategies,
    summarize_holdout,
)
from src.research.stages.candidates import build_candidate_strategy
from src.research.tools import ResearchToolResult
from src.strategy.base import BaseStrategy, required_feature_union
from src.strategy.market_impulse import MarketImpulseStrategy


DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA"]


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Could not parse floats from: {value}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--calibration-end", type=date.fromisoformat, default=date(2025, 11, 30))
    parser.add_argument("--holdout-start", type=date.fromisoformat, default=date(2025, 12, 1))
    parser.add_argument("--holdout-end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--ratios", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--m1-cost-bps", type=float, default=8.0)
    parser.add_argument("--cost-grid-bps", default="5,8,12")
    parser.add_argument("--min-signals", type=int, default=10)
    parser.add_argument("--gate-min-oos-windows", type=int, default=6)
    parser.add_argument("--gate-min-oos-signals", type=int, default=100)
    parser.add_argument("--gate-min-pct-positive", type=float, default=0.55)
    parser.add_argument("--gate-min-exp-r", type=float, default=0.0)
    parser.add_argument("--min-calibration-signals", type=int, default=40)
    parser.add_argument("--min-holdout-signals", type=int, default=15)
    parser.add_argument("--base-cost-r", type=float, default=0.08)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--top-per-ticker", type=int, default=1)
    parser.add_argument("--out-dir", default="data/results/agentic_runs")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "null"
    return f"{float(value):+.{digits}f}"


def market_impulse_configs() -> list[dict[str, object]]:
    return [
        {
            "entry_buffer_minutes": entry_buffer_minutes,
            "entry_window_minutes": entry_window_minutes,
            "regime_timeframe": regime_timeframe,
        }
        for entry_buffer_minutes, entry_window_minutes, regime_timeframe in product(
            [3, 5],
            [45, 60, 90],
            ["5m", "15m", "30m", "1h"],
        )
    ]


def build_market_impulse(config: dict[str, object]) -> BaseStrategy:
    return MarketImpulseStrategy(
        entry_buffer_minutes=int(config["entry_buffer_minutes"]),
        entry_window_minutes=int(config["entry_window_minutes"]),
        regime_timeframe=str(config["regime_timeframe"]),
    )


def load_frames(*, tickers: list[str], start: date, end: date, configs: list[dict[str, object]]) -> dict[str, pl.DataFrame]:
    storage = LocalStorage()
    physics = PhysicsEngine()
    strategies = [build_market_impulse(config) for config in configs]
    needed_features = required_feature_union(strategies)

    frames: dict[str, pl.DataFrame] = {}
    for ticker in tickers:
        raw = storage.load_bars(ticker, start, end)
        if raw.is_empty():
            log(f"SKIP_NO_DATA {ticker}")
            continue
        frames[ticker] = physics.enrich_for_features(raw, needed_features)
        log(f"LOADED {ticker} rows={frames[ticker].height}")
    return frames


def run_m1(*, frames: dict[str, pl.DataFrame], windows, ratios: list[float], metrics: MetricsCalculator, min_signals: int, m1_cost_bps: float, top_per_ticker: int, configs: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    for idx, config in enumerate(configs, start=1):
        strategy = build_market_impulse(config)
        log(
            f"M1_CONFIG {idx}/{len(configs)} buffer={config['entry_buffer_minutes']} "
            f"window={config['entry_window_minutes']} tf={config['regime_timeframe']}"
        )
        for ticker, frame in frames.items():
            rows = run_walk_forward_for_strategies(
                ticker=ticker,
                df=frame,
                strategies=[strategy],
                windows=windows,
                ratios=ratios,
                metrics=metrics,
                min_signals=min_signals,
                cost_bps=m1_cost_bps,
            )
            if not rows:
                continue
            for row in rows:
                detail_rows.append({**row, **config})
            aggregate = aggregate_walk_forward(rows)
            for row in aggregate.iter_rows(named=True):
                aggregate_rows.append({**row, **config})

    detail_df = pl.DataFrame(detail_rows) if detail_rows else pl.DataFrame()
    aggregate_df = pl.DataFrame(aggregate_rows) if aggregate_rows else pl.DataFrame()
    if aggregate_df.is_empty():
        raise RuntimeError("M1 produced no aggregate rows")

    ranked_df = (
        aggregate_df
        .filter(pl.col("direction").is_in(["long", "short"]))
        .filter(pl.col("avg_test_exp_r").is_not_null())
        .filter(pl.col("pct_positive_oos_windows").is_not_null())
        .filter(pl.col("avg_test_exp_r") > 0)
        .with_columns([
            (
                pl.col("avg_test_exp_r") * 1000
                + pl.col("pct_positive_oos_windows") * 100
                + pl.col("oos_signals") / 1000
            ).alias("m1_score")
        ])
        .sort(
            ["ticker", "oos_windows", "m1_score", "avg_test_exp_r", "oos_signals"],
            descending=[False, True, True, True, True],
        )
    )

    top_rows: list[dict[str, object]] = []
    for ticker in frames:
        subset = ranked_df.filter(pl.col("ticker") == ticker)
        if subset.is_empty():
            continue
        top_rows.extend(subset.head(top_per_ticker).iter_rows(named=True))

    top_df = pl.DataFrame(top_rows) if top_rows else pl.DataFrame()
    if top_df.is_empty():
        raise RuntimeError("No top candidates selected from M1")
    return detail_df, aggregate_df, top_df


def run_m2(*, frames: dict[str, pl.DataFrame], windows, ratios: list[float], metrics: MetricsCalculator, min_signals: int, cost_grid_bps: list[float], top_candidates: pl.DataFrame, gate_min_oos_windows: int, gate_min_oos_signals: int, gate_min_pct_positive: float, gate_min_exp_r: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    convergence_frames: list[pl.DataFrame] = []

    for cost_bps in cost_grid_bps:
        rows: list[dict[str, object]] = []
        log(f"M2_COST_BPS {cost_bps}")
        for candidate in top_candidates.iter_rows(named=True):
            strategy = build_market_impulse(candidate)
            ticker = str(candidate["ticker"])
            walk_forward_rows = run_walk_forward_for_strategies(
                ticker=ticker,
                df=frames[ticker],
                strategies=[strategy],
                windows=windows,
                ratios=ratios,
                metrics=metrics,
                min_signals=min_signals,
                cost_bps=cost_bps,
            )
            if not walk_forward_rows:
                continue
            aggregate = aggregate_walk_forward(walk_forward_rows)
            aggregate = aggregate.filter(pl.col("direction") == candidate["direction"])
            if aggregate.is_empty():
                continue
            for row in aggregate.iter_rows(named=True):
                rows.append(
                    {
                        **row,
                        "cost_bps": cost_bps,
                        "entry_buffer_minutes": int(candidate["entry_buffer_minutes"]),
                        "entry_window_minutes": int(candidate["entry_window_minutes"]),
                        "regime_timeframe": str(candidate["regime_timeframe"]),
                    }
                )
        if rows:
            convergence_frames.append(pl.DataFrame(rows))

    if not convergence_frames:
        raise RuntimeError("M2 produced no convergence rows")

    combined_df = pl.concat(convergence_frames, how="vertical")
    gate_report = build_gate_report(
        combined=combined_df,
        cost_count=len(cost_grid_bps),
        gate_min_oos_windows=gate_min_oos_windows,
        gate_min_oos_signals=gate_min_oos_signals,
        gate_min_pct_positive=gate_min_pct_positive,
        gate_min_exp_r=gate_min_exp_r,
    )
    promoted_df = promoted_candidates_from_gate_report(gate_report)
    return combined_df, gate_report, promoted_df


def run_m3(*, frames: dict[str, pl.DataFrame], windows, ratios: list[float], metrics: MetricsCalculator, min_signals: int, m1_cost_bps: float, promoted_candidates: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in promoted_candidates.iter_rows(named=True):
        strategy = build_candidate_strategy(candidate)
        ticker = str(candidate["ticker"])
        walk_forward_rows = run_walk_forward_for_strategies(
            ticker=ticker,
            df=frames[ticker],
            strategies=[strategy],
            windows=windows,
            ratios=ratios,
            metrics=metrics,
            min_signals=min_signals,
            cost_bps=m1_cost_bps,
        )
        for row in walk_forward_rows:
            if row["direction"] != candidate["direction"]:
                continue
            rows.append(
                {
                    **row,
                    "entry_buffer_minutes": candidate["entry_buffer_minutes"],
                    "entry_window_minutes": candidate["entry_window_minutes"],
                    "regime_timeframe": candidate["regime_timeframe"],
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def run_m4_m5(*, frames: dict[str, pl.DataFrame], metrics: MetricsCalculator, promoted_candidates: pl.DataFrame, start: date, calibration_end: date, holdout_start: date, holdout_end: date, ratios: list[float], cost_grid_bps: list[float], min_calibration_signals: int, min_holdout_signals: int, base_cost_r: float, bootstrap_iters: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    holdout_rows = run_holdout_validation_for_candidates(
        promoted=promoted_candidates,
        ticker_frames=frames,
        metrics=metrics,
        start_date=start,
        calibration_end=calibration_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        ratios=ratios,
        costs=cost_grid_bps,
        min_calibration_signals=min_calibration_signals,
        min_holdout_signals=min_holdout_signals,
    )
    holdout_detail = pl.DataFrame(holdout_rows) if holdout_rows else pl.DataFrame()
    holdout_summary = summarize_holdout(holdout_detail, cost_count=len(cost_grid_bps)) if not holdout_detail.is_empty() else pl.DataFrame()
    execution_candidates = promoted_candidates_from_holdout(holdout_summary) if not holdout_summary.is_empty() else pl.DataFrame()
    execution_rows = run_execution_mapping_for_candidates(
        promoted=execution_candidates,
        holdout_detail=holdout_detail,
        ticker_frames=frames,
        metrics=metrics,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        base_cost_r=base_cost_r,
        stress_cfg=ExecutionStressConfig(bootstrap_iters=bootstrap_iters),
    ) if not execution_candidates.is_empty() else []
    execution_df = pl.DataFrame(execution_rows) if execution_rows else pl.DataFrame()
    return holdout_detail, holdout_summary, execution_df


def write_outputs(*, out_dir: Path, m1_detail: pl.DataFrame, m1_aggregate: pl.DataFrame, m1_top: pl.DataFrame, m2_combined: pl.DataFrame, m2_gate_report: pl.DataFrame, m3_detail: pl.DataFrame, m4_detail: pl.DataFrame, m4_summary: pl.DataFrame, m5_execution: pl.DataFrame) -> None:
    m1_detail.write_csv(out_dir / "m1_detail.csv")
    m1_aggregate.write_csv(out_dir / "m1_aggregate.csv")
    m1_top.write_csv(out_dir / "m1_top_candidates.csv")
    m2_combined.write_csv(out_dir / "m2_convergence_combined.csv")
    m2_gate_report.write_csv(out_dir / "m2_gate_report.csv")
    if not m3_detail.is_empty():
        m3_detail.write_csv(out_dir / "m3_walk_forward_detail.csv")
    if not m4_detail.is_empty():
        m4_detail.write_csv(out_dir / "m4_holdout_detail.csv")
    if not m4_summary.is_empty():
        m4_summary.write_csv(out_dir / "m4_holdout_summary.csv")
    if not m5_execution.is_empty():
        m5_execution.write_csv(out_dir / "m5_execution_mapping.csv")


def write_summary(*, out_dir: Path, m1_top: pl.DataFrame, m2_promoted: pl.DataFrame, m3_detail: pl.DataFrame, m4_summary: pl.DataFrame, m5_execution: pl.DataFrame) -> None:
    m4_promoted = promoted_candidates_from_holdout(m4_summary) if not m4_summary.is_empty() else pl.DataFrame()
    lines = [
        f"OUT_DIR={out_dir}",
        f"M1_TOP={m1_top.height}",
        f"M2_PROMOTED={m2_promoted.height}",
        f"M3_ROWS={m3_detail.height}",
        f"M4_PROMOTED={m4_promoted.height}",
        f"M5_ROWS={m5_execution.height}",
        "TOP_CANDIDATES:",
    ]
    for row in m1_top.iter_rows(named=True):
        lines.append(
            "|".join(
                [
                    str(row["ticker"]),
                    str(row["direction"]),
                    f"buffer={int(row['entry_buffer_minutes'])}",
                    f"window={int(row['entry_window_minutes'])}",
                    f"tf={row['regime_timeframe']}",
                    f"exp={fmt_float(row['avg_test_exp_r'])}",
                    f"pct={fmt_float(row['pct_positive_oos_windows'], digits=3)}",
                    f"n={int(row['oos_signals'])}",
                ]
            )
        )
    lines.append("PROMOTED_CANDIDATES:")
    for row in m2_promoted.iter_rows(named=True):
        lines.append(
            "|".join(
                [
                    str(row["ticker"]),
                    str(row["direction"]),
                    f"buffer={int(row['entry_buffer_minutes'])}",
                    f"window={int(row['entry_window_minutes'])}",
                    f"tf={row['regime_timeframe']}",
                ]
            )
        )
    lines.append("HOLDOUT_PROMOTED:")
    for row in m4_promoted.iter_rows(named=True):
        lines.append(
            "|".join(
                [
                    str(row["ticker"]),
                    str(row["direction"]),
                    f"buffer={int(row['entry_buffer_minutes'])}",
                    f"window={int(row['entry_window_minutes'])}",
                    f"tf={row['regime_timeframe']}",
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    args = parse_args()
    ratios = parse_csv_floats(args.ratios)
    cost_grid_bps = parse_csv_floats(args.cost_grid_bps)
    out_dir = create_run_dir(args.out_dir, "market_impulse")
    journal = ResearchJournal(out_dir)
    metrics = MetricsCalculator()
    windows = build_windows(args.start, args.end, args.train_months, args.test_months)
    configs = market_impulse_configs()
    frames = load_frames(tickers=args.tickers, start=args.start, end=args.end, configs=configs)

    m1_detail, m1_aggregate, m1_top = run_m1(
        frames=frames,
        windows=windows,
        ratios=ratios,
        metrics=metrics,
        min_signals=args.min_signals,
        m1_cost_bps=args.m1_cost_bps,
        top_per_ticker=args.top_per_ticker,
        configs=configs,
    )
    journal.record_stage(
        stage=ResearchStage.M1_DISCOVERY,
        result=ResearchToolResult(
            tool_name="parameter_sweep",
            summary={"ticker_count": len(frames), "aggregate_rows": m1_aggregate.height, "top_candidates": m1_top.height},
            artifacts={"detail": m1_detail, "aggregate": m1_aggregate, "top_candidates": m1_top},
        ),
        decision="promote" if not m1_top.is_empty() else "kill",
        rationale="Completed bounded discovery sweep across Market Impulse regime timeframes and selected the strongest per-ticker directional candidates.",
        next_action="Run convergence grid on the M1 shortlist.",
        context={"strategy_family": "Market Impulse (Cross & Reclaim)"},
    )

    m2_combined, m2_gate_report, m2_promoted = run_m2(
        frames=frames,
        windows=windows,
        ratios=ratios,
        metrics=metrics,
        min_signals=args.min_signals,
        cost_grid_bps=cost_grid_bps,
        top_candidates=m1_top,
        gate_min_oos_windows=args.gate_min_oos_windows,
        gate_min_oos_signals=args.gate_min_oos_signals,
        gate_min_pct_positive=args.gate_min_pct_positive,
        gate_min_exp_r=args.gate_min_exp_r,
    )
    journal.record_stage(
        stage=ResearchStage.M2_CONVERGENCE,
        result=ResearchToolResult(
            tool_name="convergence_grid",
            summary={"candidate_count": m2_gate_report.height, "promoted_count": m2_promoted.height},
            artifacts={"combined": m2_combined, "gate_report": m2_gate_report, "promoted": m2_promoted},
        ),
        decision="promote" if not m2_promoted.is_empty() else "retune",
        rationale="Applied deterministic robustness gates across the friction grid for the Market Impulse shortlist.",
        next_action="Run walk-forward review on M2 survivors." if not m2_promoted.is_empty() else "Retune or replace weak candidates before holdout.",
        context={
            "gate_min_oos_windows": args.gate_min_oos_windows,
            "gate_min_oos_signals": args.gate_min_oos_signals,
            "gate_min_pct_positive": args.gate_min_pct_positive,
            "gate_min_exp_r": args.gate_min_exp_r,
        },
    )

    m3_detail = run_m3(
        frames=frames,
        windows=windows,
        ratios=ratios,
        metrics=metrics,
        min_signals=args.min_signals,
        m1_cost_bps=args.m1_cost_bps,
        promoted_candidates=m2_promoted,
    )
    journal.record_stage(
        stage=ResearchStage.M3_WALK_FORWARD,
        result=ResearchToolResult(
            tool_name="walk_forward",
            summary={"detail_rows": m3_detail.height, "survivor_count": m2_promoted.height},
            artifacts={"detail": m3_detail},
        ),
        decision="promote" if not m3_detail.is_empty() else "gather_more_evidence",
        rationale="Reviewed window-by-window OOS behavior for each convergence survivor before holdout.",
        next_action="Run untouched holdout validation." if not m3_detail.is_empty() else "Gather more walk-forward evidence.",
        context={"m1_cost_bps": args.m1_cost_bps},
    )

    m4_detail, m4_summary, m5_execution = run_m4_m5(
        frames=frames,
        metrics=metrics,
        promoted_candidates=m2_promoted,
        start=args.start,
        calibration_end=args.calibration_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        ratios=ratios,
        cost_grid_bps=cost_grid_bps,
        min_calibration_signals=args.min_calibration_signals,
        min_holdout_signals=args.min_holdout_signals,
        base_cost_r=args.base_cost_r,
        bootstrap_iters=args.bootstrap_iters,
    )
    m4_promoted = promoted_candidates_from_holdout(m4_summary) if not m4_summary.is_empty() else pl.DataFrame()
    journal.record_stage(
        stage=ResearchStage.M4_HOLDOUT,
        result=ResearchToolResult(
            tool_name="holdout_validation",
            summary={"detail_rows": m4_detail.height, "promoted_count": m4_promoted.height},
            artifacts={"detail": m4_detail, "summary": m4_summary, "promoted": m4_promoted},
        ),
        decision="promote" if not m4_promoted.is_empty() else "retune",
        rationale="Evaluated the untouched holdout segment using calibration-selected ratios and friction stress.",
        next_action="Run execution mapping on holdout survivors." if not m4_promoted.is_empty() else "Retune or reject candidates that failed holdout.",
        context={"min_calibration_signals": args.min_calibration_signals, "min_holdout_signals": args.min_holdout_signals},
    )

    max_mc_prob = float(m5_execution.get_column("mc_prob_positive_exp").max()) if not m5_execution.is_empty() else None
    m5_decision = "promote" if max_mc_prob is not None and max_mc_prob >= 0.55 else "gather_more_evidence" if not m5_execution.is_empty() else "kill"
    journal.record_stage(
        stage=ResearchStage.M5_EXECUTION,
        result=ResearchToolResult(
            tool_name="execution_mapping",
            summary={"mapped_count": m5_execution.height, "max_mc_prob_positive_exp": max_mc_prob},
            artifacts={"detail": m5_execution},
        ),
        decision=m5_decision,
        rationale="Mapped holdout survivors into practical option structures and stress-tested execution robustness.",
        next_action="Candidate is execution-robust enough for the next promotion step." if m5_decision == "promote" else "Collect more execution evidence or retune before any live trial.",
        context={"base_cost_r": args.base_cost_r, "bootstrap_iters": args.bootstrap_iters},
    )

    write_outputs(
        out_dir=out_dir,
        m1_detail=m1_detail,
        m1_aggregate=m1_aggregate,
        m1_top=m1_top,
        m2_combined=m2_combined,
        m2_gate_report=m2_gate_report,
        m3_detail=m3_detail,
        m4_detail=m4_detail,
        m4_summary=m4_summary,
        m5_execution=m5_execution,
    )
    write_summary(
        out_dir=out_dir,
        m1_top=m1_top,
        m2_promoted=m2_promoted,
        m3_detail=m3_detail,
        m4_summary=m4_summary,
        m5_execution=m5_execution,
    )
    append_strategy_index(
        out_dir,
        strategy_label="Market Impulse (Cross & Reclaim)",
        headline=(
            f"M2={m2_promoted.height}, "
            f"M4={m4_promoted.height}, "
            f"M5={m5_execution.height}"
        ),
    )
    LoopArtifactExporter().export_runs([out_dir], out_dir=out_dir)


if __name__ == "__main__":
    main()
