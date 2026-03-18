"""Tests for reusable research stage helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.research.stages import (
    aggregate_walk_forward,
    build_gate_report,
    build_windows,
    choose_ratio,
    cost_r_from_bps,
    cost_tag,
    evaluate_df,
    eval_holdout_direction,
    latest_csv,
    median_selected_ratio,
    option_mapping_for,
    parse_costs,
    parse_floats,
    promoted_candidates_from_gate_report,
    promoted_candidates_from_holdout,
    run_execution_mapping_for_candidates,
    run_holdout_validation_for_candidates,
    run_walk_forward_for_strategies,
    summarize_holdout,
)


class _StubStrategy:
    name = "Stub Directional"
    evaluation_mode = "directional"
    required_features = {"timestamp", "close", "high", "low", "signal", "signal_direction"}
    parameter_space = {}

    def strategy_config(self) -> dict:
        return {}

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


def _sample_eval_df() -> pl.DataFrame:
    ts = [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(5)]
    return pl.DataFrame(
        {
            "timestamp": ts,
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 12.0, 11.0, 13.0, 12.0],
            "low": [10.0, 9.0, 8.0, 9.0, 9.0],
            "signal": [True, False, True, False, False],
            "signal_direction": ["long", None, "short", None, None],
            "forward_mfe_eod": [2.0, None, 2.0, None, None],
            "forward_mae_eod": [1.0, None, 1.0, None, None],
        }
    )


def test_build_windows_creates_expected_rolls() -> None:
    windows = build_windows(date(2025, 1, 1), date(2025, 12, 31), 6, 3)
    assert len(windows) == 2
    assert windows[0].train_start == date(2025, 1, 1)
    assert windows[0].test_start == date(2025, 7, 1)


def test_cost_helpers() -> None:
    assert parse_costs("0.05, 0.08") == [0.05, 0.08]
    assert parse_floats("1.0, 2.0") == [1.0, 2.0]
    assert cost_tag(0.05) == "cost050"
    assert cost_r_from_bps(8.0, avg_mae_dollars=1.0, avg_entry_price=100.0) == 0.08


def test_evaluate_df_combined() -> None:
    result = evaluate_df(_sample_eval_df(), "combined", ratio=1.5, cost_r=0.05)
    assert result["signals"] == 2
    assert result["confidence"] == 1.0
    assert result["exp_r"] == 1.45


def test_aggregate_walk_forward() -> None:
    agg = aggregate_walk_forward(
        [
            {
                "ticker": "SPY",
                "strategy": "Stub",
                "direction": "long",
                "window_idx": 1,
                "test_signals": 10,
                "test_confidence": 0.5,
                "test_exp_r": 0.1,
            },
            {
                "ticker": "SPY",
                "strategy": "Stub",
                "direction": "long",
                "window_idx": 2,
                "test_signals": 20,
                "test_confidence": 0.7,
                "test_exp_r": -0.1,
            },
        ]
    )
    row = agg.row(0, named=True)
    assert row["oos_windows"] == 2
    assert row["oos_signals"] == 30
    assert row["avg_test_exp_r"] == 0.0


def test_build_gate_report_promotes_passing_candidate() -> None:
    combined = pl.DataFrame(
        [
            {
                "ticker": "SPY",
                "strategy": "Stub",
                "direction": "long",
                "oos_windows": 6,
                "oos_signals": 4000,
                "avg_test_exp_r": 0.1,
                "pct_positive_oos_windows": 0.8,
                "avg_test_confidence": 0.6,
            }
        ]
    )
    report = build_gate_report(
        combined=combined,
        cost_count=1,
        gate_min_oos_windows=6,
        gate_min_oos_signals=3000,
        gate_min_pct_positive=0.67,
        gate_min_exp_r=0.0,
    )
    row = report.row(0, named=True)
    assert row["passes_all_gates"] is True
    assert row["decision"] == "promote_to_holdout"


def test_holdout_helpers(tmp_path) -> None:
    gate_dir = tmp_path / "results"
    gate_dir.mkdir()
    strict = gate_dir / "convergence_gate_report_2026-01-01.csv"
    relaxed = gate_dir / "convergence_gate_report_relaxed_2026-01-02.csv"
    strict.write_text("ticker,strategy,direction,decision\nSPY,Elastic Band z=1.25/w=360+dm,short,promote_to_holdout\n", encoding="utf-8")
    relaxed.write_text("ignored", encoding="utf-8")

    latest = latest_csv(gate_dir, "convergence_gate_report", exclude_substrings=("relaxed",))
    assert latest == strict

    gate_df = pl.DataFrame(
        [{"ticker": "SPY", "strategy": "Elastic Band z=1.25/w=360+dm", "direction": "short", "decision": "promote_to_holdout"}]
    )
    promoted = promoted_candidates_from_gate_report(gate_df)
    assert promoted.height == 1


def test_holdout_stage_logic() -> None:
    df_eval = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(4)],
            "close": [10.0, 10.0, 10.0, 10.0],
            "signal": [True, False, True, False],
            "signal_direction": ["short", None, "short", None],
            "forward_mfe_eod": [2.0, None, 2.0, None],
            "forward_mae_eod": [1.0, None, 1.0, None],
        }
    )
    chosen_ratio, stats = choose_ratio(
        calib_df=df_eval,
        direction="short",
        ratios=[1.0, 1.5],
        cost_bps=8.0,
        min_calib_signals=1,
    )
    assert chosen_ratio in {1.0, 1.5}
    assert stats["signals"] == 2

    holdout_stats = eval_holdout_direction(df_eval, "short", 1.5, 8.0)
    assert holdout_stats["signals"] == 2
    assert holdout_stats["exp_r"] is not None

    detail_df = pl.DataFrame(
        [
            {
                "ticker": "SPY",
                "strategy": "Elastic Band z=1.25/w=360+dm",
                "direction": "short",
                "cost_bps": 8.0,
                "selected_ratio": 1.5,
                "calib_signals": 10,
                "calib_exp_r": 0.1,
                "holdout_signals": 600,
                "holdout_confidence": 0.7,
                "holdout_exp_r": 0.2,
                "passes_cost_gate": True,
            }
        ]
    )
    summary_df = summarize_holdout(detail_df, cost_count=1)
    row = summary_df.row(0, named=True)
    assert row["decision"] == "promote_to_execution_mapping"


def test_execution_stage_helpers() -> None:
    holdout_summary = pl.DataFrame(
        [{"ticker": "SPY", "strategy": "Elastic Band z=1.25/w=360+dm", "direction": "short", "decision": "promote_to_execution_mapping"}]
    )
    promoted = promoted_candidates_from_holdout(holdout_summary)
    assert promoted.height == 1

    holdout_detail = pl.DataFrame(
        [
            {"ticker": "SPY", "strategy": "Elastic Band z=1.25/w=360+dm", "direction": "short", "selected_ratio": 1.0},
            {"ticker": "SPY", "strategy": "Elastic Band z=1.25/w=360+dm", "direction": "short", "selected_ratio": 1.5},
        ]
    )
    assert median_selected_ratio(
        holdout_detail,
        ticker="SPY",
        strategy="Elastic Band z=1.25/w=360+dm",
        direction="short",
    ) == 1.25

    mapping = option_mapping_for("Elastic Band Reversion", "short")
    assert mapping["structure"] == "put_debit_spread"


def test_run_late_stage_helpers_with_real_strategy() -> None:
    timestamps = [datetime(2025, 12, 1, 14, 30) + timedelta(minutes=i) for i in range(40)]
    opens = [10.0] * 40
    close = []
    high = []
    low = []
    volume = []
    accel = []
    jerk = []
    directional_mass = []

    for idx in range(40):
        if idx < 25:
            price = 10.0 - (idx * 0.012)
            close.append(price)
            high.append(price + 0.01)
            low.append(price - 0.02)
            volume.append(1000.0)
            accel.append(-0.1)
            jerk.append(-0.1)
            directional_mass.append(-5.0)
        elif idx == 25:
            price = 9.65
            close.append(price)
            high.append(price + 0.01)
            low.append(price - 0.03)
            volume.append(2500.0)
            accel.append(-0.2)
            jerk.append(-0.2)
            directional_mass.append(-10.0)
        else:
            price = 9.6 - ((idx - 25) * 0.01)
            close.append(price)
            high.append(price + 0.01)
            low.append(price - 0.02)
            volume.append(1200.0)
            accel.append(-0.1)
            jerk.append(-0.1)
            directional_mass.append(-5.0)

    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "accel_1m": accel,
            "jerk_1m": jerk,
            "directional_mass": directional_mass,
        }
    )
    promoted = pl.DataFrame(
        [{"ticker": "SPY", "strategy": "Opening Drive v2 (Short Continue)", "direction": "short"}]
    )
    metrics = MetricsCalculator()

    detail_rows = run_holdout_validation_for_candidates(
        promoted=promoted,
        ticker_frames={"SPY": frame},
        metrics=metrics,
        start_date=date(2025, 12, 1),
        calibration_end=date(2025, 12, 1),
        holdout_start=date(2025, 12, 1),
        holdout_end=date(2025, 12, 1),
        ratios=[1.0],
        costs=[8.0],
        min_calibration_signals=1,
        min_holdout_signals=1,
    )
    assert detail_rows

    execution_rows = run_execution_mapping_for_candidates(
        promoted=pl.DataFrame(
            [{"ticker": "SPY", "strategy": "Opening Drive v2 (Short Continue)", "direction": "short"}]
        ),
        holdout_detail=pl.DataFrame(detail_rows),
        ticker_frames={"SPY": frame},
        metrics=metrics,
        holdout_start=date(2025, 12, 1),
        holdout_end=date(2025, 12, 1),
        base_cost_r=0.08,
        stress_cfg=ExecutionStressConfig(bootstrap_iters=10),
    )
    assert execution_rows
    assert execution_rows[0]["structure"] == "put_debit_spread"


def test_run_walk_forward_for_strategies() -> None:
    ts = [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(8)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "close": [10.0] * 8,
            "high": [10.0, 12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0],
            "low": [10.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0],
            "signal": [True, False, False, False, True, False, False, False],
            "signal_direction": ["long", None, None, None, "short", None, None, None],
        }
    )
    metrics = MetricsCalculator()
    windows = [
        type("W", (), {
            "train_start": date(2025, 1, 1),
            "train_end": date(2025, 1, 1),
            "test_start": date(2025, 1, 1),
            "test_end": date(2025, 1, 1),
        })()
    ]

    rows = run_walk_forward_for_strategies(
        ticker="SPY",
        df=df,
        strategies=[_StubStrategy()],
        windows=windows,
        ratios=[1.0],
        metrics=metrics,
        min_signals=1,
        cost_r=0.05,
    )

    assert rows
    assert rows[0]["ticker"] == "SPY"
    assert rows[0]["strategy"] == "Stub Directional"
