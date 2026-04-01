from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd
import polars as pl

import src.research.review_queue as review_queue_module
from src.research.nightly_matrix import NightlyRegimeMatrixConfig
from src.research.review_queue import (
    HumanReviewQueueManager,
    QUEUE_STATUS_EXECUTED,
    QUEUE_STATUS_KILLED,
    QUEUE_STATUS_PENDING,
)


def _sample_opening_drive_frame(ticker: str = "SPY") -> pl.DataFrame:
    timestamps: list[datetime] = []
    rows: list[dict[str, object]] = []
    for day_offset, day in enumerate([date(2026, 3, 27), date(2026, 3, 30), date(2026, 3, 31)]):
        base_prices = [
            (9, 30, 100.0, 100.1, 99.9, 100.0),
            (9, 35, 100.0, 100.4, 100.0, 100.35),
            (9, 45, 100.35, 100.6, 100.3, 100.55),
            (10, 0, 100.55, 100.7, 100.4, 100.65),
        ]
        for hour, minute, open_, high, low, close in base_prices:
            ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
            timestamps.append(ts)
            rows.append(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "open": open_ + day_offset,
                    "high": high + day_offset,
                    "low": low + day_offset,
                    "close": close + day_offset,
                    "volume": 1_000,
                    "accel_1m": 1.0,
                }
            )
    return pl.DataFrame(rows)


def _write_candidate_run(
    run_dir: Path,
    *,
    ticker: str = "SPY",
    strategy: str = "Opening Drive Classifier",
    direction: str = "long",
    include_m3: bool = True,
    include_m4: bool = True,
    include_m5: bool = True,
) -> None:
    params = {
        "opening_window_minutes": 25,
        "entry_start_offset_minutes": 30,
        "entry_end_offset_minutes": 120,
        "min_drive_return_pct": 0.0015,
        "breakout_buffer_pct": 0.0,
        "use_volume_filter": False,
        "volume_multiplier": 1.2,
        "use_directional_mass": False,
        "use_jerk_confirmation": False,
        "kinematic_periods_back": 1,
    }
    m1_top = pl.DataFrame(
        [
            {
                "ticker": ticker,
                "strategy": strategy,
                "direction": direction,
                **params,
                "oos_windows": 6,
                "oos_signals": 180,
                "avg_test_exp_r": 0.12,
                "pct_positive_oos_windows": 0.67,
                "avg_test_confidence": 0.58,
                "m1_score": 130.0,
            }
        ]
    )
    m2_gate = pl.DataFrame(
        [
            {
                "ticker": ticker,
                "strategy": strategy,
                "direction": direction,
                **params,
                "observed_cost_points": 3,
                "min_oos_windows": 6,
                "min_oos_signals": 180,
                "min_avg_test_exp_r": 0.08,
                "mean_avg_test_exp_r": 0.1,
                "min_pct_positive_oos_windows": 0.67,
                "mean_pct_positive_oos_windows": 0.72,
                "mean_test_confidence": 0.58,
                "has_all_cost_points": True,
                "passes_window_gate": True,
                "passes_signal_gate": True,
                "passes_stability_gate": True,
                "passes_exp_gate": True,
                "passes_all_gates": True,
                "decision": "promote_to_holdout",
                "score": 160.0,
            }
        ]
    )
    m1_top.write_csv(run_dir / "m1_top_candidates.csv")
    m2_gate.write_csv(run_dir / "m2_gate_report.csv")
    if include_m3:
        pl.DataFrame(
            [{"ticker": ticker, "strategy": strategy, "direction": direction, **params, "window_idx": 1}]
        ).write_csv(run_dir / "m3_walk_forward_detail.csv")
    if include_m4:
        pl.DataFrame(
            [
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "direction": direction,
                    **params,
                    "observed_cost_points": 3,
                    "min_holdout_signals": 25,
                    "min_holdout_exp_r": 0.05,
                    "mean_holdout_exp_r": 0.07,
                    "passes_all_cost_gates": True,
                    "passes_holdout": True,
                    "decision": "promote_to_execution_mapping",
                }
            ]
        ).write_csv(run_dir / "m4_holdout_summary.csv")
    if include_m5:
        pl.DataFrame(
            [
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "direction": direction,
                    **params,
                    "selected_ratio": 1.5,
                    "holdout_trades": 25,
                    "holdout_win_rate": 0.6,
                    "base_exp_r": 0.2,
                    "execution_profile": "single_option",
                    "stress_profile": "single_option",
                }
            ]
        ).write_csv(run_dir / "m5_execution_mapping.csv")


def _manager(tmp_path: Path) -> HumanReviewQueueManager:
    return HumanReviewQueueManager(
        tmp_path / "control",
        frame_loader=lambda ticker, start, end: _sample_opening_drive_frame(ticker),
        enricher=lambda frame, required: frame,
        followup_executor=None,
    )


def test_queue_row_creation_and_chart_link_population(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    run_dir = tmp_path / "family_run"
    run_dir.mkdir()
    _write_candidate_run(run_dir)
    config = NightlyRegimeMatrixConfig(
        research_control_root=str(tmp_path / "control"),
        watchlist=["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL"],
    )

    observations = manager.collect_observations(
        run_dirs={"opening_drive_classifier": run_dir},
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert len(observations) == 1
    row = observations[0]
    assert row["ticker"] == "SPY"
    assert row["research_slice_id"].startswith("opening_drive_classifier-")
    assert row["chart_link"].endswith(".html")
    assert Path(row["chart_link"]).exists()
    assert row["passes_m1"] is True
    assert row["passes_m2"] is True
    assert row["passes_m3"] is True
    assert row["passes_m4"] is True
    assert row["passes_m5"] is True
    assert row["is_full_m1_m5_survivor"] is True
    assert row["latest_stage_reached"] == "M5"


def test_queue_marks_m3_passed_when_later_stages_exist(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    run_dir = tmp_path / "family_run"
    run_dir.mkdir()
    _write_candidate_run(run_dir, include_m3=False, include_m4=True, include_m5=True)
    config = NightlyRegimeMatrixConfig(
        research_control_root=str(tmp_path / "control"),
        watchlist=["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL"],
    )

    observations = manager.collect_observations(
        run_dirs={"opening_drive_classifier": run_dir},
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert len(observations) == 1
    row = observations[0]
    assert row["passes_m3"] is True
    assert row["passes_m4"] is True
    assert row["passes_m5"] is True
    assert row["is_full_m1_m5_survivor"] is True


def test_refresh_queue_writes_empty_review_surface_for_zero_survivor_night(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    run_dir = tmp_path / "family_run"
    run_dir.mkdir()
    for name in [
        "m1_top_candidates.csv",
        "m2_gate_report.csv",
        "m3_walk_forward_detail.csv",
        "m4_holdout_summary.csv",
        "m5_execution_mapping.csv",
    ]:
        (run_dir / name).write_text("\n", encoding="utf-8")
    config = NightlyRegimeMatrixConfig(
        research_control_root=str(tmp_path / "control"),
        watchlist=["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL"],
    )

    artifacts = manager.refresh_queue(
        run_dirs={"opening_drive_classifier": run_dir},
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert artifacts.queue_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.workbook_path.exists()
    assert (artifacts.review_bundle_dir / "m2_review.csv").exists()
    assert (artifacts.review_bundle_dir / "execution_queue.csv").exists()
    queue_df = pd.read_csv(artifacts.queue_path)
    assert queue_df.empty
    assert "candidate_key" in queue_df.columns


def test_repeated_nightly_survivor_merge_updates_observation_fields(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    existing = [
        {
            "candidate_key": "abc",
            "ticker": "SPY",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "queue_status": "NEW",
            "human_decision": "",
            "human_notes": "",
            "priority": 0,
            "last_seen_run_date": "2026-03-31",
            "m2_score": 100.0,
        }
    ]
    observation = [
        {
            "candidate_key": "abc",
            "ticker": "SPY",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "chart_link": str(tmp_path / "chart.html"),
            "passes_m1": True,
            "passes_m2": True,
            "passes_m3": False,
            "passes_m4": False,
            "passes_m5": False,
            "latest_stage_reached": "M2",
            "latest_stage_decision": "promote_to_holdout",
            "last_seen_run_date": "2026-04-01",
            "m2_score": 160.0,
        }
    ]

    merged = manager._merge_rows(
        existing_rows=existing,
        snapshot={},
        observations=observation,
        run_date=date(2026, 4, 1),
        max_new_rows=10,
    )

    assert len(merged) == 1
    assert merged[0]["queue_status"] == "NEW"
    assert merged[0]["last_seen_run_date"] == "2026-04-01"
    assert merged[0]["m2_score"] == 160.0


def test_terminal_state_preservation_for_executed_and_killed_rows(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    observation = [
        {
            "candidate_key": "terminal",
            "ticker": "SPY",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "chart_link": str(tmp_path / "chart.html"),
            "passes_m1": True,
            "passes_m2": True,
            "last_seen_run_date": "2026-04-01",
        }
    ]
    for status in (QUEUE_STATUS_EXECUTED, QUEUE_STATUS_KILLED):
        merged = manager._merge_rows(
            existing_rows=[
                {
                    "candidate_key": "terminal",
                    "ticker": "SPY",
                    "strategy": "Opening Drive Classifier",
                    "direction": "long",
                    "queue_status": status,
                    "human_decision": "promote_to_m3",
                    "human_notes": "keep terminal",
                    "priority": 2,
                    "last_seen_run_date": "2026-03-31",
                }
            ],
            snapshot={
                "terminal": {
                    "queue_status": status,
                    "human_decision": "promote_to_m3",
                    "human_notes": "keep terminal",
                    "priority": 2,
                    "human_updated_at": "",
                }
            },
            observations=observation,
            run_date=date(2026, 4, 1),
            max_new_rows=10,
        )
        assert merged[0]["queue_status"] == status


def test_manual_override_reopens_terminal_row(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    merged = manager._merge_rows(
        existing_rows=[
            {
                "candidate_key": "terminal",
                "ticker": "SPY",
                "strategy": "Opening Drive Classifier",
                "direction": "long",
                "queue_status": QUEUE_STATUS_EXECUTED,
                "human_decision": "retune",
                "human_notes": "reopen",
                "priority": 3,
                "last_seen_run_date": "2026-03-31",
            }
        ],
        snapshot={
            "terminal": {
                "queue_status": QUEUE_STATUS_EXECUTED,
                "human_decision": "promote_to_m3",
                "human_notes": "",
                "priority": 1,
                "human_updated_at": "",
            }
        },
        observations=[
            {
                "candidate_key": "terminal",
                "ticker": "SPY",
                "strategy": "Opening Drive Classifier",
                "direction": "long",
                "chart_link": str(tmp_path / "chart.html"),
                "passes_m1": True,
                "passes_m2": True,
                "last_seen_run_date": "2026-04-01",
            }
        ],
        run_date=date(2026, 4, 1),
        max_new_rows=10,
    )

    assert merged[0]["queue_status"] == QUEUE_STATUS_PENDING
    assert merged[0]["human_decision"] == "retune"


def test_queue_consumption_honors_budget_caps(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def executor(row: dict[str, object], decision: str, config: NightlyRegimeMatrixConfig, artifact_dir: Path) -> dict[str, object]:
        calls.append((str(row["candidate_key"]), decision))
        return {
            "queue_status": QUEUE_STATUS_EXECUTED,
            "latest_stage_reached": "M2",
            "latest_stage_decision": f"{decision}_done",
            "latest_artifact_dir": str(artifact_dir),
        }

    manager = HumanReviewQueueManager(
        tmp_path / "control",
        frame_loader=lambda ticker, start, end: _sample_opening_drive_frame(str(ticker)),
        enricher=lambda frame, required: frame,
        followup_executor=executor,
    )
    config = NightlyRegimeMatrixConfig(
        research_control_root=str(tmp_path / "control"),
        followup_budgets={
            "max_new_m2_rows_per_night": 10,
            "max_retune_tasks_per_night": 1,
            "max_symbol_expansion_tasks_per_night": 1,
            "max_m3_promotions_per_night": 1,
            "max_total_followup_tasks_per_night": 2,
        },
    )
    rows = [
        {
            "candidate_key": "c1",
            "ticker": "SPY",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "queue_status": "PENDING",
            "human_decision": "promote_to_m3",
            "priority": 5,
            "last_seen_run_date": "2026-04-01",
            "human_updated_at": "2026-04-01T00:00:00+00:00",
        },
        {
            "candidate_key": "c2",
            "ticker": "QQQ",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "queue_status": "PENDING",
            "human_decision": "retune",
            "priority": 4,
            "last_seen_run_date": "2026-04-01",
            "human_updated_at": "2026-04-01T00:00:00+00:00",
        },
        {
            "candidate_key": "c3",
            "ticker": "IWM",
            "strategy": "Opening Drive Classifier",
            "direction": "long",
            "queue_status": "PENDING",
            "human_decision": "expand_symbols",
            "priority": 3,
            "last_seen_run_date": "2026-04-01",
            "human_updated_at": "2026-04-01T00:00:00+00:00",
        },
    ]

    updated, followup_actions_run_count = manager._execute_followups(
        rows=rows,
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert calls == [("c1", "promote_to_m3"), ("c2", "retune")]
    assert followup_actions_run_count == 2
    status_by_key = {row["candidate_key"]: row["queue_status"] for row in updated}
    assert status_by_key["c1"] == QUEUE_STATUS_EXECUTED
    assert status_by_key["c2"] == QUEUE_STATUS_EXECUTED
    assert status_by_key["c3"] == "PENDING"


def test_promote_to_m3_marks_row_executed_via_followup_execution(tmp_path: Path) -> None:
    manager = HumanReviewQueueManager(
        tmp_path / "control",
        frame_loader=lambda ticker, start, end: _sample_opening_drive_frame(str(ticker)),
        enricher=lambda frame, required: frame,
        followup_executor=lambda row, decision, config, artifact_dir: {
            "queue_status": QUEUE_STATUS_EXECUTED,
            "passes_m3": True,
            "passes_m4": True,
            "passes_m5": False,
            "latest_stage_reached": "M4",
            "latest_stage_decision": "promote_to_execution_mapping",
            "latest_artifact_dir": str(artifact_dir),
        },
    )
    config = NightlyRegimeMatrixConfig(research_control_root=str(tmp_path / "control"))
    updated, followup_actions_run_count = manager._execute_followups(
        rows=[
            {
                "candidate_key": "promo",
                "ticker": "SPY",
                "strategy": "Opening Drive Classifier",
                "direction": "long",
                "queue_status": "PENDING",
                "human_decision": "promote_to_m3",
                "priority": 1,
                "last_seen_run_date": "2026-04-01",
                "human_updated_at": "2026-04-01T00:00:00+00:00",
                "passes_m1": True,
                "passes_m2": True,
            }
        ],
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert followup_actions_run_count == 1
    assert updated[0]["queue_status"] == QUEUE_STATUS_EXECUTED
    assert updated[0]["latest_stage_reached"] == "M4"


def test_review_bundle_generation_writes_required_views(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    run_dir = tmp_path / "family_run"
    run_dir.mkdir()
    _write_candidate_run(run_dir)
    config = NightlyRegimeMatrixConfig(research_control_root=str(tmp_path / "control"))

    artifacts = manager.refresh_queue(
        run_dirs={"opening_drive_classifier": run_dir},
        config=config,
        run_date=date(2026, 4, 1),
    )

    assert artifacts.queue_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.workbook_path.exists()
    assert (artifacts.review_bundle_dir / "m2_review.csv").exists()
    assert (artifacts.review_bundle_dir / "recent_history.csv").exists()
    assert (artifacts.review_bundle_dir / "execution_queue.csv").exists()
    assert (artifacts.review_bundle_dir / "full_survivors.csv").exists()
    assert (artifacts.review_bundle_dir / "charts_index.csv").exists()

    queue_df = pd.read_csv(artifacts.queue_path)
    assert {"passes_m1", "passes_m2", "passes_m3", "passes_m4", "passes_m5", "is_full_m1_m5_survivor"} <= set(queue_df.columns)


def test_candidate_stage_payload_uses_config_json_only(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    payload = manager._candidate_stage_payload(
        {
            "ticker": "TSLA",
            "strategy": "Market Impulse (Cross & Reclaim)",
            "direction": "short",
            "config_json": json.dumps(
                {
                    "entry_buffer_minutes": 3,
                    "entry_window_minutes": 45,
                    "regime_timeframe": "30m",
                }
            ),
            "candidate_key": "should-not-leak",
            "m2_score": 123.0,
        }
    )

    assert payload == {
        "ticker": "TSLA",
        "strategy": "Market Impulse (Cross & Reclaim)",
        "direction": "short",
        "entry_buffer_minutes": 3,
        "entry_window_minutes": 45,
        "regime_timeframe": "30m",
    }


def test_catalog_strategy_name_uses_family_for_elastic_variants(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    assert manager._catalog_strategy_name(
        {
            "strategy_family": "elastic_band_reversion",
            "strategy": "Elastic Band z=2.0/w=240+dm",
        }
    ) == "Elastic Band Reversion"
    assert manager._catalog_strategy_name(
        {
            "strategy_family": "market_impulse",
            "strategy": "Market Impulse (Cross & Reclaim)",
        }
    ) == "Market Impulse (Cross & Reclaim)"


def test_load_enriched_frame_for_strategies_uses_union_of_required_features(tmp_path: Path) -> None:
    captured: list[set[str]] = []

    class _StrategyA:
        required_features = {"alpha"}
        feature_requests: tuple[str, ...] = ()

    class _StrategyB:
        required_features = {"beta"}
        feature_requests: tuple[str, ...] = ()

    manager = HumanReviewQueueManager(
        tmp_path / "control",
        frame_loader=lambda ticker, start, end: _sample_opening_drive_frame(str(ticker)),
        enricher=lambda frame, required: captured.append(set(required)) or frame,
        followup_executor=None,
    )

    frame = manager._load_enriched_frame_for_strategies(
        ticker="SPY",
        strategies=[_StrategyA(), _StrategyB()],
        start_date=date(2026, 3, 27),
        end_date=date(2026, 3, 31),
    )

    assert not frame.is_empty()
    assert captured == [{"alpha", "beta"}]


def test_write_frame_if_not_empty_stringifies_nested_values(tmp_path: Path) -> None:
    from src.research.review_queue import _write_frame_if_not_empty

    path = tmp_path / "nested.csv"
    frame = pl.DataFrame(
        [
            {
                "ticker": "QQQ",
                "payload": {"window": 60, "tf": "1h"},
                "points": [1, 2, 3],
            }
        ]
    )

    _write_frame_if_not_empty(frame, path)

    text = path.read_text(encoding="utf-8")
    assert "QQQ" in text
    assert '""window"": 60' in text
    assert "[1, 2, 3]" in text


def test_execute_retune_loads_per_neighbor_and_allows_gated_schema_differences(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _manager(tmp_path)

    class _DummyEntry:
        pass

    class _DummyStrategy:
        required_features = {"timestamp"}
        feature_requests: tuple[str, ...] = ()

    load_calls: list[str] = []
    build_calls: list[dict[str, object]] = []
    combined_frames: list[pl.DataFrame] = []

    monkeypatch.setattr(manager.registry, "catalog_entry", lambda strategy_name, params=None: _DummyEntry())
    monkeypatch.setattr(
        review_queue_module,
        "build_neighbor_configs",
        lambda entry, base_config, max_configs: [
            {"vpoc_proximity_pct": 0.002, "use_volume_filter": True, "volume_multiplier": 1.3},
            {"vpoc_proximity_pct": 0.002, "use_volume_filter": False},
        ],
    )
    monkeypatch.setattr(
        manager.registry,
        "build",
        lambda strategy_name, params=None: build_calls.append(dict(params or {})) or _DummyStrategy(),
    )
    monkeypatch.setattr(
        manager,
        "_load_enriched_frame_for_strategy",
        lambda **kwargs: load_calls.append(str(kwargs["ticker"])) or _sample_opening_drive_frame(str(kwargs["ticker"])),
    )
    monkeypatch.setattr(
        review_queue_module,
        "run_walk_forward_for_strategies",
        lambda **kwargs: [{"direction": "short", "window_idx": 1}],
    )
    monkeypatch.setattr(
        review_queue_module,
        "aggregate_walk_forward",
        lambda rows: pl.DataFrame(
            [
                {
                    "ticker": "TSLA",
                    "strategy": "Jerk-Pivot Momentum (tight)",
                    "direction": "short",
                    "avg_test_exp_r": 0.2,
                    "pct_positive_oos_windows": 0.7,
                    "oos_windows": 6,
                    "oos_signals": 100,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        review_queue_module,
        "build_gate_report",
        lambda **kwargs: combined_frames.append(kwargs["combined"]) or pl.DataFrame([{"decision": "ok"}]),
    )

    result = manager._execute_retune(
        row={
            "ticker": "TSLA",
            "strategy_family": "jerk_pivot_momentum",
            "strategy": "Jerk-Pivot Momentum (tight)",
            "direction": "short",
            "config_json": json.dumps(
                {
                    "vpoc_proximity_pct": 0.002,
                    "jerk_lookback": 10,
                    "volume_multiplier": 1.3,
                    "volume_ma_period": 20,
                    "use_volume_filter": True,
                    "use_time_filter": True,
                    "session_start": "09:35",
                    "session_end": "15:30",
                }
            ),
        },
        config=NightlyRegimeMatrixConfig(research_control_root=str(tmp_path / "control")),
        artifact_dir=tmp_path / "retune",
    )

    assert result["latest_stage_decision"] == "retune_completed"
    assert load_calls == ["TSLA", "TSLA"]
    assert len(build_calls) == 2
    assert len(combined_frames) == 1
    assert "volume_multiplier" in combined_frames[0].columns
