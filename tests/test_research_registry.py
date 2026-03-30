"""Tests for the phase-1 research registry and orchestrator."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.research import (
    ResearchOrchestrator,
    ResearchRegistry,
    ResearchStage,
    ResearchToolbox,
    load_research_state,
)
from src.research.tools import _annotate_plateau_metrics, _bounded_param_grid


def _write_state_file(path: Path) -> None:
    path.write_text(
        """
architecture_decisions:
  workflow_model: hybrid_agentic
  stage_governance: deterministic
  agent_authority: auto_experiment_only
  entrypoint_model: dual_layer
  notes: test fixture

research_agent:
  name: research-experiment-agent
  spec_path: skills/research-experiment-agent/SKILL.md
  role: test role
  persona: skeptical
  objective: test objective
  allowed_tasks:
    - run bounded experiments
  forbidden_tasks:
    - change gates

refactor_validation:
  strategies:
    - strategy: "Elastic Band Reversion"
      status: active
      representative_tickers: [META]
      expected_directions: [short]
      why_it_matters: baseline
      minimum_smoke_test: smoke

strategies:
  "Elastic Band Reversion":
    status: active
    tickers: [META]
    directions: [short]
    optimal_params:
      z_score_threshold: 1.25
      z_score_window: 360
      use_directional_mass: true
    notes: elastic notes
  "Market Impulse (Cross & Reclaim)":
    status: candidate
    tickers: [QQQ, SPY]
    directions: [short]
    optimal_params:
      entry_buffer_minutes: 5
      entry_window_minutes: 60
      regime_timeframe: 1h
      vma_col: vma_10
    evidence: market impulse evidence
    notes: market impulse notes
        """.strip(),
        encoding="utf-8",
    )


def _sample_raw_frame() -> pl.DataFrame:
    timestamps = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(160)]
    closes = [100.0 + ((i % 12) - 6) * 0.4 for i in range(160)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [price + 0.5 for price in closes],
            "low": [price - 0.5 for price in closes],
            "close": closes,
            "volume": [1000 + (i % 5) * 100 for i in range(160)],
        }
    )


def test_load_research_state(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    state = load_research_state(state_path)

    assert state.architecture.workflow_model == "hybrid_agentic"
    assert state.research_agent is not None
    assert "Elastic Band Reversion" in state.strategies
    assert "Market Impulse (Cross & Reclaim)" in state.strategies
    assert state.validation[0].strategy == "Elastic Band Reversion"


def test_registry_builds_strategy_from_state_params(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    registry = ResearchRegistry(state_path)
    strategy = registry.build("Elastic Band Reversion")

    assert strategy.name == "Elastic Band z=1.25/w=360+dm"


def test_registry_builds_tracked_and_validation_strategies(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    registry = ResearchRegistry(state_path)

    tracked = registry.build_tracked_strategies()
    validation = registry.build_validation_strategies()

    assert [strategy.name for strategy in tracked] == [
        "Elastic Band z=1.25/w=360+dm",
        "Market Impulse (Cross & Reclaim)",
    ]
    assert [strategy.name for strategy in validation] == ["Elastic Band z=1.25/w=360+dm"]


def test_orchestrator_exposes_next_actions(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    orchestrator = ResearchOrchestrator(state_path)
    actions = orchestrator.next_actions(ResearchStage.M2_CONVERGENCE)

    assert [action.action for action in actions] == ["convergence_grid", "ablation_check"]
    assert [action.tool_name for action in actions] == ["convergence_grid", "ablation_check"]
    assert all(action.agent_can_run for action in actions)


def test_orchestrator_exposes_toolbox(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    orchestrator = ResearchOrchestrator(state_path)
    toolbox = orchestrator.toolbox()

    assert isinstance(toolbox, ResearchToolbox)
    assert toolbox.available_tools(ResearchStage.M1_DISCOVERY) == [
        "parameter_sweep",
        "baseline_comparison",
    ]


def test_toolbox_parameter_sweep_and_baselines(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    toolbox = ResearchToolbox(state_path)

    sweep = toolbox.parameter_sweep("Elastic Band Reversion", max_configs=3)
    assert sweep.summary["config_count"] == 3
    assert len(sweep.artifacts["configs"]) == 3

    comparison = toolbox.baseline_comparison("Elastic Band Reversion")
    assert comparison.summary["baseline_count"] == 1
    assert comparison.artifacts["comparisons"][0]["baseline"] == "Elastic Band z=1.25/w=360+dm"


def test_bounded_param_grid_samples_across_full_space() -> None:
    configs = _bounded_param_grid(
        {
            "a": [1, 2],
            "b": [10, 20],
            "c": [100, 200],
        },
        max_configs=4,
    )

    assert len(configs) == 4
    assert {config["a"] for config in configs} == {1, 2}
    assert {config["b"] for config in configs} == {10, 20}


def test_toolbox_parameter_sweep_executes_walk_forward(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    toolbox = ResearchToolbox(state_path)
    result = toolbox.parameter_sweep(
        "Elastic Band Reversion",
        parameter_space={
            "z_score_threshold": [1.0],
            "z_score_window": [5],
            "use_directional_mass": [False],
        },
        max_configs=1,
        tickers=["META"],
        ticker_frames={"META": _sample_raw_frame()},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 5, 31),
        train_months=2,
        test_months=1,
        ratios=[1.0],
        min_signals=1,
        cost_r=0.05,
        min_total_signals=1,
    )

    assert result.summary["ticker_count"] == 1
    assert "detail" in result.artifacts
    assert "aggregate" in result.artifacts
    assert isinstance(result.artifacts["detail"], pl.DataFrame)
    assert isinstance(result.artifacts["aggregate"], pl.DataFrame)
    assert {"oos_signals", "avg_test_exp_r", "pct_positive_oos_windows", "avg_test_confidence"} <= set(
        result.artifacts["aggregate"].columns
    )


def test_orchestrator_can_run_allowed_action(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    orchestrator = ResearchOrchestrator(state_path)
    result = orchestrator.run_action(
        ResearchStage.M1_DISCOVERY,
        "parameter_sweep",
        strategy_name="Elastic Band Reversion",
        max_configs=1,
    )

    assert result.tool_name == "parameter_sweep"
    assert result.summary["config_count"] == 1


def test_parameter_sweep_dedupes_noop_kinematic_variants() -> None:
    toolbox = ResearchToolbox()

    result = toolbox.parameter_sweep(
        "Kinematic Ladder",
        parameter_space={
            "regime_window": [20],
            "accel_window": [8],
            "use_volume_filter": [False],
            "volume_multiplier": [1.0, 1.2],
        },
        max_configs=8,
    )

    assert result.summary["requested_config_count"] == 2
    assert result.summary["duplicate_config_count"] == 1
    assert result.summary["config_count"] == 1
    assert len(result.artifacts["configs"]) == 1


def test_aggregate_sweep_adds_plateau_metrics() -> None:
    aggregate_df = pl.DataFrame(
        [
            {
                "ticker": "TSLA",
                "direction": "short",
                "catalog_strategy": "Kinematic Ladder",
                "base_strategy": "KinematicLadderStrategy",
                "regime_window": 20,
                "accel_window": 8,
                "use_volume_filter": False,
                "volume_multiplier": 1.0,
                "avg_test_exp_r": 0.12,
            },
            {
                "ticker": "TSLA",
                "direction": "short",
                "catalog_strategy": "Kinematic Ladder",
                "base_strategy": "KinematicLadderStrategy",
                "regime_window": 30,
                "accel_window": 8,
                "use_volume_filter": False,
                "volume_multiplier": 1.0,
                "avg_test_exp_r": 0.09,
            },
            {
                "ticker": "TSLA",
                "direction": "short",
                "catalog_strategy": "Kinematic Ladder",
                "base_strategy": "KinematicLadderStrategy",
                "regime_window": 45,
                "accel_window": 20,
                "use_volume_filter": False,
                "volume_multiplier": 1.0,
                "avg_test_exp_r": 0.22,
            },
        ]
    )

    annotated = _annotate_plateau_metrics(
        aggregate_df,
        parameter_space={
            "regime_window": [20, 30, 45],
            "accel_window": [8, 12, 20],
            "use_volume_filter": [True, False],
            "volume_multiplier": [1.0, 1.05, 1.1, 1.2],
        },
    )

    assert "plateau_neighbor_count" in annotated.columns
    assert "plateau_positive_ratio" in annotated.columns
    tsla_30 = annotated.filter(pl.col("regime_window") == 30).row(0, named=True)
    assert tsla_30["plateau_neighbor_count"] >= 1
    assert tsla_30["plateau_positive_ratio"] > 0


def test_toolbox_convergence_grid_returns_report(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    toolbox = ResearchToolbox(state_path)
    combined = pl.DataFrame(
        [
            {
                "ticker": "META",
                "strategy": "Elastic Band z=1.25/w=360+dm",
                "direction": "short",
                "oos_windows": 6,
                "oos_signals": 5000,
                "avg_test_exp_r": 0.12,
                "pct_positive_oos_windows": 0.8,
                "avg_test_confidence": 0.6,
            }
        ]
    )

    result = toolbox.convergence_grid(
        combined=combined,
        cost_count=1,
        gate_min_oos_windows=6,
        gate_min_oos_signals=3000,
        gate_min_pct_positive=0.67,
        gate_min_exp_r=0.0,
    )

    assert result.summary["promoted_count"] == 1
    assert result.artifacts["gate_report"].row(0, named=True)["decision"] == "promote_to_holdout"
