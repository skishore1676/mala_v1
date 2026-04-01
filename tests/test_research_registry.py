"""Tests for the phase-1 research registry and orchestrator."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import subprocess
import sys
from typing import Any

import polars as pl

from src.research import (
    ResearchOrchestrator,
    ResearchRegistry,
    ResearchStage,
    ResearchToolbox,
    load_research_state,
)
from src.research.models import (
    ConstraintSpec,
    DomainSpec,
    GatingCondition,
    ObjectiveSpec,
    ParameterSpec,
    StrategyCatalogEntry,
    StrategySearchSpec,
    StrategyStatus,
)
from src.research.tools import ResearchToolResult, _annotate_plateau_metrics, _bounded_param_grid
from src.strategy.base import BaseStrategy


class _StubStrategy(BaseStrategy):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})

    @property
    def name(self) -> str:
        return "Stub Strategy"

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def strategy_config(self) -> dict[str, Any]:
        return dict(self._params)


class _StubRegistry:
    def __init__(self, entry: StrategyCatalogEntry) -> None:
        self._entry = entry

    def catalog_entry(self, strategy_name: str, params: dict | None = None) -> StrategyCatalogEntry:
        return self._entry

    def build(self, strategy_name: str, params: dict | None = None) -> BaseStrategy:
        return _StubStrategy(params)


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

    assert [action.action for action in actions] == [
        "convergence_grid",
        "ablation_check",
        "evaluate_config",
    ]
    assert [action.tool_name for action in actions] == [
        "convergence_grid",
        "ablation_check",
        "evaluate_config",
    ]
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
        "evaluate_config",
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


def test_research_package_lazy_imports_avoid_strategy_circular_import_regression() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        (
            "from src.strategy.kinematic_ladder import KinematicLadderStrategy; "
            "from src.strategy.factory import build_strategy; "
            "from src.research import ResearchToolbox; "
            "print(KinematicLadderStrategy.__name__, build_strategy('Kinematic Ladder').name, ResearchToolbox.__name__)"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "KinematicLadderStrategy" in completed.stdout
    assert "ResearchToolbox" in completed.stdout


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


def test_parameter_sweep_prefers_search_spec_over_legacy_parameter_space(monkeypatch) -> None:
    toolbox = ResearchToolbox()
    entry = StrategyCatalogEntry(
        name="Stub Strategy",
        status=StrategyStatus.CANDIDATE,
        tickers=[],
        directions=[],
        evaluation_mode="directional",
        required_features=[],
        parameter_space={"fast_window": [5]},
        search_spec=StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="fast_window",
                    type="discrete",
                    domain=DomainSpec(values=[8, 13]),
                    default=8,
                    prior_center=13,
                )
            ],
            objective=ObjectiveSpec(
                primary_metric="avg_test_exp_r",
                minimum_signals=20,
            ),
        ),
        strategy_config={"fast_window": 8},
    )
    toolbox.registry = _StubRegistry(entry)

    monkeypatch.setattr(
        toolbox,
        "_strategy_sweep_result",
        lambda **kwargs: ResearchToolResult(
            tool_name=kwargs["tool_name"],
            summary={"config_count": len(kwargs["configs"])},
            artifacts={"configs": kwargs["configs"]},
        ),
    )

    result = toolbox.parameter_sweep("Stub Strategy", max_configs=1)

    assert result.artifacts["configs"] == [{"fast_window": 13}]


def test_parameter_sweep_prunes_inactive_parameters_before_budget_is_spent(monkeypatch) -> None:
    toolbox = ResearchToolbox()
    entry = StrategyCatalogEntry(
        name="Stub Strategy",
        status=StrategyStatus.CANDIDATE,
        tickers=[],
        directions=[],
        evaluation_mode="directional",
        required_features=[],
        parameter_space={"use_filter": [True, False], "threshold": [1, 2]},
        search_spec=StrategySearchSpec(
            parameters=[
                ParameterSpec(
                    name="use_filter",
                    type="categorical",
                    domain=DomainSpec(values=[True, False]),
                    default=True,
                    prior_center=True,
                ),
                ParameterSpec(
                    name="threshold",
                    type="discrete",
                    domain=DomainSpec(values=[1, 2]),
                    default=1,
                    prior_center=1,
                ),
            ],
            constraints=ConstraintSpec(
                gating_conditions=[
                    GatingCondition(parameter="threshold", requires={"use_filter": True})
                ]
            ),
        ),
        strategy_config={"use_filter": True, "threshold": 1},
    )
    toolbox.registry = _StubRegistry(entry)
    monkeypatch.setattr(
        toolbox,
        "_strategy_sweep_result",
        lambda **kwargs: ResearchToolResult(
            tool_name=kwargs["tool_name"],
            summary={"config_count": len(kwargs["configs"])},
            artifacts={"configs": kwargs["configs"]},
        ),
    )

    result = toolbox.parameter_sweep("Stub Strategy", max_configs=10)

    assert result.summary["requested_config_count"] == 4
    assert result.summary["config_count"] == 3
    assert {"use_filter": False} in result.artifacts["configs"]
    assert sum(1 for config in result.artifacts["configs"] if config == {"use_filter": False}) == 1


def test_parameter_sweep_falls_back_to_parameter_space_when_search_spec_missing(monkeypatch) -> None:
    toolbox = ResearchToolbox()
    entry = StrategyCatalogEntry(
        name="Stub Strategy",
        status=StrategyStatus.CANDIDATE,
        tickers=[],
        directions=[],
        evaluation_mode="directional",
        required_features=[],
        parameter_space={"fast_window": [5, 9]},
        search_spec=None,
        strategy_config={"fast_window": 5},
    )
    toolbox.registry = _StubRegistry(entry)
    monkeypatch.setattr(
        toolbox,
        "_strategy_sweep_result",
        lambda **kwargs: ResearchToolResult(
            tool_name=kwargs["tool_name"],
            summary={"config_count": len(kwargs["configs"])},
            artifacts={"configs": kwargs["configs"]},
        ),
    )

    result = toolbox.parameter_sweep("Stub Strategy", max_configs=2)

    assert result.artifacts["configs"] == [{"fast_window": 5}, {"fast_window": 9}]


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


def test_catalog_entry_exposes_search_spec_and_inactive_parameter_rules(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    registry = ResearchRegistry(state_path)
    entry = registry.catalog_entry("Kinematic Ladder")

    assert entry.search_spec is not None
    assert [parameter.name for parameter in entry.search_spec.parameters][:2] == [
        "regime_window",
        "accel_window",
    ]
    assert entry.search_spec.objective is not None
    assert entry.search_spec.objective.primary_metric == "avg_test_exp_r"

    normalized = entry.search_spec.normalize_config(
        {"use_volume_filter": False, "volume_multiplier": 1.2},
        base_config=entry.strategy_config,
    )

    assert normalized.valid
    assert normalized.inactive_parameters == ["volume_multiplier"]
    assert "volume_multiplier" not in normalized.config


def test_search_spec_rejects_structurally_invalid_combinations(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    registry = ResearchRegistry(state_path)
    entry = registry.catalog_entry("Market Impulse (Cross & Reclaim)")
    assert entry.search_spec is not None

    normalized = entry.search_spec.normalize_config(
        {"entry_buffer_minutes": 45, "entry_window_minutes": 5},
        base_config=entry.strategy_config,
    )

    assert not normalized.valid
    assert any("Monotonic ordering violated" in error for error in normalized.errors)


def test_aggregate_sweep_treats_nan_metrics_as_missing() -> None:
    detail_df = pl.DataFrame(
        {
            "ticker": ["AMD", "AMD"],
            "strategy": ["Elastic Band Reversion", "Elastic Band Reversion"],
            "direction": ["long", "long"],
            "catalog_strategy": ["Elastic Band Reversion", "Elastic Band Reversion"],
            "base_strategy": ["ElasticBandReversionStrategy", "ElasticBandReversionStrategy"],
            "z_score_threshold": [2.0, 2.0],
            "z_score_window": [240, 240],
            "use_directional_mass": [False, False],
            "test_signals": [100, 120],
            "test_exp_r": [float("nan"), 0.25],
            "test_confidence": [float("nan"), 0.55],
            "effective_cost_r": [float("nan"), 0.08],
            "selected_ratio": [2.0, 2.0],
        }
    )

    aggregate = ResearchToolbox._aggregate_sweep(
        detail_df,
        [{"z_score_threshold": 2.0, "z_score_window": 240, "use_directional_mass": False}],
        min_total_signals=100,
    )

    row = aggregate.row(0, named=True)
    assert row["avg_oos_exp_r"] == 0.25
    assert row["avg_confidence"] == 0.55
    assert row["avg_effective_cost_r"] == 0.08
    assert row["discovery_score"] is not None


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


def test_parameter_sweep_dedupes_noop_jerk_pivot_volume_variants() -> None:
    toolbox = ResearchToolbox()

    result = toolbox.parameter_sweep(
        "Jerk-Pivot Momentum (tight)",
        parameter_space={
            "vpoc_proximity_pct": [0.002],
            "jerk_lookback": [10],
            "volume_multiplier": [1.0, 1.1, 1.2],
            "use_volume_filter": [False],
        },
        max_configs=3,
        tickers=["NVDA"],
        ticker_frames={"NVDA": _sample_raw_frame()},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 5, 31),
        train_months=2,
        test_months=1,
        ratios=[1.0],
        min_signals=1,
        cost_r=0.05,
        min_total_signals=1,
    )

    assert result.summary["requested_config_count"] == 3
    assert result.summary["duplicate_config_count"] == 2
    assert result.summary["config_count"] == 1


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


def test_toolbox_evaluate_config_runs_single_point_and_returns_optimizer_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)
    toolbox = ResearchToolbox(state_path, results_db_path=tmp_path / "research_results.db")

    result = toolbox.evaluate_config(
        "Elastic Band Reversion",
        config={
            "z_score_threshold": 1.0,
            "z_score_window": 120,
            "kinematic_periods_back": 1,
            "use_directional_mass": False,
        },
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

    assert result.summary["status"] in {"ok", "insufficient_signals"}
    assert result.summary["already_evaluated"] is False
    assert result.summary["config_signature"]
    assert result.summary["objective"]["primary_metric"] == "avg_test_exp_r"
    assert "total_signals" in result.summary["constraints"]
    assert result.summary["runtime_seconds"] >= 0.0
    assert isinstance(result.artifacts["detail"], pl.DataFrame)
    assert isinstance(result.artifacts["aggregate"], pl.DataFrame)


def test_toolbox_evaluate_config_short_circuits_duplicates(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)
    toolbox = ResearchToolbox(state_path, results_db_path=tmp_path / "research_results.db")
    kwargs = {
        "strategy_name": "Elastic Band Reversion",
        "config": {
            "z_score_threshold": 1.0,
            "z_score_window": 120,
            "kinematic_periods_back": 1,
            "use_directional_mass": False,
        },
        "tickers": ["META"],
        "ticker_frames": {"META": _sample_raw_frame()},
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 5, 31),
        "train_months": 2,
        "test_months": 1,
        "ratios": [1.0],
        "min_signals": 1,
        "cost_r": 0.05,
        "min_total_signals": 1,
    }

    first = toolbox.evaluate_config(**kwargs)
    second = toolbox.evaluate_config(**kwargs)

    assert first.summary["status"] in {"ok", "insufficient_signals"}
    assert second.summary["status"] == "duplicate"
    assert second.summary["already_evaluated"] is True
    assert second.summary["config_signature"] == first.summary["config_signature"]


def test_toolbox_compact_memory_queries_summarize_without_dumping_history(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)
    toolbox = ResearchToolbox(state_path, results_db_path=tmp_path / "research_results.db")

    shared_kwargs = {
        "strategy_name": "Elastic Band Reversion",
        "tickers": ["META"],
        "ticker_frames": {"META": _sample_raw_frame()},
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 5, 31),
        "train_months": 2,
        "test_months": 1,
        "ratios": [1.0],
        "min_signals": 1,
        "cost_r": 0.05,
        "min_total_signals": 1,
    }
    configs = [
        {"z_score_threshold": 1.0, "z_score_window": 120, "kinematic_periods_back": 1, "use_directional_mass": False},
        {"z_score_threshold": 1.25, "z_score_window": 120, "kinematic_periods_back": 1, "use_directional_mass": False},
        {"z_score_threshold": 1.75, "z_score_window": 120, "kinematic_periods_back": 1, "use_directional_mass": True},
    ]
    for config in configs:
        toolbox.evaluate_config(config=config, **shared_kwargs)
    toolbox._results_db.store_research_evaluation(
        {
            "strategy": "Elastic Band Reversion",
            "config": {"z_score_threshold": 1.25, "z_score_window": 120},
            "config_signature": "manual-competitive",
            "request_signature": "manual-competitive:req",
            "status": "ok",
            "already_evaluated": False,
            "inactive_parameters": [],
            "objective": {"primary_metric": "avg_test_exp_r", "value": 0.11, "confidence": 0.58},
            "constraints": {"total_signals": 140, "passes_signal_floor": True},
            "runtime_seconds": 0.1,
            "slice": {"tickers": ["META"]},
            "errors": [],
        }
    )

    incumbent = toolbox.query_incumbent("Elastic Band Reversion", ticker="META")
    neighborhood = toolbox.query_neighborhood(
        "Elastic Band Reversion",
        {"z_score_threshold": 1.25, "z_score_window": 120, "kinematic_periods_back": 1, "use_directional_mass": False},
        radius=1,
        limit=3,
    )
    pareto = toolbox.query_pareto_front("Elastic Band Reversion", ticker="META", limit=3)

    assert incumbent.summary["status"] == "ok"
    assert incumbent.summary["evaluated_configs"] >= 1
    assert set(incumbent.summary["incumbent"]) >= {"config_signature", "config", "objective", "constraints"}
    assert neighborhood.summary["status"] == "ok"
    assert len(neighborhood.summary["neighbors"]) >= 1
    assert pareto.summary["status"] == "ok"
    assert pareto.summary["front_size"] >= 1


def test_incumbent_and_pareto_exclude_insufficient_configs_by_default(tmp_path: Path) -> None:
    toolbox = ResearchToolbox(results_db_path=tmp_path / "research_results.db")
    toolbox._results_db.store_research_evaluation(
        {
            "strategy": "Elastic Band Reversion",
            "config": {"z_score_threshold": 1.0},
            "config_signature": "competitive",
            "request_signature": "competitive:req",
            "status": "ok",
            "already_evaluated": False,
            "inactive_parameters": [],
            "objective": {"primary_metric": "avg_test_exp_r", "value": 0.08, "confidence": 0.55},
            "constraints": {"total_signals": 120, "passes_signal_floor": True},
            "runtime_seconds": 0.1,
            "slice": {"tickers": ["META"]},
            "errors": [],
        }
    )
    toolbox._results_db.store_research_evaluation(
        {
            "strategy": "Elastic Band Reversion",
            "config": {"z_score_threshold": 3.0},
            "config_signature": "insufficient",
            "request_signature": "insufficient:req",
            "status": "insufficient_signals",
            "already_evaluated": False,
            "inactive_parameters": [],
            "objective": {"primary_metric": "avg_test_exp_r", "value": 0.5, "confidence": 0.9},
            "constraints": {"total_signals": 3, "passes_signal_floor": False},
            "runtime_seconds": 0.1,
            "slice": {"tickers": ["META"]},
            "errors": [],
        }
    )

    incumbent = toolbox.query_incumbent("Elastic Band Reversion", ticker="META")
    incumbent_all = toolbox.query_incumbent(
        "Elastic Band Reversion",
        ticker="META",
        include_non_competitive=True,
    )
    pareto = toolbox.query_pareto_front("Elastic Band Reversion", ticker="META")

    assert incumbent.summary["incumbent"]["config_signature"] == "competitive"
    assert incumbent_all.summary["incumbent"]["config_signature"] == "insufficient"
    assert [row["config_signature"] for row in pareto.summary["pareto_front"]] == ["competitive"]


def test_query_dead_zones_returns_compact_failure_summary(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)
    toolbox = ResearchToolbox(state_path, results_db_path=tmp_path / "research_results.db")

    shared_kwargs = {
        "strategy_name": "Elastic Band Reversion",
        "tickers": ["META"],
        "ticker_frames": {"META": _sample_raw_frame()},
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 5, 31),
        "train_months": 2,
        "test_months": 1,
        "ratios": [1.0],
        "min_signals": 1,
        "cost_r": 0.05,
        "min_total_signals": 10_000,
    }
    toolbox.evaluate_config(
        config={
            "z_score_threshold": 1.0,
            "z_score_window": 120,
            "kinematic_periods_back": 1,
            "use_directional_mass": False,
        },
        **shared_kwargs,
    )
    second_kwargs = dict(shared_kwargs)
    second_kwargs["end_date"] = date(2025, 6, 30)
    toolbox.evaluate_config(
        config={
            "z_score_threshold": 1.25,
            "z_score_window": 120,
            "kinematic_periods_back": 1,
            "use_directional_mass": False,
        },
        **second_kwargs,
    )

    dead_zones = toolbox.query_dead_zones("Elastic Band Reversion", ticker="META")

    assert dead_zones.summary["status"] == "ok"
    assert any(zone["parameter"] == "use_directional_mass" for zone in dead_zones.summary["dead_zones"])
