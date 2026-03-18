"""Tests for the phase-1 research registry and orchestrator."""

from __future__ import annotations

from pathlib import Path

from src.research import ResearchOrchestrator, ResearchRegistry, ResearchStage, load_research_state


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
        """.strip(),
        encoding="utf-8",
    )


def test_load_research_state(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    state = load_research_state(state_path)

    assert state.architecture.workflow_model == "hybrid_agentic"
    assert state.research_agent is not None
    assert "Elastic Band Reversion" in state.strategies
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

    assert [strategy.name for strategy in tracked] == ["Elastic Band z=1.25/w=360+dm"]
    assert [strategy.name for strategy in validation] == ["Elastic Band z=1.25/w=360+dm"]


def test_orchestrator_exposes_next_actions(tmp_path: Path) -> None:
    state_path = tmp_path / "research_state.yaml"
    _write_state_file(state_path)

    orchestrator = ResearchOrchestrator(state_path)
    actions = orchestrator.next_actions(ResearchStage.M2_CONVERGENCE)

    assert [action.action for action in actions] == ["convergence_grid", "ablation_check"]
    assert all(action.agent_can_run for action in actions)
