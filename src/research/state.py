"""Load the repository research registry and agent contract memory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

from src.research.models import (
    ArchitectureDecisions,
    ResearchAgentSpec,
    ResearchState,
    StrategyStatus,
    TrackedStrategy,
    ValidationStrategy,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        ruby_cmd = [
            "ruby",
            "-e",
            "require 'yaml'; require 'json'; puts JSON.generate(YAML.load_file(ARGV[0]))",
            str(path),
        ]
        try:
            completed = subprocess.run(
                ruby_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Loading research_state.yaml requires PyYAML or a Ruby runtime with YAML support."
            ) from exc
        return json.loads(completed.stdout)

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)  # type: ignore[attr-defined]
    return data or {}


def _coerce_status(value: str | None) -> StrategyStatus:
    if not value:
        return StrategyStatus.CANDIDATE
    return StrategyStatus(value)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_research_state(path: Path | None = None) -> ResearchState:
    state_path = path or (PROJECT_ROOT / "research_state.yaml")
    raw = _read_yaml(state_path)

    arch_raw = raw.get("architecture_decisions", {})
    architecture = ArchitectureDecisions(
        workflow_model=str(arch_raw.get("workflow_model", "hybrid_agentic")),
        stage_governance=str(arch_raw.get("stage_governance", "deterministic")),
        agent_authority=str(arch_raw.get("agent_authority", "auto_experiment_only")),
        entrypoint_model=str(arch_raw.get("entrypoint_model", "dual_layer")),
        notes=str(arch_raw.get("notes", "")),
    )

    agent_raw = raw.get("research_agent")
    research_agent: ResearchAgentSpec | None = None
    if isinstance(agent_raw, dict):
        research_agent = ResearchAgentSpec(
            name=str(agent_raw.get("name", "")),
            spec_path=str(agent_raw.get("spec_path", "")),
            role=str(agent_raw.get("role", "")),
            persona=str(agent_raw.get("persona", "")),
            objective=str(agent_raw.get("objective", "")),
            allowed_tasks=_coerce_str_list(agent_raw.get("allowed_tasks")),
            forbidden_tasks=_coerce_str_list(agent_raw.get("forbidden_tasks")),
        )

    validation_raw = raw.get("refactor_validation", {}).get("strategies", [])
    validation: list[ValidationStrategy] = []
    for item in validation_raw:
        if not isinstance(item, dict):
            continue
        validation.append(
            ValidationStrategy(
                strategy=str(item.get("strategy", "")),
                status=_coerce_status(item.get("status")),
                representative_tickers=_coerce_str_list(item.get("representative_tickers")),
                expected_directions=_coerce_str_list(item.get("expected_directions")),
                why_it_matters=str(item.get("why_it_matters", "")),
                minimum_smoke_test=str(item.get("minimum_smoke_test", "")),
            )
        )

    strategies: dict[str, TrackedStrategy] = {}
    for name, item in raw.get("strategies", {}).items():
        if not isinstance(item, dict):
            continue
        strategies[str(name)] = TrackedStrategy(
            name=str(name),
            status=_coerce_status(item.get("status")),
            tickers=_coerce_str_list(item.get("tickers")),
            directions=_coerce_str_list(item.get("directions")),
            optimal_params=dict(item.get("optimal_params", {}) or {}),
            evidence=str(item.get("evidence", "")),
            notes=str(item.get("notes", "")),
        )

    return ResearchState(
        architecture=architecture,
        research_agent=research_agent,
        validation=validation,
        strategies=strategies,
    )
