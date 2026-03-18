"""Typed models for the research orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrategyStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    UNDER_EVAL = "under_eval"
    DEAD = "dead"


class ResearchStage(str, Enum):
    M1_DISCOVERY = "M1"
    M2_CONVERGENCE = "M2"
    M3_WALK_FORWARD = "M3"
    M4_HOLDOUT = "M4"
    M5_EXECUTION = "M5"


class ResearchDecision(str, Enum):
    PROMOTE = "promote"
    RETUNE = "retune"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    KILL = "kill"


@dataclass(slots=True)
class ArchitectureDecisions:
    workflow_model: str = "hybrid_agentic"
    stage_governance: str = "deterministic"
    agent_authority: str = "auto_experiment_only"
    entrypoint_model: str = "dual_layer"
    notes: str = ""


@dataclass(slots=True)
class ResearchAgentSpec:
    name: str
    spec_path: str
    role: str
    persona: str
    objective: str
    allowed_tasks: list[str] = field(default_factory=list)
    forbidden_tasks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationStrategy:
    strategy: str
    status: StrategyStatus
    representative_tickers: list[str]
    expected_directions: list[str]
    why_it_matters: str
    minimum_smoke_test: str


@dataclass(slots=True)
class TrackedStrategy:
    name: str
    status: StrategyStatus
    tickers: list[str]
    directions: list[str]
    optimal_params: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""
    notes: str = ""


@dataclass(slots=True)
class ResearchState:
    architecture: ArchitectureDecisions
    research_agent: ResearchAgentSpec | None
    validation: list[ValidationStrategy]
    strategies: dict[str, TrackedStrategy]


@dataclass(slots=True)
class StrategyCatalogEntry:
    name: str
    status: StrategyStatus
    tickers: list[str]
    directions: list[str]
    evaluation_mode: str
    required_features: list[str]
    parameter_space: dict[str, list[Any]]
    strategy_config: dict[str, Any]
    notes: str = ""
    evidence: str = ""


@dataclass(slots=True)
class OrchestrationAction:
    stage: ResearchStage
    action: str
    summary: str
    agent_can_run: bool
    tool_name: str
