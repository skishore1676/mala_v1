"""Typed models for the research orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


ParameterType = Literal["categorical", "discrete"]
OrderingDirection = Literal["strictly_ascending", "strictly_descending"]


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
class DomainSpec:
    values: list[Any] | None = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None

    def enumerate_values(self) -> list[Any]:
        if self.values is not None:
            return list(self.values)
        if self.min is None or self.max is None or self.step in (None, 0):
            return []
        values: list[Any] = []
        current = float(self.min)
        upper = float(self.max)
        step = float(self.step)
        is_integral = all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (self.min, self.max, self.step)
        )
        while current <= upper + (step / 10.0):
            values.append(int(round(current)) if is_integral else round(current, 10))
            current += step
        return values


@dataclass(slots=True)
class ParameterSpec:
    name: str
    type: ParameterType
    domain: DomainSpec
    default: Any
    prior_center: Any | None = None

    def legal_values(self) -> list[Any]:
        values = self.domain.enumerate_values()
        if self.default not in values:
            values.append(self.default)
        if self.prior_center is not None and self.prior_center not in values:
            values.append(self.prior_center)
        return values

    def search_values(self) -> list[Any]:
        ordered: list[Any] = []
        for preferred in (self.prior_center, self.default):
            if preferred is None or preferred in ordered:
                continue
            ordered.append(preferred)
        for value in self.legal_values():
            if value not in ordered:
                ordered.append(value)
        return ordered


@dataclass(slots=True)
class GatingCondition:
    parameter: str
    requires: dict[str, Any]

    def is_active(self, config: dict[str, Any]) -> bool:
        return all(config.get(key) == value for key, value in self.requires.items())


@dataclass(slots=True)
class ForbiddenPair:
    left: dict[str, Any]
    right: dict[str, Any] | None = None
    message: str = ""

    def matches(self, config: dict[str, Any]) -> bool:
        left_matches = all(config.get(key) == value for key, value in self.left.items())
        if not left_matches:
            return False
        if self.right is None:
            return True
        return all(config.get(key) == value for key, value in self.right.items())


@dataclass(slots=True)
class MonotonicOrdering:
    parameters: list[str]
    direction: OrderingDirection = "strictly_ascending"


@dataclass(slots=True)
class ConstraintSpec:
    gating_conditions: list[GatingCondition] = field(default_factory=list)
    monotonic_ordering: list[MonotonicOrdering] = field(default_factory=list)
    forbidden_pairs: list[ForbiddenPair] = field(default_factory=list)


@dataclass(slots=True)
class ObjectiveSpec:
    primary_metric: str
    minimum_signals: int
    tie_breakers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedSearchConfig:
    config: dict[str, Any]
    inactive_parameters: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class StrategySearchSpec:
    parameters: list[ParameterSpec]
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    objective: ObjectiveSpec | None = None

    @classmethod
    def from_parameter_space(
        cls,
        parameter_space: dict[str, list[Any]],
        *,
        strategy_config: dict[str, Any] | None = None,
        objective: ObjectiveSpec | None = None,
    ) -> StrategySearchSpec | None:
        if not parameter_space:
            return None
        strategy_config = strategy_config or {}
        parameters: list[ParameterSpec] = []
        for name, raw_values in parameter_space.items():
            values = list(raw_values)
            default = strategy_config.get(name, values[0] if values else None)
            if default not in values:
                values.append(default)
            inferred_type: ParameterType
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                inferred_type = "discrete"
                values = sorted(values)
            else:
                inferred_type = "categorical"
            parameters.append(
                ParameterSpec(
                    name=name,
                    type=inferred_type,
                    domain=DomainSpec(values=values),
                    default=default,
                    prior_center=default,
                )
            )
        return cls(parameters=parameters, objective=objective)

    def parameter_map(self) -> dict[str, ParameterSpec]:
        return {parameter.name: parameter for parameter in self.parameters}

    def parameter_space(self) -> dict[str, list[Any]]:
        return {
            parameter.name: parameter.legal_values()
            for parameter in self.parameters
        }

    def search_space(self) -> dict[str, list[Any]]:
        return {
            parameter.name: parameter.search_values()
            for parameter in self.parameters
        }

    def default_config(self) -> dict[str, Any]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def prior_config(self) -> dict[str, Any]:
        return {
            parameter.name: (
                parameter.prior_center
                if parameter.prior_center is not None
                else parameter.default
            )
            for parameter in self.parameters
        }

    def normalize_config(
        self,
        config: dict[str, Any],
        *,
        base_config: dict[str, Any] | None = None,
        prune_inactive: bool = True,
        validate_values: bool = True,
    ) -> NormalizedSearchConfig:
        normalized = dict(base_config or {})
        normalized.update(config)
        parameter_map = self.parameter_map()
        errors: list[str] = []

        for key in config:
            if key not in parameter_map and (base_config is None or key not in base_config):
                errors.append(f"Unknown parameter '{key}'")

        inactive: list[str] = []
        if prune_inactive:
            for condition in self.constraints.gating_conditions:
                if condition.parameter not in parameter_map:
                    continue
                if condition.is_active(normalized):
                    continue
                if condition.parameter in normalized:
                    normalized.pop(condition.parameter, None)
                inactive.append(condition.parameter)

        if validate_values:
            for name, spec in parameter_map.items():
                if name not in normalized:
                    continue
                legal_values = spec.legal_values()
                if legal_values and normalized[name] not in legal_values:
                    errors.append(
                        f"Parameter '{name}' has illegal value {normalized[name]!r}; "
                        f"expected one of {legal_values!r}"
                    )

        for ordering in self.constraints.monotonic_ordering:
            values = [normalized.get(name) for name in ordering.parameters]
            if any(value is None for value in values):
                continue
            pairs = zip(values, values[1:])
            if ordering.direction == "strictly_ascending":
                valid = all(left < right for left, right in pairs)
                relation = "<"
            else:
                valid = all(left > right for left, right in pairs)
                relation = ">"
            if not valid:
                joined = f" {relation} ".join(ordering.parameters)
                errors.append(f"Monotonic ordering violated: expected {joined}")

        for pair in self.constraints.forbidden_pairs:
            if pair.matches(normalized):
                message = pair.message or "Forbidden parameter combination"
                errors.append(message)

        return NormalizedSearchConfig(
            config=normalized,
            inactive_parameters=sorted(set(inactive)),
            errors=errors,
        )


@dataclass(slots=True)
class StrategyCatalogEntry:
    name: str
    status: StrategyStatus
    tickers: list[str]
    directions: list[str]
    evaluation_mode: str
    required_features: list[str]
    parameter_space: dict[str, list[Any]]
    search_spec: StrategySearchSpec | None
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
