"""Research orchestration package.

Keep package import side effects small so strategy modules can safely import
`src.research.models` without pulling in the full research runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from src.research.models import (
    ConstraintSpec,
    DomainSpec,
    ForbiddenPair,
    GatingCondition,
    MonotonicOrdering,
    OrchestrationAction,
    ObjectiveSpec,
    ParameterSpec,
    ResearchDecision,
    ResearchStage,
    StrategyCatalogEntry,
    StrategySearchSpec,
    StrategyStatus,
)

__all__ = [
    "ConstraintSpec",
    "DomainSpec",
    "ForbiddenPair",
    "GatingCondition",
    "LoopArtifactExporter",
    "MonotonicOrdering",
    "NightlyRegimeMatrixConfig",
    "NightlyRegimeMatrixResult",
    "ObjectiveSpec",
    "OrchestrationAction",
    "ParameterSpec",
    "ResearchDecision",
    "ResearchJournal",
    "ResearchOrchestrator",
    "ResearchRegistry",
    "ResearchStage",
    "ResearchToolResult",
    "ResearchToolbox",
    "StrategyCatalogEntry",
    "StrategySearchSpec",
    "StrategyStatus",
    "load_nightly_regime_matrix_config",
    "load_research_state",
    "run_nightly_regime_matrix",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LoopArtifactExporter": ("src.research.loop_export", "LoopArtifactExporter"),
    "NightlyRegimeMatrixConfig": ("src.research.nightly_matrix", "NightlyRegimeMatrixConfig"),
    "NightlyRegimeMatrixResult": ("src.research.nightly_matrix", "NightlyRegimeMatrixResult"),
    "ResearchJournal": ("src.research.reporting", "ResearchJournal"),
    "ResearchOrchestrator": ("src.research.orchestrator", "ResearchOrchestrator"),
    "ResearchRegistry": ("src.research.registry", "ResearchRegistry"),
    "ResearchToolResult": ("src.research.tools", "ResearchToolResult"),
    "ResearchToolbox": ("src.research.tools", "ResearchToolbox"),
    "load_nightly_regime_matrix_config": ("src.research.nightly_matrix", "load_nightly_regime_matrix_config"),
    "load_research_state": ("src.research.state", "load_research_state"),
    "run_nightly_regime_matrix": ("src.research.nightly_matrix", "run_nightly_regime_matrix"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
