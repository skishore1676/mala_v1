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
    "ACTIVE_SESSION_CONTRACT_NAME",
    "BiasInputRow",
    "BhikshaPublishReport",
    "ConstraintSpec",
    "DomainSpec",
    "ForbiddenPair",
    "GatingCondition",
    "HumanReviewQueueManager",
    "LiveObservationRecord",
    "LoopArtifactExporter",
    "MonotonicOrdering",
    "ManualEntryRow",
    "NightlyRegimeMatrixConfig",
    "NightlyFollowupBudgets",
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
    "PlaybookRecord",
    "StrategyCatalogEntry",
    "StrategySearchSpec",
    "StrategyStatus",
    "augment_playbook_catalog_from_queue",
    "build_playbook_records_from_queue",
    "compile_active_session_from_google_sheets",
    "compile_active_session_from_rows",
    "load_nightly_regime_matrix_config",
    "load_bias_inputs_sheet",
    "load_live_observation_records",
    "load_manual_entries_sheet",
    "load_playbook_records",
    "publish_active_session_to_bhiksha",
    "publish_armed_playbooks_to_bhiksha",
    "route_google_sheet_bias_inputs",
    "load_research_state",
    "route_bias_inputs",
    "route_bias_rows",
    "run_nightly_regime_matrix",
    "write_live_observation_records",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ACTIVE_SESSION_CONTRACT_NAME": ("src.research.session_payloads", "ACTIVE_SESSION_CONTRACT_NAME"),
    "BiasInputRow": ("src.research.playbooks", "BiasInputRow"),
    "BhikshaPublishReport": ("src.research.bhiksha_bridge", "BhikshaPublishReport"),
    "HumanReviewQueueManager": ("src.research.review_queue", "HumanReviewQueueManager"),
    "LiveObservationRecord": ("src.research.playbooks", "LiveObservationRecord"),
    "LoopArtifactExporter": ("src.research.loop_export", "LoopArtifactExporter"),
    "ManualEntryRow": ("src.research.session_payloads", "ManualEntryRow"),
    "NightlyRegimeMatrixConfig": ("src.research.nightly_matrix", "NightlyRegimeMatrixConfig"),
    "NightlyFollowupBudgets": ("src.research.nightly_matrix", "NightlyFollowupBudgets"),
    "NightlyRegimeMatrixResult": ("src.research.nightly_matrix", "NightlyRegimeMatrixResult"),
    "PlaybookRecord": ("src.research.playbooks", "PlaybookRecord"),
    "ResearchJournal": ("src.research.reporting", "ResearchJournal"),
    "ResearchOrchestrator": ("src.research.orchestrator", "ResearchOrchestrator"),
    "ResearchRegistry": ("src.research.registry", "ResearchRegistry"),
    "ResearchToolResult": ("src.research.tools", "ResearchToolResult"),
    "ResearchToolbox": ("src.research.tools", "ResearchToolbox"),
    "augment_playbook_catalog_from_queue": ("src.research.playbooks", "augment_playbook_catalog_from_queue"),
    "build_playbook_records_from_queue": ("src.research.playbooks", "build_playbook_records_from_queue"),
    "compile_active_session_from_google_sheets": ("src.research.session_payloads", "compile_active_session_from_google_sheets"),
    "compile_active_session_from_rows": ("src.research.session_payloads", "compile_active_session_from_rows"),
    "load_bias_inputs_sheet": ("src.research.playbooks", "load_bias_inputs_sheet"),
    "load_live_observation_records": ("src.research.playbooks", "load_live_observation_records"),
    "load_manual_entries_sheet": ("src.research.session_payloads", "load_manual_entries_sheet"),
    "load_nightly_regime_matrix_config": ("src.research.nightly_matrix", "load_nightly_regime_matrix_config"),
    "load_playbook_records": ("src.research.playbooks", "load_playbook_records"),
    "load_research_state": ("src.research.state", "load_research_state"),
    "publish_active_session_to_bhiksha": ("src.research.session_payloads", "publish_active_session_to_bhiksha"),
    "publish_armed_playbooks_to_bhiksha": ("src.research.bhiksha_bridge", "publish_armed_playbooks_to_bhiksha"),
    "route_google_sheet_bias_inputs": ("src.research.playbooks", "route_google_sheet_bias_inputs"),
    "route_bias_inputs": ("src.research.playbooks", "route_bias_inputs"),
    "route_bias_rows": ("src.research.playbooks", "route_bias_rows"),
    "run_nightly_regime_matrix": ("src.research.nightly_matrix", "run_nightly_regime_matrix"),
    "write_live_observation_records": ("src.research.playbooks", "write_live_observation_records"),
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
