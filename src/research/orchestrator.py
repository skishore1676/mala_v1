"""Phase-1 research orchestrator: registry-backed planning and stage actions."""

from __future__ import annotations

from pathlib import Path

from src.research.models import OrchestrationAction, ResearchStage, StrategyCatalogEntry
from src.research.registry import ResearchRegistry
from src.research.tools import ResearchToolbox


_STAGE_ACTIONS: dict[ResearchStage, list[tuple[str, str, str]]] = {
    ResearchStage.M1_DISCOVERY: [
        ("parameter_sweep", "parameter_sweep", "Search for any edge in the declared parameter space."),
        ("baseline_comparison", "baseline_comparison", "Compare the candidate against the refactor validation baselines."),
    ],
    ResearchStage.M2_CONVERGENCE: [
        ("convergence_grid", "convergence_grid", "Stress the candidate across friction and stability assumptions."),
        ("ablation_check", "ablation_check", "Test whether the proposed edge survives key feature removal."),
    ],
    ResearchStage.M3_WALK_FORWARD: [
        ("walk_forward", "walk_forward", "Select parameters on train windows and verify OOS adaptation."),
    ],
    ResearchStage.M4_HOLDOUT: [
        ("holdout_validation", "holdout_validation", "Evaluate the untouched quarantine segment without retuning."),
    ],
    ResearchStage.M5_EXECUTION: [
        ("execution_mapping", "execution_mapping", "Map the edge to practical execution assumptions and stress it."),
    ],
}


class ResearchOrchestrator:
    """Small orchestration facade used while scripts are being refactored."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.registry = ResearchRegistry(state_path)
        self._state_path = state_path

    def validation_entries(self) -> list[StrategyCatalogEntry]:
        return self.registry.validation_entries()

    def strategy_entry(self, strategy_name: str) -> StrategyCatalogEntry:
        return self.registry.catalog_entry(strategy_name)

    def next_actions(self, stage: ResearchStage) -> list[OrchestrationAction]:
        authority = self.registry.state.architecture.agent_authority
        agent_can_run = authority == "auto_experiment_only"
        return [
            OrchestrationAction(
                stage=stage,
                action=action,
                summary=summary,
                agent_can_run=agent_can_run,
                tool_name=tool_name,
            )
            for action, tool_name, summary in _STAGE_ACTIONS[stage]
        ]

    def toolbox(self) -> ResearchToolbox:
        return ResearchToolbox(self._state_path)
