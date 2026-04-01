"""Registry helpers for tracked strategies and validation fixtures."""

from __future__ import annotations

from pathlib import Path

from src.research.models import (
    ObjectiveSpec,
    StrategyCatalogEntry,
    StrategySearchSpec,
    StrategyStatus,
    ValidationStrategy,
)
from src.research.state import load_research_state
from src.strategy.base import BaseStrategy, required_feature_union
from src.strategy.factory import build_strategy


DEFAULT_AGENT_CATALOG: tuple[str, ...] = (
    "Elastic Band Reversion",
    "Jerk-Pivot Momentum (tight)",
    "Opening Drive Classifier",
    "Market Impulse (Cross & Reclaim)",
)


class ResearchRegistry:
    """Bridge repo memory to runtime strategy instances."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state = load_research_state(state_path)

    def tracked_names(self, *, include_lineage: bool = False) -> list[str]:
        if include_lineage:
            return sorted(self.state.strategies)
        return [
            name for name in DEFAULT_AGENT_CATALOG
            if name in self.state.strategies
        ]

    def validation_set(self) -> list[ValidationStrategy]:
        return list(self.state.validation)

    def tracked_entries(
        self,
        include_dead: bool = False,
        *,
        include_lineage: bool = False,
    ) -> list[StrategyCatalogEntry]:
        names = self.tracked_names(include_lineage=include_lineage)
        entries = [self.catalog_entry(name) for name in names]
        if include_dead:
            return entries
        return [entry for entry in entries if entry.status is not StrategyStatus.DEAD]

    def build(self, strategy_name: str, params: dict | None = None) -> BaseStrategy:
        tracked = self.state.strategies.get(strategy_name)
        merged_params = dict(tracked.optimal_params) if tracked else {}
        if params:
            merged_params.update(params)
        return build_strategy(strategy_name, merged_params or None)

    def build_tracked_strategies(
        self,
        include_dead: bool = False,
        *,
        include_lineage: bool = False,
    ) -> list[BaseStrategy]:
        names = self.tracked_names(include_lineage=include_lineage)
        if not include_dead:
            names = [
                name for name in names
                if self.state.strategies[name].status is not StrategyStatus.DEAD
            ]
        return [self.build(name) for name in names]

    def build_validation_strategies(self) -> list[BaseStrategy]:
        return [self.build(item.strategy) for item in self.validation_set()]

    def catalog_entry(self, strategy_name: str, params: dict | None = None) -> StrategyCatalogEntry:
        tracked = self.state.strategies.get(strategy_name)
        strategy = self.build(strategy_name, params)
        strategy_config = strategy.strategy_config()
        search_spec = strategy.search_spec or StrategySearchSpec.from_parameter_space(
            strategy.parameter_space,
            strategy_config=strategy_config,
            objective=ObjectiveSpec(
                primary_metric="avg_test_exp_r",
                minimum_signals=20,
                tie_breakers=["pct_positive_oos_windows", "oos_signals"],
            ),
        )
        return StrategyCatalogEntry(
            name=strategy.name,
            status=tracked.status if tracked is not None else StrategyStatus.CANDIDATE,
            tickers=list(tracked.tickers) if tracked is not None else [],
            directions=list(tracked.directions) if tracked is not None else [],
            evaluation_mode=strategy.evaluation_mode,
            required_features=sorted(required_feature_union([strategy])),
            parameter_space={key: list(values) for key, values in strategy.parameter_space.items()},
            search_spec=search_spec,
            strategy_config=strategy_config,
            notes=tracked.notes if tracked is not None else "",
            evidence=tracked.evidence if tracked is not None else "",
        )

    def validation_entries(self) -> list[StrategyCatalogEntry]:
        entries: list[StrategyCatalogEntry] = []
        for item in self.validation_set():
            entry = self.catalog_entry(item.strategy)
            entry.tickers = list(item.representative_tickers)
            entry.directions = list(item.expected_directions)
            entry.notes = item.minimum_smoke_test
            entry.evidence = item.why_it_matters
            entries.append(entry)
        return entries
