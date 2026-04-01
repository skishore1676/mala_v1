"""Registry helpers for tracked strategies and validation fixtures."""

from __future__ import annotations

from pathlib import Path

from src.research.models import StrategyCatalogEntry, StrategyStatus, ValidationStrategy
from src.research.state import load_research_state
from src.strategy.base import BaseStrategy, required_feature_union
from src.strategy.factory import build_strategy


class ResearchRegistry:
    """Bridge repo memory to runtime strategy instances."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state = load_research_state(state_path)

    def tracked_names(self) -> list[str]:
        return sorted(self.state.strategies)

    def validation_set(self) -> list[ValidationStrategy]:
        return list(self.state.validation)

    def tracked_entries(self, include_dead: bool = False) -> list[StrategyCatalogEntry]:
        names = self.tracked_names()
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

    def build_tracked_strategies(self, include_dead: bool = False) -> list[BaseStrategy]:
        names = self.tracked_names()
        if not include_dead:
            names = [
                name for name in names
                if self.state.strategies[name].status is not StrategyStatus.DEAD
            ]
        return [self.build(name) for name in names]

    def build_validation_strategies(self) -> list[BaseStrategy]:
        return [self.build(item.strategy) for item in self.validation_set()]

    def catalog_entry(self, strategy_name: str, params: dict | None = None) -> StrategyCatalogEntry:
        tracked = self.state.strategies[strategy_name]
        strategy = self.build(strategy_name, params)
        return StrategyCatalogEntry(
            name=strategy.name,
            status=tracked.status,
            tickers=list(tracked.tickers),
            directions=list(tracked.directions),
            evaluation_mode=strategy.evaluation_mode,
            required_features=sorted(required_feature_union([strategy])),
            parameter_space={key: list(values) for key, values in strategy.parameter_space.items()},
            strategy_config=strategy.strategy_config(),
            notes=tracked.notes,
            evidence=tracked.evidence,
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
