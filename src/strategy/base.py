"""
Abstract base class for all strategy agents.

Every strategy receives a physics-enriched DataFrame and must return
a boolean signal column indicating where setups trigger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import time
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from src.research.models import StrategySearchSpec


class BaseStrategy(ABC):
    """Interface that every strategy must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Accept a physics-enriched DataFrame.
        Return the same DataFrame with an added boolean column named "signal".
        """
        ...

    @property
    def required_features(self) -> set[str]:
        """Columns this strategy expects on the input frame."""
        return set()

    @property
    def feature_requests(self) -> set[str]:
        """Additional Newton feature requests that are not final DataFrame columns."""
        return set()

    @property
    def parameter_space(self) -> dict[str, list[Any]]:
        """Optional bounded search space for orchestrators and agents."""
        return {}

    @property
    def evaluation_mode(self) -> str:
        """Evaluation family used by research orchestration."""
        return "binary"

    @property
    def search_spec(self) -> StrategySearchSpec | None:
        """Optional compact search metadata for agent-native research loops."""
        return None

    def strategy_config(self) -> dict[str, Any]:
        """Serializable runtime configuration for registry and orchestration use."""
        return {}

    def search_config(self) -> dict[str, Any]:
        """Canonical config used to dedupe equivalent research search cells."""
        return self.strategy_config()

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"


def required_feature_union(strategies: Iterable[BaseStrategy]) -> set[str]:
    """Return the union of declared feature dependencies for the given strategies."""
    required: set[str] = set()
    for strategy in strategies:
        required |= set(strategy.required_features)
        required |= set(strategy.feature_requests)
    return required


def coerce_time(value: time | str) -> time:
    """Accept time objects or HH:MM-style strings from serialized strategy configs."""
    if isinstance(value, time):
        return value
    return time.fromisoformat(value)
