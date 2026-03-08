"""
Abstract base class for all strategy agents.

Every strategy receives a physics-enriched DataFrame and must return
a boolean signal column indicating where setups trigger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


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

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
