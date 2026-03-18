"""Reusable evaluation policies for research and metrics layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


def _ratio_label(ratio: float) -> str:
    if float(ratio).is_integer():
        return f"{int(ratio)}to1"
    return f"{ratio:g}".replace(".", "p") + "to1"


@dataclass(frozen=True, slots=True)
class RewardRiskWinCondition:
    """A simple reward:risk win definition that can be injected into evaluators."""

    ratio: float = 2.0
    label: str | None = None

    @property
    def label_suffix(self) -> str:
        return self.label or _ratio_label(self.ratio)

    def directional_column(self) -> str:
        return f"win_{self.label_suffix}"

    def flags(self, mfe: np.ndarray, mae: np.ndarray) -> np.ndarray:
        return mfe >= (self.ratio * mae)

    def expr(self, mfe_col: str, mae_col: str) -> pl.Expr:
        return (pl.col(mfe_col) >= self.ratio * pl.col(mae_col))

    def confidence(self, mfe: np.ndarray, mae: np.ndarray) -> float:
        wins = self.flags(mfe, mae)
        return float(np.mean(wins)) if len(wins) else 0.0

    def expectancy(self, mfe: np.ndarray, mae: np.ndarray, cost_r: float = 0.0) -> float:
        confidence = self.confidence(mfe, mae)
        return confidence * self.ratio - (1.0 - confidence) - cost_r
