"""Underlying-anchored thesis exit optimization for validated research candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from src.oracle.trade_simulator import (
    FixedPercentRewardRiskExitPolicy,
    TradeSimulator,
    VmaTrailingExitPolicy,
)
from src.strategy.base import BaseStrategy


class ExitPolicyEvaluation(BaseModel):
    policy_name: str
    thesis_exit_anchor: str = "underlying"
    thesis_exit_policy: str
    thesis_exit_params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExitOptimizationResult(BaseModel):
    generated_at: str
    strategy_key: str
    symbol: str
    direction: str
    selection_metric: str = "expectancy"
    selection_slice: dict[str, str]
    selected_policy_name: str
    thesis_exit_anchor: str = "underlying"
    thesis_exit_policy: str
    thesis_exit_params: dict[str, Any] = Field(default_factory=dict)
    catastrophe_exit_anchor: str = "option_premium"
    catastrophe_exit_params: dict[str, Any] = Field(default_factory=dict)
    selected_metrics: dict[str, Any] = Field(default_factory=dict)
    candidate_policies: list[ExitPolicyEvaluation] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PolicyCandidate:
    name: str
    thesis_exit_policy: str
    thesis_exit_params: dict[str, Any]
    simulator: TradeSimulator


_FIXED_RR_POLICY_GRID: dict[str, list[tuple[float, float]]] = {
    "market_impulse": [(0.0035, 1.5), (0.005, 2.0), (0.0075, 2.0)],
    "jerk_pivot_momentum": [(0.0025, 1.25), (0.0035, 1.5), (0.005, 1.75)],
    "elastic_band_reversion": [(0.0035, 1.0), (0.005, 1.5), (0.0075, 2.0)],
    "opening_drive_classifier": [(0.0035, 1.25), (0.005, 1.5), (0.0075, 2.0)],
}


def optimize_underlying_exit(
    *,
    strategy_key: str,
    symbol: str,
    direction: str,
    strategy: BaseStrategy,
    enriched_frame: pl.DataFrame,
    holdout_start: date,
    holdout_end: date,
    catastrophe_exit_params: dict[str, Any],
) -> ExitOptimizationResult | None:
    if enriched_frame.is_empty():
        return None

    signal_frame = strategy.generate_signals(enriched_frame.clone())
    filtered = _holdout_signal_frame(
        signal_frame=signal_frame,
        direction=direction,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    if filtered.is_empty():
        return None

    candidates = _policy_candidates(strategy_key=strategy_key, strategy=strategy)
    evaluations: list[ExitPolicyEvaluation] = []
    selected: ExitPolicyEvaluation | None = None
    for candidate in candidates:
        result = candidate.simulator.simulate(filtered)
        metrics = _simulation_metrics(result=result)
        if metrics["trade_count"] <= 0:
            continue
        evaluation = ExitPolicyEvaluation(
            policy_name=candidate.name,
            thesis_exit_policy=candidate.thesis_exit_policy,
            thesis_exit_params=candidate.thesis_exit_params,
            metrics=metrics,
        )
        evaluations.append(evaluation)
        if selected is None or _evaluation_sort_key(evaluation) > _evaluation_sort_key(selected):
            selected = evaluation

    if selected is None:
        return None

    return ExitOptimizationResult(
        generated_at=datetime.now(UTC).isoformat(),
        strategy_key=strategy_key,
        symbol=symbol.upper(),
        direction=direction,
        selection_slice={
            "holdout_start": holdout_start.isoformat(),
            "holdout_end": holdout_end.isoformat(),
        },
        selected_policy_name=selected.policy_name,
        thesis_exit_policy=selected.thesis_exit_policy,
        thesis_exit_params=selected.thesis_exit_params,
        catastrophe_exit_params=catastrophe_exit_params,
        selected_metrics=selected.metrics,
        candidate_policies=evaluations,
    )


def write_exit_optimization_result(
    result: ExitOptimizationResult,
    *,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_exit_optimization_result(path: str | Path) -> ExitOptimizationResult:
    return ExitOptimizationResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _holdout_signal_frame(
    *,
    signal_frame: pl.DataFrame,
    direction: str,
    holdout_start: date,
    holdout_end: date,
) -> pl.DataFrame:
    filtered = signal_frame.filter(
        pl.col("timestamp").dt.date().is_between(holdout_start, holdout_end, closed="both")
    )
    if filtered.is_empty():
        return filtered
    selected_direction = direction.lower()
    if "signal_direction" in filtered.columns:
        filtered = filtered.with_columns(
            (
                pl.col("signal")
                & (pl.col("signal_direction").str.to_lowercase() == selected_direction)
            ).fill_null(False).alias("signal"),
            pl.when(pl.col("signal_direction").str.to_lowercase() == selected_direction)
            .then(pl.col("signal_direction"))
            .otherwise(pl.lit(None))
            .alias("signal_direction"),
        )
    return filtered


def _policy_candidates(*, strategy_key: str, strategy: BaseStrategy) -> list[_PolicyCandidate]:
    candidates: list[_PolicyCandidate] = []
    if strategy_key == "market_impulse":
        vma_col = getattr(strategy, "vma_col", "vma_10")
        candidates.append(
            _PolicyCandidate(
                name=f"trailing_vma_underlying:{vma_col}",
                thesis_exit_policy="trailing_vma_underlying",
                thesis_exit_params={"vma_col": vma_col},
                simulator=TradeSimulator(exit_policy=VmaTrailingExitPolicy(vma_col=vma_col, policy_name="trailing_vma_underlying")),
            )
        )
    for stop_loss_pct, reward_multiple in _FIXED_RR_POLICY_GRID.get(strategy_key, []):
        policy_name = f"fixed_rr_underlying:{stop_loss_pct:.4f}x{reward_multiple:.2f}"
        candidates.append(
            _PolicyCandidate(
                name=policy_name,
                thesis_exit_policy="fixed_rr_underlying",
                thesis_exit_params={
                    "stop_loss_underlying_pct": stop_loss_pct,
                    "take_profit_underlying_r_multiple": reward_multiple,
                },
                simulator=TradeSimulator(
                    exit_policy=FixedPercentRewardRiskExitPolicy(
                        stop_loss_pct=stop_loss_pct,
                        reward_multiple=reward_multiple,
                    )
                ),
            )
        )
    return candidates


def _simulation_metrics(*, result: Any) -> dict[str, Any]:
    return {
        "trade_count": int(result.total_trades),
        "win_rate": float(result.win_rate),
        "expectancy": float(result.expectancy),
        "profit_factor": float(result.profit_factor),
        "total_pnl": float(result.total_pnl),
        "avg_winner": float(result.avg_winner),
        "avg_loser": float(result.avg_loser),
    }


def _evaluation_sort_key(evaluation: ExitPolicyEvaluation) -> tuple[float, float, float, float]:
    metrics = evaluation.metrics
    return (
        float(metrics.get("expectancy") or float("-inf")),
        float(metrics.get("profit_factor") or float("-inf")),
        float(metrics.get("win_rate") or float("-inf")),
        float(metrics.get("trade_count") or 0.0),
    )

