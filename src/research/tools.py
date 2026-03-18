"""Callable research tools for bounded experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice, product
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.research.models import ResearchStage, StrategyCatalogEntry
from src.research.registry import ResearchRegistry
from src.research.stages import (
    aggregate_walk_forward,
    build_gate_report,
    run_execution_mapping_for_candidates,
    run_holdout_validation_for_candidates,
    run_walk_forward_for_strategies,
    summarize_holdout,
)
from src.research.stages.holdout import choose_ratio, eval_direction


@dataclass(slots=True)
class ResearchToolResult:
    tool_name: str
    summary: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)


def _bounded_param_grid(
    parameter_space: dict[str, list[Any]],
    *,
    max_configs: int,
) -> list[dict[str, Any]]:
    if not parameter_space:
        return [{}]

    keys = sorted(parameter_space)
    iterables = [parameter_space[key] for key in keys]
    configs: list[dict[str, Any]] = []
    for values in islice(product(*iterables), max_configs):
        configs.append(dict(zip(keys, values, strict=True)))
    return configs


class ResearchToolbox:
    """Python-callable tools that an experiment agent can invoke directly."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.registry = ResearchRegistry(state_path)

    def available_tools(self, stage: ResearchStage) -> list[str]:
        stage_tools = {
            ResearchStage.M1_DISCOVERY: ["parameter_sweep", "baseline_comparison"],
            ResearchStage.M2_CONVERGENCE: ["convergence_grid", "ablation_check"],
            ResearchStage.M3_WALK_FORWARD: ["walk_forward"],
            ResearchStage.M4_HOLDOUT: ["holdout_validation"],
            ResearchStage.M5_EXECUTION: ["execution_mapping"],
        }
        return stage_tools[stage]

    def parameter_sweep(self, strategy_name: str, max_configs: int = 16) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        configs = _bounded_param_grid(entry.parameter_space, max_configs=max_configs)
        built_names = [self.registry.build(strategy_name, params).name for params in configs]
        return ResearchToolResult(
            tool_name="parameter_sweep",
            summary={
                "strategy": entry.name,
                "config_count": len(configs),
                "parameter_count": len(entry.parameter_space),
                "max_configs": max_configs,
            },
            artifacts={
                "catalog_entry": entry,
                "configs": configs,
                "strategy_names": built_names,
            },
        )

    def baseline_comparison(self, strategy_name: str) -> ResearchToolResult:
        candidate = self.registry.catalog_entry(strategy_name)
        comparisons: list[dict[str, Any]] = []
        for baseline in self.registry.validation_entries():
            candidate_features = set(candidate.required_features)
            baseline_features = set(baseline.required_features)
            comparisons.append(
                {
                    "baseline": baseline.name,
                    "same_eval_mode": candidate.evaluation_mode == baseline.evaluation_mode,
                    "feature_overlap": sorted(candidate_features & baseline_features),
                    "candidate_only_features": sorted(candidate_features - baseline_features),
                    "baseline_only_features": sorted(baseline_features - candidate_features),
                    "shared_parameters": sorted(
                        set(candidate.parameter_space) & set(baseline.parameter_space)
                    ),
                }
            )

        return ResearchToolResult(
            tool_name="baseline_comparison",
            summary={
                "strategy": candidate.name,
                "baseline_count": len(comparisons),
            },
            artifacts={"comparisons": comparisons},
        )

    def ablation_check(self, strategy_name: str, max_variants: int = 8) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        base_config = entry.strategy_config
        variants: list[dict[str, Any]] = []

        for param_name, candidates in entry.parameter_space.items():
            current_value = base_config.get(param_name)
            alt_values = [value for value in candidates if value != current_value]
            for alt_value in alt_values[:1]:
                ablated = dict(base_config)
                ablated[param_name] = alt_value
                variants.append(
                    {
                        "parameter": param_name,
                        "base_value": current_value,
                        "test_value": alt_value,
                        "config": ablated,
                    }
                )
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break

        return ResearchToolResult(
            tool_name="ablation_check",
            summary={
                "strategy": entry.name,
                "variant_count": len(variants),
                "supports_parameter_ablation": bool(entry.parameter_space),
            },
            artifacts={"variants": variants},
        )

    def walk_forward(
        self,
        *,
        ticker: str,
        df: pl.DataFrame,
        strategies: Iterable,
        windows,
        ratios: list[float],
        metrics: MetricsCalculator,
        min_signals: int,
        cost_r: float | None = None,
        cost_bps: float | None = None,
    ) -> ResearchToolResult:
        rows = run_walk_forward_for_strategies(
            ticker=ticker,
            df=df,
            strategies=strategies,
            windows=windows,
            ratios=ratios,
            metrics=metrics,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
        )
        detail_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        aggregate_df = aggregate_walk_forward(rows) if rows else pl.DataFrame()
        return ResearchToolResult(
            tool_name="walk_forward",
            summary={
                "ticker": ticker,
                "detail_rows": detail_df.height,
                "aggregate_rows": aggregate_df.height,
            },
            artifacts={"detail": detail_df, "aggregate": aggregate_df},
        )

    def convergence_grid(
        self,
        *,
        combined: pl.DataFrame,
        cost_count: int,
        gate_min_oos_windows: int,
        gate_min_oos_signals: int,
        gate_min_pct_positive: float,
        gate_min_exp_r: float,
    ) -> ResearchToolResult:
        gate_report = build_gate_report(
            combined=combined,
            cost_count=cost_count,
            gate_min_oos_windows=gate_min_oos_windows,
            gate_min_oos_signals=gate_min_oos_signals,
            gate_min_pct_positive=gate_min_pct_positive,
            gate_min_exp_r=gate_min_exp_r,
        )
        promoted = gate_report.filter(pl.col("passes_all_gates")).height
        return ResearchToolResult(
            tool_name="convergence_grid",
            summary={
                "candidate_count": gate_report.height,
                "promoted_count": promoted,
            },
            artifacts={"gate_report": gate_report},
        )

    def holdout_validation(
        self,
        *,
        promoted: pl.DataFrame,
        ticker_frames: dict[str, pl.DataFrame],
        metrics: MetricsCalculator,
        start_date,
        calibration_end,
        holdout_start,
        holdout_end,
        ratios: list[float],
        costs: list[float],
        min_calibration_signals: int,
        min_holdout_signals: int,
    ) -> ResearchToolResult:
        detail_rows = run_holdout_validation_for_candidates(
            promoted=promoted,
            ticker_frames=ticker_frames,
            metrics=metrics,
            start_date=start_date,
            calibration_end=calibration_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            ratios=ratios,
            costs=costs,
            min_calibration_signals=min_calibration_signals,
            min_holdout_signals=min_holdout_signals,
        )
        detail_df = pl.DataFrame(detail_rows) if detail_rows else pl.DataFrame()
        summary_df = summarize_holdout(detail_df, cost_count=len(costs)) if detail_rows else pl.DataFrame()
        return ResearchToolResult(
            tool_name="holdout_validation",
            summary={
                "candidate_count": promoted.height,
                "detail_rows": detail_df.height,
                "promoted_count": summary_df.filter(
                    pl.col("decision") == "promote_to_execution_mapping"
                ).height if not summary_df.is_empty() else 0,
            },
            artifacts={"detail": detail_df, "summary": summary_df},
        )

    def execution_mapping(
        self,
        *,
        promoted: pl.DataFrame,
        holdout_detail: pl.DataFrame,
        ticker_frames: dict[str, pl.DataFrame],
        metrics: MetricsCalculator,
        holdout_start,
        holdout_end,
        base_cost_r: float,
        stress_cfg: ExecutionStressConfig,
    ) -> ResearchToolResult:
        rows = run_execution_mapping_for_candidates(
            promoted=promoted,
            holdout_detail=holdout_detail,
            ticker_frames=ticker_frames,
            metrics=metrics,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            base_cost_r=base_cost_r,
            stress_cfg=stress_cfg,
        )
        detail_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        return ResearchToolResult(
            tool_name="execution_mapping",
            summary={
                "candidate_count": promoted.height,
                "mapped_count": detail_df.height,
            },
            artifacts={"detail": detail_df},
        )

    def evaluate_direction(
        self,
        *,
        df_eval: pl.DataFrame,
        direction: str,
        ratio: float,
        cost_bps: float,
        min_signals: int = 1,
    ) -> ResearchToolResult:
        stats = eval_direction(df_eval, direction, ratio, cost_bps)
        passes_min_signals = int(stats["signals"]) >= min_signals
        selected_ratio, chosen_stats = choose_ratio(
            calib_df=df_eval,
            direction=direction,
            ratios=[ratio],
            cost_bps=cost_bps,
            min_calib_signals=min_signals,
        )
        return ResearchToolResult(
            tool_name="evaluate_direction",
            summary={
                "direction": direction,
                "ratio": ratio,
                "passes_min_signals": passes_min_signals,
                "selected_ratio": selected_ratio,
            },
            artifacts={"stats": stats, "chosen_stats": chosen_stats},
        )


__all__ = ["ResearchToolResult", "ResearchToolbox"]
