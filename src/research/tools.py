"""Callable research tools for bounded experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from loguru import logger

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.research.models import ResearchStage, StrategyCatalogEntry
from src.research.registry import ResearchRegistry
from src.research.stages import (
    aggregate_walk_forward,
    build_gate_report,
    build_windows,
    run_execution_mapping_for_candidates,
    run_holdout_validation_for_candidates,
    run_walk_forward_for_strategies,
    summarize_holdout,
)
from src.research.stages.holdout import choose_ratio, eval_direction
from src.strategy.base import BaseStrategy, required_feature_union


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
    all_configs = [dict(zip(keys, values, strict=True)) for values in product(*iterables)]
    if len(all_configs) <= max_configs:
        return all_configs
    if max_configs <= 1:
        return [all_configs[0]]

    # Sample evenly across the Cartesian product instead of truncating to the
    # first N lexicographic configs, which can over-focus on one parameter slice.
    indices = sorted(
        {
            round(i * (len(all_configs) - 1) / (max_configs - 1))
            for i in range(max_configs)
        }
    )
    return [all_configs[index] for index in indices]


def _config_columns(configs: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for config in configs:
        keys.update(config)
    return sorted(keys)


class ResearchToolbox:
    """Python-callable tools that an experiment agent can invoke directly."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.registry = ResearchRegistry(state_path)
        self._storage = LocalStorage()
        self._physics = PhysicsEngine()

    def available_tools(self, stage: ResearchStage) -> list[str]:
        stage_tools = {
            ResearchStage.M1_DISCOVERY: ["parameter_sweep", "baseline_comparison"],
            ResearchStage.M2_CONVERGENCE: ["convergence_grid", "ablation_check"],
            ResearchStage.M3_WALK_FORWARD: ["walk_forward"],
            ResearchStage.M4_HOLDOUT: ["holdout_validation"],
            ResearchStage.M5_EXECUTION: ["execution_mapping"],
        }
        return stage_tools[stage]

    def invoke(self, tool_name: str, /, **kwargs: Any) -> ResearchToolResult:
        method = getattr(self, tool_name, None)
        if method is None or tool_name.startswith("_"):
            raise ValueError(f"Unknown research tool: {tool_name}")
        result = method(**kwargs)
        if not isinstance(result, ResearchToolResult):
            raise TypeError(f"Research tool {tool_name!r} did not return ResearchToolResult")
        return result

    def parameter_sweep(
        self,
        strategy_name: str,
        max_configs: int = 16,
        *,
        parameter_space: dict[str, list[Any]] | None = None,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        train_months: int = 6,
        test_months: int = 3,
        ratios: list[float] | None = None,
        min_signals: int = 20,
        cost_r: float | None = None,
        cost_bps: float | None = None,
        min_total_signals: int | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        sweep_space = parameter_space or entry.parameter_space
        configs = _bounded_param_grid(sweep_space, max_configs=max_configs)
        strategies = [self.registry.build(strategy_name, params) for params in configs]

        result = self._strategy_sweep_result(
            tool_name="parameter_sweep",
            strategy_name=strategy_name,
            entry=entry,
            configs=configs,
            strategies=strategies,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            ratios=ratios,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
            min_total_signals=min_total_signals,
            ticker_frames=ticker_frames,
            metrics=metrics,
        )
        result.summary["parameter_count"] = len(sweep_space)
        return result

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

    def ablation_check(
        self,
        strategy_name: str,
        max_variants: int = 8,
        *,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        train_months: int = 6,
        test_months: int = 3,
        ratios: list[float] | None = None,
        min_signals: int = 20,
        cost_r: float | None = None,
        cost_bps: float | None = None,
        min_total_signals: int | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        base_config = entry.strategy_config
        variants: list[dict[str, Any]] = []

        for param_name, candidates in entry.parameter_space.items():
            current_value = base_config.get(param_name)
            alt_values = [value for value in candidates if value != current_value]
            for alt_value in alt_values[:1]:
                ablated = dict(base_config)
                ablated[param_name] = alt_value
                variants.append(ablated)
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break

        strategies = [self.registry.build(strategy_name, params) for params in variants]
        result = self._strategy_sweep_result(
            tool_name="ablation_check",
            strategy_name=strategy_name,
            entry=entry,
            configs=variants,
            strategies=strategies,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            ratios=ratios,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
            min_total_signals=min_total_signals,
            ticker_frames=ticker_frames,
            metrics=metrics,
        )
        result.summary["supports_parameter_ablation"] = bool(entry.parameter_space)
        result.summary["variant_count"] = len(variants)
        return result

    def walk_forward(
        self,
        *,
        ticker: str,
        df: pl.DataFrame,
        strategies: Iterable[BaseStrategy],
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
        summary_df = (
            summarize_holdout(detail_df, cost_count=len(costs))
            if detail_rows
            else pl.DataFrame()
        )
        return ResearchToolResult(
            tool_name="holdout_validation",
            summary={
                "candidate_count": promoted.height,
                "detail_rows": detail_df.height,
                "promoted_count": summary_df.filter(
                    pl.col("decision") == "promote_to_execution_mapping"
                ).height
                if not summary_df.is_empty()
                else 0,
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

    def retune_search(
        self,
        *,
        candidates: list[dict[str, Any]],
        ratio: float,
        cost_r: float,
        bootstrap_iters: int,
        min_signals: int,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        strategies = [candidate["strategy"] for candidate in candidates]
        metrics = metrics or MetricsCalculator()
        enriched_frames = self._load_enriched_frames(
            strategies=strategies,
            tickers=tickers or sorted(ticker_frames or {}),
            start_date=start_date,
            end_date=end_date,
            ticker_frames=ticker_frames,
        )

        rng = np.random.default_rng(7)
        rows: list[dict[str, Any]] = []
        for ticker, df in enriched_frames.items():
            for candidate in candidates:
                strategy = candidate["strategy"]
                df_eval = metrics.add_directional_forward_metrics(
                    strategy.generate_signals(df.clone()),
                    snapshot_windows=(30, 60),
                )
                scored = self._evaluate_retune_frame(
                    df_eval=df_eval,
                    ratio=ratio,
                    cost_r=cost_r,
                    bootstrap_iters=bootstrap_iters,
                    min_signals=min_signals,
                    rng=rng,
                )
                for row in scored["rows"]:
                    rows.append(
                        {
                            "ticker": ticker,
                            "strategy": candidate["strategy_name"],
                            "params": candidate["params_label"],
                            "direction": row["direction"],
                            "signals": row["signals"],
                            "confidence": row["confidence"],
                            "exp_r": row["exp_r"],
                            "prob_pos_exp": row["prob_pos_exp"],
                            "objective": scored["objective"],
                            **candidate.get("metadata", {}),
                        }
                    )

        detail_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        ranked_df = (
            detail_df.filter(pl.col("direction") == "combined")
            .sort(["exp_r", "prob_pos_exp", "signals"], descending=[True, True, True])
            if not detail_df.is_empty()
            else pl.DataFrame()
        )
        return ResearchToolResult(
            tool_name="retune_search",
            summary={
                "candidate_count": len(candidates),
                "ticker_count": len(enriched_frames),
                "detail_rows": detail_df.height,
                "ranked_rows": ranked_df.height,
            },
            artifacts={"detail": detail_df, "ranked": ranked_df},
        )

    def _strategy_sweep_result(
        self,
        *,
        tool_name: str,
        strategy_name: str,
        entry: StrategyCatalogEntry,
        configs: list[dict[str, Any]],
        strategies: list[BaseStrategy],
        tickers: list[str] | None,
        start_date: date | None,
        end_date: date | None,
        train_months: int,
        test_months: int,
        ratios: list[float] | None,
        min_signals: int,
        cost_r: float | None,
        cost_bps: float | None,
        min_total_signals: int | None,
        ticker_frames: dict[str, pl.DataFrame] | None,
        metrics: MetricsCalculator | None,
    ) -> ResearchToolResult:
        built_names = [strategy.name for strategy in strategies]
        summary: dict[str, Any] = {
            "strategy": entry.name,
            "config_count": len(configs),
            "max_configs": len(configs),
        }
        artifacts: dict[str, Any] = {
            "catalog_entry": entry,
            "configs": configs,
            "strategy_names": built_names,
        }

        if not strategies or not self._has_execution_request(tickers, ticker_frames, start_date, end_date):
            return ResearchToolResult(tool_name=tool_name, summary=summary, artifacts=artifacts)

        assert start_date is not None
        assert end_date is not None
        windows = build_windows(start_date, end_date, train_months, test_months)
        if not windows:
            summary["detail_rows"] = 0
            summary["aggregate_rows"] = 0
            artifacts["detail"] = pl.DataFrame()
            artifacts["aggregate"] = pl.DataFrame()
            return ResearchToolResult(tool_name=tool_name, summary=summary, artifacts=artifacts)

        ratios = ratios or [1.0, 1.25, 1.5, 2.0]
        metrics = metrics or MetricsCalculator()
        enriched_frames = self._load_enriched_frames(
            strategies=strategies,
            tickers=tickers or sorted(ticker_frames or {}),
            start_date=start_date,
            end_date=end_date,
            ticker_frames=ticker_frames,
        )
        detail_df = self._run_strategy_sweep(
            strategy_name=strategy_name,
            strategies=strategies,
            configs=configs,
            ticker_frames=enriched_frames,
            windows=windows,
            ratios=ratios,
            metrics=metrics,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
        )
        aggregate_df = self._aggregate_sweep(detail_df, configs, min_total_signals=min_total_signals)
        summary.update(
            {
                "ticker_count": len(enriched_frames),
                "detail_rows": detail_df.height,
                "aggregate_rows": aggregate_df.height,
            }
        )
        artifacts["detail"] = detail_df
        artifacts["aggregate"] = aggregate_df
        return ResearchToolResult(tool_name=tool_name, summary=summary, artifacts=artifacts)

    @staticmethod
    def _has_execution_request(
        tickers: list[str] | None,
        ticker_frames: dict[str, pl.DataFrame] | None,
        start_date: date | None,
        end_date: date | None,
    ) -> bool:
        has_tickers = bool(tickers) or bool(ticker_frames)
        return has_tickers and start_date is not None and end_date is not None

    def _load_enriched_frames(
        self,
        *,
        strategies: list[BaseStrategy],
        tickers: list[str],
        start_date: date,
        end_date: date,
        ticker_frames: dict[str, pl.DataFrame] | None,
    ) -> dict[str, pl.DataFrame]:
        needed_features = required_feature_union(strategies)
        enriched: dict[str, pl.DataFrame] = {}
        for ticker in tickers:
            raw = ticker_frames.get(ticker) if ticker_frames is not None else None
            if raw is None:
                raw = self._storage.load_bars(ticker, start_date, end_date)
            else:
                self._warn_on_frame_date_mismatch(
                    ticker=ticker,
                    frame=raw,
                    start_date=start_date,
                    end_date=end_date,
                )
            if raw.is_empty():
                continue
            enriched[ticker] = self._physics.enrich_for_features(raw, needed_features)
        return enriched

    @staticmethod
    def _warn_on_frame_date_mismatch(
        *,
        ticker: str,
        frame: pl.DataFrame,
        start_date: date,
        end_date: date,
    ) -> None:
        if frame.is_empty() or "timestamp" not in frame.columns:
            return
        bounds = frame.select([
            pl.col("timestamp").dt.date().min().alias("frame_start"),
            pl.col("timestamp").dt.date().max().alias("frame_end"),
        ]).row(0, named=True)
        frame_start = bounds["frame_start"]
        frame_end = bounds["frame_end"]
        if frame_start is None or frame_end is None:
            return
        if frame_start > start_date or frame_end < end_date:
            logger.warning(
                "Injected ticker frame for {} covers {} -> {}, but requested research window is {} -> {}",
                ticker,
                frame_start,
                frame_end,
                start_date,
                end_date,
            )

    @staticmethod
    def _run_strategy_sweep(
        *,
        strategy_name: str,
        strategies: list[BaseStrategy],
        configs: list[dict[str, Any]],
        ticker_frames: dict[str, pl.DataFrame],
        windows,
        ratios: list[float],
        metrics: MetricsCalculator,
        min_signals: int,
        cost_r: float | None,
        cost_bps: float | None,
    ) -> pl.DataFrame:
        detail_rows: list[dict[str, Any]] = []
        for strategy, config in zip(strategies, configs, strict=True):
            for ticker, df in ticker_frames.items():
                rows = run_walk_forward_for_strategies(
                    ticker=ticker,
                    df=df,
                    strategies=[strategy],
                    windows=windows,
                    ratios=ratios,
                    metrics=metrics,
                    min_signals=min_signals,
                    cost_r=cost_r,
                    cost_bps=cost_bps,
                )
                for row in rows:
                    detail_rows.append(
                        {
                            **row,
                            "catalog_strategy": strategy_name,
                            "base_strategy": strategy.__class__.__name__,
                            **config,
                        }
                    )
        return pl.DataFrame(detail_rows) if detail_rows else pl.DataFrame()

    @staticmethod
    def _aggregate_sweep(
        detail_df: pl.DataFrame,
        configs: list[dict[str, Any]],
        *,
        min_total_signals: int | None,
    ) -> pl.DataFrame:
        if detail_df.is_empty():
            return pl.DataFrame()

        config_cols = [column for column in _config_columns(configs) if column in detail_df.columns]
        context_cols = [
            column for column in ("catalog_strategy", "base_strategy") if column in detail_df.columns
        ]
        group_cols = ["ticker", "strategy", "direction", *context_cols, *config_cols]
        aggregate_df = (
            detail_df.group_by(group_cols)
            .agg([
                pl.len().alias("oos_windows"),
                pl.col("test_signals").sum().alias("total_oos_signals"),
                pl.col("test_exp_r").mean().alias("avg_oos_exp_r"),
                pl.col("test_exp_r").median().alias("med_oos_exp_r"),
                (pl.col("test_exp_r") > 0).mean().alias("pct_positive_windows"),
                pl.col("test_confidence").mean().alias("avg_confidence"),
                pl.col("effective_cost_r").mean().alias("avg_effective_cost_r"),
                pl.col("selected_ratio").median().alias("median_selected_ratio"),
            ])
        )
        if min_total_signals is not None:
            aggregate_df = aggregate_df.with_columns(
                pl.when(pl.col("total_oos_signals") >= min_total_signals)
                .then(pl.lit("valid"))
                .otherwise(pl.lit("low_n"))
                .alias("signal_quality")
            )
        aggregate_df = aggregate_df.with_columns([
            pl.col("total_oos_signals").alias("oos_signals"),
            pl.col("avg_oos_exp_r").alias("avg_test_exp_r"),
            pl.col("pct_positive_windows").alias("pct_positive_oos_windows"),
            pl.col("avg_confidence").alias("avg_test_confidence"),
        ])
        return aggregate_df.sort(["ticker", "direction", "avg_oos_exp_r"], descending=[False, False, True])

    @staticmethod
    def _evaluate_retune_frame(
        *,
        df_eval: pl.DataFrame,
        ratio: float,
        cost_r: float,
        bootstrap_iters: int,
        min_signals: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        base = (
            df_eval.filter(pl.col("signal"))
            .drop_nulls(subset=["forward_mfe_eod", "forward_mae_eod", "signal_direction"])
        )

        rows: list[dict[str, Any]] = []
        for direction in ("combined", "long", "short"):
            subset = base if direction == "combined" else base.filter(pl.col("signal_direction") == direction)
            mfe = subset["forward_mfe_eod"].to_numpy() if len(subset) else np.array([])
            mae = subset["forward_mae_eod"].to_numpy() if len(subset) else np.array([])
            stats = ResearchToolbox._score_retune_group(
                mfe=mfe,
                mae=mae,
                ratio=ratio,
                cost_r=cost_r,
                bootstrap_iters=bootstrap_iters,
                rng=rng,
            )
            if stats["signals"] >= min_signals:
                rows.append({"direction": direction, **stats})

        combined = next((row for row in rows if row["direction"] == "combined"), None)
        if combined is None:
            return {"rows": rows, "objective": -1e9}

        objective = (
            float(combined["exp_r"]) * 1000
            + float(combined["prob_pos_exp"]) * 100
            + float(combined["signals"]) / 1000
        )
        return {"rows": rows, "objective": objective}

    @staticmethod
    def _score_retune_group(
        *,
        mfe: np.ndarray,
        mae: np.ndarray,
        ratio: float,
        cost_r: float,
        bootstrap_iters: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        n = len(mfe)
        if n == 0:
            return {"signals": 0, "confidence": None, "exp_r": None, "prob_pos_exp": None}

        from src.oracle.policies import RewardRiskWinCondition

        policy = RewardRiskWinCondition(ratio=ratio)
        confidence = policy.confidence(mfe, mae)
        exp_r = policy.expectancy(mfe, mae, cost_r)
        p_boot = rng.binomial(n=n, p=confidence, size=bootstrap_iters) / n
        exp_boot = p_boot * ratio - (1.0 - p_boot) - cost_r

        return {
            "signals": n,
            "confidence": round(confidence, 4),
            "exp_r": round(exp_r, 4),
            "prob_pos_exp": round(float(np.mean(exp_boot > 0.0)), 4),
        }


__all__ = ["ResearchToolResult", "ResearchToolbox"]
