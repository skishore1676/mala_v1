"""Callable research tools for bounded experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from time import perf_counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from loguru import logger

from src.chronos.storage import LocalStorage
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.oracle.results_db import ResultsDB
from src.research.models import ResearchStage, StrategyCatalogEntry, StrategySearchSpec
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


def _sample_configs(
    configs: list[dict[str, Any]],
    *,
    max_configs: int,
) -> list[dict[str, Any]]:
    if len(configs) <= max_configs:
        return list(configs)
    if max_configs <= 1:
        return [configs[0]]
    indices = sorted(
        {
            round(i * (len(configs) - 1) / (max_configs - 1))
            for i in range(max_configs)
        }
    )
    return [configs[index] for index in indices]


def _config_columns(configs: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for config in configs:
        keys.update(config)
    return sorted(keys)


def _freeze_for_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), _freeze_for_signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_signature(item) for item in value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _stable_signature(payload: Any) -> str:
    canonical = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _strategy_signature(strategy: BaseStrategy) -> tuple[str, Any]:
    return (strategy.__class__.__name__, _freeze_for_signature(strategy.search_config()))


def _normalize_strategy_config(
    entry: StrategyCatalogEntry,
    config: dict[str, Any],
    *,
    validate_values: bool = True,
) -> tuple[dict[str, Any], list[str], list[str]]:
    search_spec = entry.search_spec
    if search_spec is None:
        normalized = dict(entry.strategy_config)
        normalized.update(config)
        return normalized, [], []
    normalized_result = search_spec.normalize_config(
        config,
        base_config=entry.strategy_config,
        validate_values=validate_values,
    )
    return (
        normalized_result.config,
        normalized_result.inactive_parameters,
        normalized_result.errors,
    )


def _resolve_domain_index(search_spec: StrategySearchSpec | None, parameter: str, value: Any) -> int | None:
    if search_spec is None:
        return None
    parameter_map = search_spec.parameter_map()
    spec = parameter_map.get(parameter)
    if spec is None:
        return None
    legal_values = spec.legal_values()
    try:
        return legal_values.index(value)
    except ValueError:
        return None


def _config_distance(
    search_spec: StrategySearchSpec | None,
    left: dict[str, Any],
    right: dict[str, Any],
) -> int:
    if search_spec is None:
        return 0 if left == right else 1
    distance = 0
    for parameter in search_spec.parameters:
        left_value = left.get(parameter.name)
        right_value = right.get(parameter.name)
        if left_value == right_value:
            continue
        left_index = _resolve_domain_index(search_spec, parameter.name, left_value)
        right_index = _resolve_domain_index(search_spec, parameter.name, right_value)
        if left_index is None or right_index is None:
            distance += 1
            continue
        distance += abs(left_index - right_index)
    return distance


def _config_identity(config: dict[str, Any]) -> str:
    return json.dumps(_json_ready(config), sort_keys=True, separators=(",", ":"))


def _canonical_sweep_configs(
    entry: StrategyCatalogEntry,
    *,
    parameter_space_override: dict[str, list[Any]] | None,
    max_configs: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    validate_values = parameter_space_override is None
    if entry.search_spec is not None and parameter_space_override is None:
        raw_configs = _bounded_param_grid(entry.search_spec.search_space(), max_configs=10**9)
    else:
        raw_configs = _bounded_param_grid(
            parameter_space_override or entry.parameter_space,
            max_configs=10**9,
        )

    normalized_configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid_config_count = 0
    for raw_config in raw_configs:
        normalized_config, _, errors = _normalize_strategy_config(
            entry,
            raw_config,
            validate_values=validate_values,
        )
        if errors:
            invalid_config_count += 1
            continue
        identity = _config_identity(normalized_config)
        if identity in seen:
            continue
        seen.add(identity)
        normalized_configs.append(normalized_config)

    normalized_duplicate_count = len(raw_configs) - invalid_config_count - len(normalized_configs)
    return (
        _sample_configs(normalized_configs, max_configs=max_configs),
        len(raw_configs),
        invalid_config_count,
        normalized_duplicate_count,
    )


def _entry_search_space(
    entry: StrategyCatalogEntry,
    *,
    parameter_space_override: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    if parameter_space_override is not None:
        return {key: list(values) for key, values in parameter_space_override.items()}
    if entry.search_spec is not None:
        return entry.search_spec.search_space()
    return {key: list(values) for key, values in entry.parameter_space.items()}


def _normalized_slice_filter(
    *,
    ticker: str | None = None,
    slice_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _json_ready(slice_filter or {})
    if ticker is not None:
        existing_tickers = normalized.get("tickers")
        if existing_tickers is None:
            normalized["tickers"] = [ticker]
        elif isinstance(existing_tickers, list) and ticker not in existing_tickers:
            normalized["tickers"] = sorted([*existing_tickers, ticker])
    if "tickers" in normalized and isinstance(normalized["tickers"], list):
        normalized["tickers"] = sorted(str(item) for item in normalized["tickers"])
    return normalized


def _normalized_slice_payload(slice_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _json_ready(slice_payload)
    tickers = normalized.get("tickers")
    if isinstance(tickers, list):
        normalized["tickers"] = sorted(str(item) for item in tickers)
    return normalized


def _research_slice_id(
    strategy_name: str,
    slice_payload: dict[str, Any],
) -> str:
    return _stable_signature(
        {
            "strategy": strategy_name,
            "slice": _normalized_slice_payload(slice_payload),
        }
    )


def _slice_matches(payload_slice: dict[str, Any], slice_filter: dict[str, Any]) -> bool:
    if not slice_filter:
        return True
    normalized_slice = _normalized_slice_payload(payload_slice)
    for key, expected in slice_filter.items():
        actual = normalized_slice.get(key)
        if actual != expected:
            return False
    return True


def _dedupe_strategy_variants(
    configs: list[dict[str, Any]],
    strategies: list[BaseStrategy],
) -> tuple[list[dict[str, Any]], list[BaseStrategy], int]:
    deduped_configs: list[dict[str, Any]] = []
    deduped_strategies: list[BaseStrategy] = []
    seen: set[tuple[str, Any]] = set()
    duplicate_count = 0

    for config, strategy in zip(configs, strategies, strict=True):
        signature = _strategy_signature(strategy)
        if signature in seen:
            duplicate_count += 1
            continue
        seen.add(signature)
        deduped_configs.append(config)
        deduped_strategies.append(strategy)

    return deduped_configs, deduped_strategies, duplicate_count


def _annotate_plateau_metrics(
    aggregate_df: pl.DataFrame,
    *,
    parameter_space: dict[str, list[Any]],
) -> pl.DataFrame:
    if aggregate_df.is_empty():
        return aggregate_df

    config_cols = [column for column in parameter_space if column in aggregate_df.columns]
    if not config_cols:
        return aggregate_df

    order_maps = {
        key: {json.dumps(value, sort_keys=True): index for index, value in enumerate(values)}
        for key, values in parameter_space.items()
        if key in config_cols
    }
    if not order_maps:
        return aggregate_df

    group_cols = [
        column
        for column in ("ticker", "direction", "catalog_strategy", "base_strategy")
        if column in aggregate_df.columns
    ]

    rows = aggregate_df.to_dicts()
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(column) for column in group_cols), []).append(row)

    annotations: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(column) for column in group_cols)
        peers = grouped.get(key, [])
        neighbors: list[dict[str, Any]] = []
        for peer in peers:
            if peer is row:
                continue
            is_neighbor = True
            differs = False
            for column, order_map in order_maps.items():
                row_value = row.get(column)
                peer_value = peer.get(column)
                row_index = order_map.get(json.dumps(row_value, sort_keys=True))
                peer_index = order_map.get(json.dumps(peer_value, sort_keys=True))
                if row_index is None or peer_index is None:
                    if row_value != peer_value:
                        is_neighbor = False
                        break
                    continue
                distance = abs(row_index - peer_index)
                if distance > 1:
                    is_neighbor = False
                    break
                if distance > 0:
                    differs = True
            if is_neighbor and differs:
                neighbors.append(peer)

        neighbor_exp = [
            float(item["avg_test_exp_r"])
            for item in neighbors
            if item.get("avg_test_exp_r") is not None and np.isfinite(float(item["avg_test_exp_r"]))
        ]
        positive_neighbors = [value for value in neighbor_exp if value > 0]
        neighbor_count = len(neighbors)
        positive_ratio = (len(positive_neighbors) / neighbor_count) if neighbor_count else 0.0
        min_neighbor_exp = min(neighbor_exp) if neighbor_exp else None
        mean_neighbor_exp = (sum(neighbor_exp) / len(neighbor_exp)) if neighbor_exp else None
        plateau_score = None
        if min_neighbor_exp is not None:
            plateau_score = (
                float(min_neighbor_exp) * 1000
                + positive_ratio * 100
                + neighbor_count
            )
        annotations.append(
            {
                "plateau_neighbor_count": neighbor_count,
                "plateau_positive_neighbors": len(positive_neighbors),
                "plateau_positive_ratio": round(positive_ratio, 6),
                "plateau_mean_neighbor_exp_r": round(mean_neighbor_exp, 6) if mean_neighbor_exp is not None else None,
                "plateau_min_neighbor_exp_r": round(min_neighbor_exp, 6) if min_neighbor_exp is not None else None,
                "plateau_score": round(plateau_score, 6) if plateau_score is not None else None,
            }
        )

    return aggregate_df.with_columns(pl.DataFrame(annotations))


class ResearchToolbox:
    """Python-callable tools that an experiment agent can invoke directly."""

    def __init__(
        self,
        state_path: Path | None = None,
        *,
        results_db_path: Path | None = None,
    ) -> None:
        self.registry = ResearchRegistry(state_path)
        self._storage = LocalStorage()
        self._physics = PhysicsEngine()
        self._results_db = ResultsDB(db_path=results_db_path)
        self._active_research_slice_id: str | None = None
        self._active_research_slice: dict[str, Any] | None = None

    def available_tools(self, stage: ResearchStage) -> list[str]:
        stage_tools = {
            ResearchStage.M1_DISCOVERY: [
                "parameter_sweep",
                "baseline_comparison",
                "evaluate_config",
                "query_incumbent",
                "query_pareto_front",
                "query_dead_zones",
            ],
            ResearchStage.M2_CONVERGENCE: [
                "convergence_grid",
                "ablation_check",
                "evaluate_config",
                "query_incumbent",
                "query_pareto_front",
                "query_neighborhood",
                "query_dead_zones",
            ],
            ResearchStage.M3_WALK_FORWARD: ["walk_forward"],
            ResearchStage.M4_HOLDOUT: ["holdout_validation"],
            ResearchStage.M5_EXECUTION: ["execution_mapping"],
        }
        return stage_tools[stage]

    def _set_active_slice(self, strategy_name: str, slice_payload: dict[str, Any]) -> str:
        normalized_slice = _normalized_slice_payload(slice_payload)
        research_slice_id = _research_slice_id(strategy_name, normalized_slice)
        self._active_research_slice_id = research_slice_id
        self._active_research_slice = {
            "strategy": strategy_name,
            "slice": normalized_slice,
        }
        return research_slice_id

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
        evaluation_window: int | None = None,
        min_total_signals: int | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        if self._has_execution_request(tickers, ticker_frames, start_date, end_date):
            sweep_slice = _normalized_slice_payload(
                {
                    "tickers": sorted(tickers or sorted(ticker_frames or {})),
                    "start_date": start_date.isoformat() if start_date is not None else None,
                    "end_date": end_date.isoformat() if end_date is not None else None,
                    "train_months": train_months,
                    "test_months": test_months,
                    "ratios": ratios or [1.0, 1.25, 1.5, 2.0],
                    "min_signals": min_signals,
                    "min_total_signals": min_total_signals,
                    "cost_r": cost_r,
                    "cost_bps": cost_bps,
                    "evaluation_window": evaluation_window,
                }
            )
            self._set_active_slice(strategy_name, sweep_slice)
        sweep_space = _entry_search_space(entry, parameter_space_override=parameter_space)
        configs, requested_config_count, invalid_config_count, normalized_duplicate_count = _canonical_sweep_configs(
            entry,
            parameter_space_override=parameter_space,
            max_configs=max_configs,
        )
        strategies = [self.registry.build(strategy_name, params) for params in configs]
        configs, strategies, strategy_duplicate_count = _dedupe_strategy_variants(configs, strategies)

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
            evaluation_window=evaluation_window,
            min_total_signals=min_total_signals,
            ticker_frames=ticker_frames,
            metrics=metrics,
        )
        result.summary["parameter_count"] = len(sweep_space)
        result.summary["requested_config_count"] = requested_config_count
        result.summary["duplicate_config_count"] = normalized_duplicate_count + strategy_duplicate_count
        result.summary["invalid_config_count"] = invalid_config_count
        result.summary["stage_objective"] = "find_edge_anywhere"
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

    def evaluate_config(
        self,
        strategy_name: str,
        config: dict[str, Any],
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
        evaluation_window: int | None = None,
        min_total_signals: int | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        normalized_config, inactive_parameters, errors = _normalize_strategy_config(entry, config)
        config_signature = _stable_signature(
            {"strategy": strategy_name, "config": normalized_config}
        )
        slice_payload = _normalized_slice_payload({
            "tickers": sorted(tickers or sorted(ticker_frames or {})),
            "start_date": start_date.isoformat() if start_date is not None else None,
            "end_date": end_date.isoformat() if end_date is not None else None,
            "train_months": train_months,
            "test_months": test_months,
            "ratios": ratios or [1.0, 1.25, 1.5, 2.0],
            "min_signals": min_signals,
            "min_total_signals": min_total_signals,
            "cost_r": cost_r,
            "cost_bps": cost_bps,
            "evaluation_window": evaluation_window,
        })
        research_slice_id = self._set_active_slice(strategy_name, slice_payload)
        request_signature = _stable_signature(
            {"strategy": strategy_name, "config_signature": config_signature, "slice": slice_payload}
        )
        existing = self._results_db.fetch_research_evaluation(request_signature)
        if existing is not None:
            duplicate_payload = dict(existing)
            duplicate_payload["status"] = "duplicate"
            duplicate_payload["already_evaluated"] = True
            duplicate_payload["request_signature"] = request_signature
            return ResearchToolResult(
                tool_name="evaluate_config",
                summary=duplicate_payload,
                artifacts={"cached_result": existing},
            )

        status = "ok"
        if errors:
            status = "invalid"
        elif not self._has_execution_request(tickers, ticker_frames, start_date, end_date):
            status = "invalid"
            errors.append("Point evaluation requires tickers and an explicit date range")

        if status == "invalid":
            payload = self._build_evaluation_payload(
                strategy_name=strategy_name,
                normalized_config=normalized_config,
                config_signature=config_signature,
                request_signature=request_signature,
                research_slice_id=research_slice_id,
                status=status,
                inactive_parameters=inactive_parameters,
                slice_payload=slice_payload,
                objective={},
                constraints={
                    "total_signals": 0,
                    "minimum_signals": min_total_signals or min_signals,
                    "passes_signal_floor": False,
                    "insufficiency_flags": ["invalid_config"],
                },
                runtime_seconds=0.0,
                errors=errors,
            )
            self._results_db.store_research_evaluation(payload)
            return ResearchToolResult(tool_name="evaluate_config", summary=payload)

        assert start_date is not None
        assert end_date is not None
        ratios = ratios or [1.0, 1.25, 1.5, 2.0]
        metrics = metrics or MetricsCalculator()
        strategy = self.registry.build(strategy_name, normalized_config)
        runtime_start = perf_counter()
        detail_df, aggregate_df = self._evaluate_single_config(
            strategy_name=strategy_name,
            strategy=strategy,
            config=normalized_config,
            tickers=tickers,
            ticker_frames=ticker_frames,
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            ratios=ratios,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
            evaluation_window=evaluation_window,
            min_total_signals=min_total_signals,
            metrics=metrics,
        )
        runtime_seconds = round(perf_counter() - runtime_start, 4)
        objective, constraints, status = self._summarize_single_config(
            aggregate_df=aggregate_df,
            detail_df=detail_df,
            minimum_signals=min_total_signals or min_signals,
            cost_r=cost_r,
        )
        payload = self._build_evaluation_payload(
            strategy_name=strategy_name,
            normalized_config=normalized_config,
            config_signature=config_signature,
                request_signature=request_signature,
                research_slice_id=research_slice_id,
                status=status,
                inactive_parameters=inactive_parameters,
                slice_payload=slice_payload,
            objective=objective,
            constraints=constraints,
            runtime_seconds=runtime_seconds,
            errors=[],
        )
        self._results_db.store_research_evaluation(payload)
        return ResearchToolResult(
            tool_name="evaluate_config",
            summary=payload,
            artifacts={"detail": detail_df, "aggregate": aggregate_df, "catalog_entry": entry},
        )

    def query_incumbent(
        self,
        strategy_name: str,
        *,
        ticker: str | None = None,
        research_slice_id: str | None = None,
        slice_filter: dict[str, Any] | None = None,
        include_non_competitive: bool = False,
    ) -> ResearchToolResult:
        normalized_slice_filter = _normalized_slice_filter(
            ticker=ticker,
            slice_filter=slice_filter,
        )
        effective_slice_id = self._resolve_query_slice_id(
            strategy_name,
            research_slice_id=research_slice_id,
            slice_filter=normalized_slice_filter,
        )
        rows = self._memory_rows(
            strategy_name,
            ticker=ticker,
            research_slice_id=effective_slice_id,
            slice_filter=normalized_slice_filter,
            include_non_competitive=include_non_competitive,
        )
        if not rows:
            return ResearchToolResult(
                tool_name="query_incumbent",
                summary={
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "research_slice_id": effective_slice_id,
                    "slice_filter": normalized_slice_filter,
                    "status": "empty",
                    "include_non_competitive": include_non_competitive,
                },
            )
        best = self._sort_memory_rows(rows)[0]
        return ResearchToolResult(
            tool_name="query_incumbent",
            summary={
                "strategy": strategy_name,
                "ticker": ticker,
                "status": "ok",
                "evaluated_configs": len(rows),
                "research_slice_id": effective_slice_id,
                "slice_filter": normalized_slice_filter,
                "include_non_competitive": include_non_competitive,
                "incumbent": self._compact_memory_row(best),
            },
        )

    def query_pareto_front(
        self,
        strategy_name: str,
        *,
        ticker: str | None = None,
        research_slice_id: str | None = None,
        slice_filter: dict[str, Any] | None = None,
        limit: int = 10,
        include_non_competitive: bool = False,
    ) -> ResearchToolResult:
        normalized_slice_filter = _normalized_slice_filter(
            ticker=ticker,
            slice_filter=slice_filter,
        )
        effective_slice_id = self._resolve_query_slice_id(
            strategy_name,
            research_slice_id=research_slice_id,
            slice_filter=normalized_slice_filter,
        )
        rows = self._sort_memory_rows(
            self._memory_rows(
                strategy_name,
                ticker=ticker,
                research_slice_id=effective_slice_id,
                slice_filter=normalized_slice_filter,
                include_non_competitive=include_non_competitive,
            )
        )
        pareto: list[dict[str, Any]] = []
        for row in rows:
            if any(self._dominates(existing, row) for existing in pareto):
                continue
            pareto = [existing for existing in pareto if not self._dominates(row, existing)]
            pareto.append(row)
            if len(pareto) >= limit:
                break
        return ResearchToolResult(
            tool_name="query_pareto_front",
            summary={
                "strategy": strategy_name,
                "ticker": ticker,
                "status": "ok" if pareto else "empty",
                "front_size": len(pareto),
                "research_slice_id": effective_slice_id,
                "slice_filter": normalized_slice_filter,
                "include_non_competitive": include_non_competitive,
                "pareto_front": [self._compact_memory_row(row) for row in pareto[:limit]],
            },
        )

    def query_neighborhood(
        self,
        strategy_name: str,
        config: dict[str, Any],
        *,
        ticker: str | None = None,
        research_slice_id: str | None = None,
        slice_filter: dict[str, Any] | None = None,
        radius: int = 1,
        limit: int = 5,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        normalized_config, inactive_parameters, errors = _normalize_strategy_config(entry, config)
        normalized_slice_filter = _normalized_slice_filter(
            ticker=ticker,
            slice_filter=slice_filter,
        )
        effective_slice_id = self._resolve_query_slice_id(
            strategy_name,
            research_slice_id=research_slice_id,
            slice_filter=normalized_slice_filter,
        )
        if errors:
            return ResearchToolResult(
                tool_name="query_neighborhood",
                summary={
                    "strategy": strategy_name,
                    "status": "invalid",
                    "errors": errors,
                },
            )
        rows = self._memory_rows(
            strategy_name,
            ticker=ticker,
            research_slice_id=effective_slice_id,
            slice_filter=normalized_slice_filter,
        )
        nearby: list[dict[str, Any]] = []
        for row in rows:
            distance = _config_distance(entry.search_spec, normalized_config, row["config"])
            if distance > radius:
                continue
            nearby.append({**row, "distance": distance})
        def objective_value(row: dict[str, Any]) -> float:
            value = row.get("objective", {}).get("value")
            return float(value) if value is not None else float("-inf")
        nearby.sort(
            key=lambda row: (
                row["distance"],
                -objective_value(row),
                -(row["constraints"].get("total_signals") or 0),
            )
        )
        return ResearchToolResult(
            tool_name="query_neighborhood",
            summary={
                "strategy": strategy_name,
                "ticker": ticker,
                "status": "ok" if nearby else "empty",
                "center_config": normalized_config,
                "inactive_parameters": inactive_parameters,
                "research_slice_id": effective_slice_id,
                "slice_filter": normalized_slice_filter,
                "radius": radius,
                "neighbors": [self._compact_memory_row(row) for row in nearby[:limit]],
            },
        )

    def query_dead_zones(
        self,
        strategy_name: str,
        *,
        ticker: str | None = None,
        research_slice_id: str | None = None,
        slice_filter: dict[str, Any] | None = None,
    ) -> ResearchToolResult:
        normalized_slice_filter = _normalized_slice_filter(
            ticker=ticker,
            slice_filter=slice_filter,
        )
        effective_slice_id = self._resolve_query_slice_id(
            strategy_name,
            research_slice_id=research_slice_id,
            slice_filter=normalized_slice_filter,
        )
        rows = self._memory_rows(
            strategy_name,
            ticker=ticker,
            research_slice_id=effective_slice_id,
            slice_filter=normalized_slice_filter,
            include_non_ok=True,
        )
        entry = self.registry.catalog_entry(strategy_name)
        dead_zones: list[dict[str, Any]] = []
        for parameter in (entry.search_spec.parameters if entry.search_spec is not None else []):
            stats_by_value: dict[str, dict[str, Any]] = {}
            for row in rows:
                value = row["config"].get(parameter.name)
                key = json.dumps(_json_ready(value), sort_keys=True)
                stats = stats_by_value.setdefault(
                    key,
                    {"value": value, "tests": 0, "best_exp_r": None, "invalid_count": 0, "insufficient_count": 0},
                )
                stats["tests"] += 1
                exp_r = row["objective"].get("value")
                if exp_r is not None and (
                    stats["best_exp_r"] is None or float(exp_r) > float(stats["best_exp_r"])
                ):
                    stats["best_exp_r"] = exp_r
                if row["status"] == "invalid":
                    stats["invalid_count"] += 1
                if row["status"] == "insufficient_signals":
                    stats["insufficient_count"] += 1
            for stats in stats_by_value.values():
                best_exp_r = stats["best_exp_r"]
                if stats["tests"] < 2:
                    continue
                failure_count = stats["invalid_count"] + stats["insufficient_count"]
                if failure_count < stats["tests"] and best_exp_r is not None and float(best_exp_r) > 0:
                    continue
                dead_zones.append(
                    {
                        "parameter": parameter.name,
                        "value": stats["value"],
                        "tests": stats["tests"],
                        "best_exp_r": best_exp_r,
                        "invalid_count": stats["invalid_count"],
                        "insufficient_count": stats["insufficient_count"],
                    }
                )
        dead_zones.sort(
            key=lambda item: (
                -(item["invalid_count"] + item["insufficient_count"]),
                -(item["tests"]),
                item["parameter"],
            )
        )
        return ResearchToolResult(
            tool_name="query_dead_zones",
            summary={
                "strategy": strategy_name,
                "ticker": ticker,
                "research_slice_id": effective_slice_id,
                "slice_filter": normalized_slice_filter,
                "status": "ok" if dead_zones else "empty",
                "dead_zones": dead_zones[:5],
            },
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
        evaluation_window: int | None = None,
        min_total_signals: int | None = None,
        ticker_frames: dict[str, pl.DataFrame] | None = None,
        metrics: MetricsCalculator | None = None,
    ) -> ResearchToolResult:
        entry = self.registry.catalog_entry(strategy_name)
        base_config = entry.strategy_config
        variants: list[dict[str, Any]] = []

        for param_name, candidates in _entry_search_space(entry).items():
            current_value = base_config.get(param_name)
            alt_values = [value for value in candidates if value != current_value]
            for alt_value in alt_values[:1]:
                ablated = dict(base_config)
                ablated[param_name] = alt_value
                normalized_variant, _, errors = _normalize_strategy_config(entry, ablated)
                if errors:
                    continue
                variants.append(normalized_variant)
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
            evaluation_window=evaluation_window,
            min_total_signals=min_total_signals,
            ticker_frames=ticker_frames,
            metrics=metrics,
        )
        result.summary["supports_parameter_ablation"] = bool(_entry_search_space(entry))
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
        evaluation_window: int | None = None,
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
            evaluation_window=evaluation_window,
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
                "stage_objective": "verify_stable_plateau",
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
        evaluation_window: int | None = None,
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
            evaluation_window=evaluation_window,
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
        evaluation_window: int | None = None,
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
            evaluation_window=evaluation_window,
        )
        detail_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        return ResearchToolResult(
            tool_name="execution_mapping",
            summary={
                "candidate_count": promoted.height,
                "mapped_count": detail_df.height,
                "stage_objective": "check_execution_stress_acceptability",
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
        evaluation_window: int | None = None,
    ) -> ResearchToolResult:
        stats = eval_direction(
            df_eval,
            direction,
            ratio,
            cost_bps,
            evaluation_window=evaluation_window,
        )
        passes_min_signals = int(stats["signals"]) >= min_signals
        selected_ratio, chosen_stats = choose_ratio(
            calib_df=df_eval,
            direction=direction,
            ratios=[ratio],
            cost_bps=cost_bps,
            min_calib_signals=min_signals,
            evaluation_window=evaluation_window,
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

    def _evaluate_single_config(
        self,
        *,
        strategy_name: str,
        strategy: BaseStrategy,
        config: dict[str, Any],
        tickers: list[str] | None,
        ticker_frames: dict[str, pl.DataFrame] | None,
        start_date: date,
        end_date: date,
        train_months: int,
        test_months: int,
        ratios: list[float],
        min_signals: int,
        cost_r: float | None,
        cost_bps: float | None,
        evaluation_window: int | None,
        min_total_signals: int | None,
        metrics: MetricsCalculator,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        windows = build_windows(start_date, end_date, train_months, test_months)
        if not windows:
            return pl.DataFrame(), pl.DataFrame()
        enriched_frames = self._load_enriched_frames(
            strategies=[strategy],
            tickers=tickers or sorted(ticker_frames or {}),
            start_date=start_date,
            end_date=end_date,
            ticker_frames=ticker_frames,
        )
        detail_df = self._run_strategy_sweep(
            strategy_name=strategy_name,
            strategies=[strategy],
            configs=[config],
            ticker_frames=enriched_frames,
            windows=windows,
            ratios=ratios,
            metrics=metrics,
            min_signals=min_signals,
            cost_r=cost_r,
            cost_bps=cost_bps,
            evaluation_window=evaluation_window,
        )
        aggregate_df = self._aggregate_sweep(
            detail_df,
            [config],
            min_total_signals=min_total_signals,
        )
        return detail_df, aggregate_df

    @staticmethod
    def _weighted_metric(frame: pl.DataFrame, value_col: str, weight_col: str) -> float | None:
        if frame.is_empty() or value_col not in frame.columns:
            return None
        rows = frame.select([value_col, weight_col]).to_dicts()
        weighted_sum = 0.0
        total_weight = 0.0
        for row in rows:
            value = row.get(value_col)
            weight = row.get(weight_col)
            if value is None or weight is None:
                continue
            value_f = float(value)
            weight_f = float(weight)
            if not isfinite(value_f) or not isfinite(weight_f) or weight_f <= 0:
                continue
            weighted_sum += value_f * weight_f
            total_weight += weight_f
        if total_weight <= 0:
            return None
        return round(weighted_sum / total_weight, 4)

    def _summarize_single_config(
        self,
        *,
        aggregate_df: pl.DataFrame,
        detail_df: pl.DataFrame,
        minimum_signals: int,
        cost_r: float | None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        combined_df = (
            aggregate_df.filter(pl.col("direction") == "combined")
            if not aggregate_df.is_empty() and "direction" in aggregate_df.columns
            else aggregate_df
        )
        if combined_df.is_empty():
            return (
                {
                    "primary_metric": "avg_test_exp_r",
                    "value": None,
                    "confidence": None,
                    "avg_mfe_mae_ratio": None,
                },
                {
                    "total_signals": 0,
                    "minimum_signals": minimum_signals,
                    "passes_signal_floor": False,
                    "insufficiency_flags": ["no_valid_walk_forward_rows"],
                    "effective_cost_r": cost_r,
                },
                "insufficient_signals",
            )

        total_signals = int(combined_df["total_oos_signals"].sum()) if "total_oos_signals" in combined_df.columns else 0
        avg_exp_r = self._weighted_metric(combined_df, "avg_test_exp_r", "total_oos_signals")
        avg_confidence = self._weighted_metric(combined_df, "avg_test_confidence", "total_oos_signals")
        avg_mfe_mae_ratio = self._weighted_metric(combined_df, "avg_test_mfe_mae_ratio", "total_oos_signals")
        effective_cost_r = self._weighted_metric(combined_df, "avg_effective_cost_r", "total_oos_signals")
        insufficiency_flags: list[str] = []
        passes_signal_floor = total_signals >= minimum_signals
        if not passes_signal_floor:
            insufficiency_flags.append("below_signal_floor")
        status = "ok" if passes_signal_floor else "insufficient_signals"
        return (
            {
                "primary_metric": "avg_test_exp_r",
                "value": avg_exp_r,
                "confidence": avg_confidence,
                "avg_mfe_mae_ratio": avg_mfe_mae_ratio,
            },
            {
                "total_signals": total_signals,
                "minimum_signals": minimum_signals,
                "passes_signal_floor": passes_signal_floor,
                "insufficiency_flags": insufficiency_flags,
                "effective_cost_r": effective_cost_r,
                "detail_rows": detail_df.height,
            },
            status,
        )

    @staticmethod
    def _build_evaluation_payload(
        *,
        strategy_name: str,
        normalized_config: dict[str, Any],
        config_signature: str,
        request_signature: str,
        research_slice_id: str,
        status: str,
        inactive_parameters: list[str],
        slice_payload: dict[str, Any],
        objective: dict[str, Any],
        constraints: dict[str, Any],
        runtime_seconds: float,
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "strategy": strategy_name,
            "config": _json_ready(normalized_config),
            "config_signature": config_signature,
            "request_signature": request_signature,
            "research_slice_id": research_slice_id,
            "status": status,
            "already_evaluated": False,
            "inactive_parameters": inactive_parameters,
            "objective": objective,
            "constraints": constraints,
            "runtime_seconds": runtime_seconds,
            "slice": slice_payload,
            "errors": errors,
        }

    def _memory_rows(
        self,
        strategy_name: str,
        *,
        ticker: str | None = None,
        research_slice_id: str | None = None,
        slice_filter: dict[str, Any] | None = None,
        include_non_ok: bool = False,
        include_non_competitive: bool = True,
    ) -> list[dict[str, Any]]:
        rows = self._results_db.list_research_evaluations(
            strategy=strategy_name,
            ticker=ticker,
            research_slice_id=research_slice_id,
        )
        if slice_filter:
            rows = [
                row for row in rows
                if _slice_matches(row.get("slice", {}), slice_filter)
            ]
        if include_non_ok:
            return rows
        filtered = [row for row in rows if row.get("status") != "invalid"]
        if include_non_competitive:
            return filtered
        return [row for row in filtered if ResearchToolbox._is_competitive_memory_row(row)]

    def _resolve_query_slice_id(
        self,
        strategy_name: str,
        *,
        research_slice_id: str | None,
        slice_filter: dict[str, Any],
    ) -> str | None:
        if research_slice_id is not None:
            return research_slice_id
        if slice_filter:
            return None
        if self._active_research_slice is None or self._active_research_slice_id is None:
            return None
        if self._active_research_slice.get("strategy") != strategy_name:
            return None
        return self._active_research_slice_id

    @staticmethod
    def _is_competitive_memory_row(row: dict[str, Any]) -> bool:
        if row.get("status") != "ok":
            return False
        constraints = row.get("constraints", {})
        if constraints.get("passes_signal_floor") is False:
            return False
        return True

    @staticmethod
    def _sort_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def rank_value(value: Any) -> float:
            if value is None:
                return float("-inf")
            return float(value)

        return sorted(
            rows,
            key=lambda row: (
                rank_value(row.get("objective", {}).get("value")),
                rank_value(row.get("objective", {}).get("confidence")),
                float(row.get("constraints", {}).get("total_signals") or 0),
                row.get("config_signature", ""),
            ),
            reverse=True,
        )

    @staticmethod
    def _compact_memory_row(row: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "research_slice_id": row.get("research_slice_id"),
            "config_signature": row.get("config_signature"),
            "status": row.get("status"),
            "config": row.get("config"),
            "objective": row.get("objective"),
            "constraints": row.get("constraints"),
            "inactive_parameters": row.get("inactive_parameters", []),
            "tickers": row.get("slice", {}).get("tickers", []),
        }
        if "distance" in row:
            compact["distance"] = row["distance"]
        return compact

    @staticmethod
    def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_obj = left.get("objective", {})
        right_obj = right.get("objective", {})
        left_constraints = left.get("constraints", {})
        right_constraints = right.get("constraints", {})
        def score(value: Any) -> float:
            return float(value) if value is not None else float("-inf")
        comparisons = [
            (score(left_obj.get("value")), score(right_obj.get("value"))),
            (score(left_obj.get("confidence")), score(right_obj.get("confidence"))),
            (left_constraints.get("total_signals") or 0, right_constraints.get("total_signals") or 0),
        ]
        return all(left_value >= right_value for left_value, right_value in comparisons) and any(
            left_value > right_value for left_value, right_value in comparisons
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
        evaluation_window: int | None,
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
            evaluation_window=evaluation_window,
        )
        aggregate_df = self._aggregate_sweep(detail_df, configs, min_total_signals=min_total_signals)
        aggregate_df = _annotate_plateau_metrics(
            aggregate_df,
            parameter_space=_entry_search_space(entry),
        )
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
        evaluation_window: int | None,
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
                    evaluation_window=evaluation_window,
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

        detail_df = detail_df.with_columns([
            pl.when(pl.col("test_exp_r").is_nan()).then(None).otherwise(pl.col("test_exp_r")).alias("test_exp_r"),
            pl.when(pl.col("test_confidence").is_nan()).then(None).otherwise(pl.col("test_confidence")).alias("test_confidence"),
            pl.when(pl.col("effective_cost_r").is_nan()).then(None).otherwise(pl.col("effective_cost_r")).alias("effective_cost_r"),
        ])

        config_cols = [column for column in _config_columns(configs) if column in detail_df.columns]
        context_cols = [
            column for column in ("catalog_strategy", "base_strategy") if column in detail_df.columns
        ]
        group_cols = ["ticker", "strategy", "direction", *context_cols, *config_cols]
        agg_exprs = [
            pl.len().alias("oos_windows"),
            pl.col("test_signals").sum().alias("total_oos_signals"),
            pl.col("test_exp_r").mean().alias("avg_oos_exp_r"),
            pl.col("test_exp_r").median().alias("med_oos_exp_r"),
            (pl.col("test_exp_r") > 0).mean().alias("pct_positive_windows"),
            pl.col("test_confidence").mean().alias("avg_confidence"),
            pl.col("effective_cost_r").mean().alias("avg_effective_cost_r"),
            pl.col("selected_ratio").median().alias("median_selected_ratio"),
        ]
        if "test_avg_mfe_mae_ratio" in detail_df.columns:
            agg_exprs.append(
                pl.col("test_avg_mfe_mae_ratio").mean().alias("avg_mfe_mae_ratio")
            )
        aggregate_df = (
            detail_df.group_by(group_cols)
            .agg(agg_exprs)
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
            pl.col("avg_mfe_mae_ratio").alias("avg_test_mfe_mae_ratio")
            if "avg_mfe_mae_ratio" in aggregate_df.columns
            else pl.lit(None).alias("avg_test_mfe_mae_ratio"),
            (
                pl.col("avg_oos_exp_r") * 1000
                + pl.col("pct_positive_windows") * 100
                + pl.col("total_oos_signals") / 1000
            ).alias("discovery_score"),
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
