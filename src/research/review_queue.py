"""Stateful human review queue and review artifacts for nightly research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import polars as pl
from polars.exceptions import NoDataError

from src.chronos.storage import LocalStorage
from src.config import PROJECT_ROOT
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.oracle.monte_carlo import ExecutionStressConfig
from src.research.exit_optimizer import optimize_underlying_exit, write_exit_optimization_result
from src.research.models import StrategyCatalogEntry
from src.research.registry import ResearchRegistry
from src.research.stages import (
    aggregate_walk_forward,
    build_gate_report,
    build_windows,
    promoted_candidates_from_holdout,
    run_execution_mapping_for_candidates,
    run_holdout_validation_for_candidates,
    run_walk_forward_for_strategies,
    summarize_holdout,
)
from src.research.stages.candidates import build_candidate_strategy, candidate_params
from src.strategy.base import required_feature_union


QUEUE_FILENAME = "m2_human_review_queue.csv"
QUEUE_HISTORY_FILENAME = "m2_human_review_history.csv"
QUEUE_SNAPSHOT_FILENAME = "m2_human_review_queue_snapshot.json"
REVIEW_BUNDLE_DIRNAME = "review_bundle"
FOLLOWUP_DIRNAME = "followup_runs"
CHARTS_DIRNAME = "charts"

FAMILY_CATALOG_STRATEGY_NAMES = {
    "market_impulse": "Market Impulse (Cross & Reclaim)",
    "jerk_pivot_momentum": "Jerk-Pivot Momentum (tight)",
    "elastic_band_reversion": "Elastic Band Reversion",
}

QUEUE_STATUS_NEW = "NEW"
QUEUE_STATUS_PENDING = "PENDING"
QUEUE_STATUS_EXECUTING = "EXECUTING"
QUEUE_STATUS_EXECUTED = "EXECUTED"
QUEUE_STATUS_KILLED = "KILLED"
QUEUE_STATUS_STALE = "STALE"
QUEUE_STATUS_ERROR = "ERROR"

TERMINAL_QUEUE_STATUSES = {QUEUE_STATUS_EXECUTED, QUEUE_STATUS_KILLED}
ACTIONABLE_HUMAN_DECISIONS = {"promote_to_m3", "retune", "expand_symbols", "kill"}
MANUAL_QUEUE_FIELDS = {
    "human_decision",
    "human_notes",
    "priority",
    "queue_status",
    "human_updated_at",
}
OBSERVATION_COLUMN_PREFIXES = ("m1_", "m2_")
BOOL_QUEUE_FIELDS = {
    "passes_m1",
    "passes_m2",
    "passes_m3",
    "passes_m4",
    "passes_m5",
    "is_full_m1_m5_survivor",
    "m2_has_all_cost_points",
    "m2_passes_window_gate",
    "m2_passes_signal_gate",
    "m2_passes_stability_gate",
    "m2_passes_exp_gate",
    "m2_passes_all_gates",
}
INT_QUEUE_FIELDS = {
    "priority",
    "m1_oos_windows",
    "m1_oos_signals",
    "m2_observed_cost_points",
    "m2_min_oos_windows",
    "m2_min_oos_signals",
}
FLOAT_QUEUE_FIELDS = {
    "m1_avg_test_exp_r",
    "m1_pct_positive_oos_windows",
    "m1_avg_test_confidence",
    "m1_score",
    "m2_min_avg_test_exp_r",
    "m2_mean_avg_test_exp_r",
    "m2_min_pct_positive_oos_windows",
    "m2_mean_pct_positive_oos_windows",
    "m2_mean_test_confidence",
    "m2_score",
}


@dataclass(slots=True, frozen=True)
class ReviewQueuePaths:
    control_dir: Path
    queue_path: Path
    history_path: Path
    snapshot_path: Path
    charts_dir: Path
    review_bundle_dir: Path
    workbook_path: Path


@dataclass(slots=True, frozen=True)
class ReviewQueueArtifacts:
    queue_path: Path
    history_path: Path
    workbook_path: Path
    review_bundle_dir: Path
    charts_dir: Path
    followup_actions_run_count: int


def resolve_review_queue_paths(control_dir: str | Path) -> ReviewQueuePaths:
    resolved_control_dir = Path(control_dir)
    if not resolved_control_dir.is_absolute():
        resolved_control_dir = (PROJECT_ROOT / resolved_control_dir).resolve()
    charts_dir = resolved_control_dir / CHARTS_DIRNAME
    review_bundle_dir = resolved_control_dir / REVIEW_BUNDLE_DIRNAME
    return ReviewQueuePaths(
        control_dir=resolved_control_dir,
        queue_path=resolved_control_dir / QUEUE_FILENAME,
        history_path=resolved_control_dir / QUEUE_HISTORY_FILENAME,
        snapshot_path=resolved_control_dir / QUEUE_SNAPSHOT_FILENAME,
        charts_dir=charts_dir,
        review_bundle_dir=review_bundle_dir,
        workbook_path=review_bundle_dir / "human_review_workbook.xlsx",
    )


class HumanReviewQueueManager:
    """Own the durable review queue, history, charts, and follow-up actions."""

    def __init__(
        self,
        control_dir: str | Path,
        *,
        registry: ResearchRegistry | None = None,
        storage: LocalStorage | None = None,
        physics: PhysicsEngine | None = None,
        metrics: MetricsCalculator | None = None,
        frame_loader: Callable[[str, date | None, date | None], pl.DataFrame] | None = None,
        enricher: Callable[[pl.DataFrame, set[str]], pl.DataFrame] | None = None,
        followup_executor: Callable[[dict[str, Any], str, Any, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.paths = resolve_review_queue_paths(control_dir)
        self.paths.control_dir.mkdir(parents=True, exist_ok=True)
        self.paths.charts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.review_bundle_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or ResearchRegistry()
        self.storage = storage or LocalStorage()
        self.physics = physics or PhysicsEngine()
        self.metrics = metrics or MetricsCalculator()
        self.frame_loader = frame_loader or self.storage.load_bars
        self.enricher = enricher or self.physics.enrich_for_features
        self.followup_executor = followup_executor

    def refresh_queue(
        self,
        *,
        run_dirs: dict[str, Path],
        config: Any,
        run_date: date,
    ) -> ReviewQueueArtifacts:
        queue_rows = self._load_rows(self.paths.queue_path)
        snapshot = self._load_snapshot()
        observations = self.collect_observations(run_dirs=run_dirs, config=config, run_date=run_date)
        merged_rows = self._merge_rows(
            existing_rows=queue_rows,
            snapshot=snapshot,
            observations=observations,
            run_date=run_date,
            max_new_rows=getattr(config.followup_budgets, "max_new_m2_rows_per_night", 10),
        )
        self._append_history(observations, run_date=run_date)
        executed_rows, followup_actions_run_count = self._execute_followups(
            rows=merged_rows,
            config=config,
            run_date=run_date,
        )
        self._write_rows(self.paths.queue_path, executed_rows)
        self._write_snapshot(executed_rows)
        self._write_review_bundle(executed_rows)
        return ReviewQueueArtifacts(
            queue_path=self.paths.queue_path,
            history_path=self.paths.history_path,
            workbook_path=self.paths.workbook_path,
            review_bundle_dir=self.paths.review_bundle_dir,
            charts_dir=self.paths.charts_dir,
            followup_actions_run_count=followup_actions_run_count,
        )

    def replay_followups(
        self,
        *,
        config: Any,
        run_date: date,
        candidate_keys: list[str] | None = None,
        only_error_rows: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = self._load_rows(self.paths.queue_path)
        if not rows:
            self._write_review_bundle(rows)
            return rows, 0

        selected_keys = set(candidate_keys or [])
        if selected_keys:
            missing = selected_keys - {str(row.get("candidate_key", "")) for row in rows}
            if missing:
                raise ValueError(f"Unknown candidate_key(s): {sorted(missing)}")

        replayed = 0
        by_key = {str(row["candidate_key"]): dict(row) for row in rows if row.get("candidate_key")}
        for candidate_key, target in by_key.items():
            if selected_keys and candidate_key not in selected_keys:
                continue
            if only_error_rows and not _has_followup_error_note(target.get("human_notes", "")):
                continue
            decision = str(target.get("human_decision", "") or "").strip()
            if decision not in ACTIONABLE_HUMAN_DECISIONS:
                continue

            target["queue_status"] = QUEUE_STATUS_PENDING
            target["human_notes"] = _strip_followup_error_notes(target.get("human_notes", ""))
            try:
                result = self._run_followup(
                    row=target,
                    decision=decision,
                    config=config,
                    run_date=run_date,
                )
            except Exception as exc:  # pragma: no cover - exercised via targeted tests
                target["queue_status"] = QUEUE_STATUS_ERROR
                target["latest_stage_decision"] = f"error:{type(exc).__name__}"
                target["human_notes"] = _append_note(
                    target.get("human_notes", ""),
                    f"[{run_date.isoformat()}] follow-up error: {exc}",
                )
                target["last_action_run_date"] = run_date.isoformat()
            else:
                target.update(result)
                target["last_action_run_date"] = run_date.isoformat()
                target = self._normalize_row(target, run_date=run_date, manual_override=False)
                by_key[candidate_key] = target
            replayed += 1

        ordered = self._sort_rows(list(by_key.values()))
        self._write_rows(self.paths.queue_path, ordered)
        self._write_snapshot(ordered)
        self._write_review_bundle(ordered)
        return ordered, replayed

    def collect_observations(
        self,
        *,
        run_dirs: dict[str, Path],
        config: Any,
        run_date: date,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for family, run_dir in sorted(run_dirs.items()):
            research_slice_id = self._build_slice_id(family=family, config=config)
            observations.extend(
                self._collect_run_observations(
                    family=family,
                    run_dir=run_dir,
                    config=config,
                    run_date=run_date,
                    research_slice_id=research_slice_id,
                )
            )
        observations.sort(
            key=lambda row: (
                float(row.get("m2_score") or float("-inf")),
                float(row.get("m1_score") or float("-inf")),
                row.get("candidate_key", ""),
            ),
            reverse=True,
        )
        return observations

    def _collect_run_observations(
        self,
        *,
        family: str,
        run_dir: Path,
        config: Any,
        run_date: date,
        research_slice_id: str,
    ) -> list[dict[str, Any]]:
        m1_top = _read_optional_csv(run_dir, "m1_top_candidates.csv")
        m2_gate = _read_optional_csv(run_dir, "m2_gate_report.csv")
        if m2_gate.is_empty():
            return []
        m3_detail = _read_optional_csv(run_dir, "m3_walk_forward_detail.csv")
        m4_summary = _read_optional_csv(
            run_dir,
            "m4_holdout_summary.csv",
            "M4_holdout_validation_summary.csv",
        )
        m5_execution = _read_optional_csv(
            run_dir,
            "m5_execution_mapping.csv",
            "M5_execution_mapping_detail.csv",
        )

        m1_lookup = self._index_frame(m1_top, research_slice_id=research_slice_id, family=family)
        m3_keys = set(self._index_frame(m3_detail, research_slice_id=research_slice_id, family=family))
        m4_lookup = self._index_frame(m4_summary, research_slice_id=research_slice_id, family=family)
        m5_lookup = self._index_frame(m5_execution, research_slice_id=research_slice_id, family=family)

        survivors = m2_gate.filter(
            pl.when(pl.col("passes_all_gates").is_not_null())
            .then(pl.col("passes_all_gates"))
            .otherwise(pl.col("decision") == "promote_to_holdout")
        )
        observations: list[dict[str, Any]] = []
        for row in survivors.iter_rows(named=True):
            params = candidate_params(row)
            candidate_key = build_review_candidate_key(
                family=family,
                ticker=str(row["ticker"]),
                direction=str(row["direction"]),
                strategy=str(row["strategy"]),
                params=params,
                research_slice_id=research_slice_id,
            )
            m1_row = m1_lookup.get(candidate_key, {})
            m4_row = m4_lookup.get(candidate_key, {})
            passes_m3 = candidate_key in m3_keys or candidate_key in m4_lookup or candidate_key in m5_lookup
            passes_m4 = bool(_boolish(m4_row.get("passes_holdout")))
            passes_m5 = candidate_key in m5_lookup
            latest_stage_reached, latest_stage_decision = _latest_stage_state(
                passes_m1=bool(m1_row),
                passes_m2=bool(_boolish(row.get("passes_all_gates"), default=True)),
                passes_m3=passes_m3,
                passes_m4=passes_m4,
                passes_m5=passes_m5,
                m2_decision=row.get("decision"),
                m4_decision=m4_row.get("decision"),
            )
            observation = {
                "candidate_key": candidate_key,
                "strategy_family": family,
                "ticker": str(row["ticker"]),
                "strategy": str(row["strategy"]),
                "direction": str(row["direction"]),
                "research_slice_id": research_slice_id,
                "config_signature": stable_signature(params),
                "config_json": json.dumps(json_ready(params), sort_keys=True),
                "chart_link": str(
                    self._render_chart(
                        family=family,
                        candidate={**row, **params},
                        config=config,
                        candidate_key=candidate_key,
                    )
                ),
                "passes_m1": bool(m1_row),
                "passes_m2": bool(_boolish(row.get("passes_all_gates"), default=True)),
                "passes_m3": passes_m3,
                "passes_m4": passes_m4,
                "passes_m5": passes_m5,
                "is_full_m1_m5_survivor": all(
                    [
                        bool(m1_row),
                        bool(_boolish(row.get("passes_all_gates"), default=True)),
                        passes_m3,
                        passes_m4,
                        passes_m5,
                    ]
                ),
                "latest_stage_reached": latest_stage_reached,
                "latest_stage_decision": latest_stage_decision,
                "last_seen_run_date": run_date.isoformat(),
                "last_source_run_dir": str(run_dir.resolve()),
                **params,
                **prefix_keys(m1_row, "m1_", exclude={"candidate_key"}),
                **prefix_keys(row, "m2_", exclude={"ticker", "strategy", "direction"}),
            }
            observations.append(observation)
        return observations

    def _merge_rows(
        self,
        *,
        existing_rows: list[dict[str, Any]],
        snapshot: dict[str, dict[str, Any]],
        observations: list[dict[str, Any]],
        run_date: date,
        max_new_rows: int,
    ) -> list[dict[str, Any]]:
        observation_map = {row["candidate_key"]: row for row in observations}
        existing_map = {row["candidate_key"]: row for row in existing_rows if row.get("candidate_key")}
        merged: list[dict[str, Any]] = []

        for candidate_key, existing_row in existing_map.items():
            manual_override = self._is_manual_override(
                row=existing_row,
                snapshot=snapshot.get(candidate_key, {}),
            )
            observed = observation_map.get(candidate_key)
            if observed is None:
                carried = dict(existing_row)
                if carried.get("queue_status") not in TERMINAL_QUEUE_STATUSES:
                    last_seen = parse_iso_date(carried.get("last_seen_run_date"))
                    if last_seen is not None and (run_date - last_seen) >= timedelta(days=3):
                        carried["queue_status"] = QUEUE_STATUS_STALE
                merged.append(self._normalize_row(carried, run_date=run_date, manual_override=manual_override))
                continue

            combined = dict(existing_row)
            for key, value in observed.items():
                combined[key] = value
            if existing_row.get("queue_status") in TERMINAL_QUEUE_STATUSES and not manual_override:
                combined["queue_status"] = existing_row.get("queue_status")
                combined["human_decision"] = existing_row.get("human_decision", "")
                combined["human_notes"] = existing_row.get("human_notes", "")
                combined["priority"] = existing_row.get("priority", 0)
                combined["human_updated_at"] = existing_row.get("human_updated_at", "")
            merged.append(self._normalize_row(combined, run_date=run_date, manual_override=manual_override))

        new_rows = [
            row for row in observations
            if row["candidate_key"] not in existing_map
        ]
        new_rows.sort(
            key=lambda row: (
                float(row.get("m2_score") or float("-inf")),
                float(row.get("m1_score") or float("-inf")),
            ),
            reverse=True,
        )
        for observation in new_rows[:max_new_rows]:
            candidate = self._normalize_row(
                {
                    **observation,
                    "human_decision": "",
                    "human_notes": "",
                    "priority": 0,
                    "queue_status": QUEUE_STATUS_NEW,
                    "last_action_run_date": "",
                    "human_updated_at": "",
                    "latest_artifact_dir": "",
                },
                run_date=run_date,
                manual_override=False,
            )
            merged.append(candidate)

        return self._sort_rows(merged)

    def _execute_followups(
        self,
        *,
        rows: list[dict[str, Any]],
        config: Any,
        run_date: date,
    ) -> tuple[list[dict[str, Any]], int]:
        eligible = self._eligible_rows(rows=rows, run_date=run_date)
        budgets = config.followup_budgets
        totals = {
            "promote_to_m3": 0,
            "retune": 0,
            "expand_symbols": 0,
            "all": 0,
        }
        by_key = {row["candidate_key"]: row for row in rows}
        for row in eligible:
            decision = row.get("human_decision", "").strip()
            if decision == "promote_to_m3" and totals["promote_to_m3"] >= budgets.max_m3_promotions_per_night:
                continue
            if decision == "retune" and totals["retune"] >= budgets.max_retune_tasks_per_night:
                continue
            if decision == "expand_symbols" and totals["expand_symbols"] >= budgets.max_symbol_expansion_tasks_per_night:
                continue
            if totals["all"] >= budgets.max_total_followup_tasks_per_night:
                break

            target = by_key[row["candidate_key"]]
            target["queue_status"] = QUEUE_STATUS_EXECUTING
            try:
                result = self._run_followup(
                    row=target,
                    decision=decision,
                    config=config,
                    run_date=run_date,
                )
            except Exception as exc:  # pragma: no cover - exercised via targeted tests
                target["queue_status"] = QUEUE_STATUS_ERROR
                target["latest_stage_decision"] = f"error:{type(exc).__name__}"
                target["human_notes"] = _append_note(
                    target.get("human_notes", ""),
                    f"[{run_date.isoformat()}] follow-up error: {exc}",
                )
                target["last_action_run_date"] = run_date.isoformat()
            else:
                target.update(result)
                target["last_action_run_date"] = run_date.isoformat()
                target = self._normalize_row(target, run_date=run_date, manual_override=False)
                by_key[row["candidate_key"]] = target
            totals["all"] += 1
            if decision in totals:
                totals[decision] += 1
        return self._sort_rows(list(by_key.values())), totals["all"]

    def _run_followup(
        self,
        *,
        row: dict[str, Any],
        decision: str,
        config: Any,
        run_date: date,
    ) -> dict[str, Any]:
        followup_root = self.paths.control_dir / FOLLOWUP_DIRNAME / run_date.isoformat()
        followup_root.mkdir(parents=True, exist_ok=True)
        artifact_dir = followup_root / f"{row['candidate_key']}_{decision}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if self.followup_executor is not None:
            return self.followup_executor(row, decision, config, artifact_dir)
        if decision == "kill":
            return {
                "queue_status": QUEUE_STATUS_KILLED,
                "latest_stage_decision": "killed_by_human",
                "latest_artifact_dir": str(artifact_dir),
            }
        if decision == "promote_to_m3":
            return self._execute_promote_to_m3(row=row, config=config, artifact_dir=artifact_dir)
        if decision == "retune":
            return self._execute_retune(row=row, config=config, artifact_dir=artifact_dir)
        if decision == "expand_symbols":
            return self._execute_expand_symbols(row=row, config=config, artifact_dir=artifact_dir)
        return {
            "queue_status": QUEUE_STATUS_ERROR,
            "latest_stage_decision": f"unsupported_decision:{decision}",
            "latest_artifact_dir": str(artifact_dir),
        }

    def _execute_promote_to_m3(
        self,
        *,
        row: dict[str, Any],
        config: Any,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        candidate_df = pl.DataFrame([self._candidate_stage_payload(row)])
        ticker_frames = {
            row["ticker"]: self._load_enriched_candidate_frame(row=row, config=config),
        }
        windows = build_windows(
            config.defaults.start,
            config.defaults.end,
            config.defaults.train_months,
            config.defaults.test_months,
        )
        ratios = parse_csv_floats(config.defaults.ratios)
        costs = parse_csv_floats(config.defaults.cost_grid_bps)
        strategy = build_candidate_strategy(self._candidate_stage_payload(row))
        m3_rows = run_walk_forward_for_strategies(
            ticker=row["ticker"],
            df=ticker_frames[row["ticker"]],
            strategies=[strategy],
            windows=windows,
            ratios=ratios,
            metrics=self.metrics,
            min_signals=config.defaults.min_signals,
            cost_bps=config.defaults.m1_cost_bps,
        )
        m3_detail = pl.DataFrame(
            [item for item in m3_rows if item["direction"] == row["direction"]]
        ) if m3_rows else pl.DataFrame()
        holdout_rows = run_holdout_validation_for_candidates(
            promoted=candidate_df,
            ticker_frames=ticker_frames,
            metrics=self.metrics,
            start_date=config.defaults.start,
            calibration_end=config.defaults.calibration_end,
            holdout_start=config.defaults.holdout_start,
            holdout_end=config.defaults.holdout_end,
            ratios=ratios,
            costs=costs,
            min_calibration_signals=config.defaults.min_calibration_signals,
            min_holdout_signals=config.defaults.min_holdout_signals,
        )
        m4_detail = pl.DataFrame(holdout_rows) if holdout_rows else pl.DataFrame()
        m4_summary = summarize_holdout(m4_detail, cost_count=len(costs)) if not m4_detail.is_empty() else pl.DataFrame()
        m5_candidates = promoted_candidates_from_holdout(m4_summary) if not m4_summary.is_empty() else pl.DataFrame()
        m5_rows = run_execution_mapping_for_candidates(
            promoted=m5_candidates,
            holdout_detail=m4_detail,
            ticker_frames=ticker_frames,
            metrics=self.metrics,
            holdout_start=config.defaults.holdout_start,
            holdout_end=config.defaults.holdout_end,
            base_cost_r=config.defaults.base_cost_r,
            stress_cfg=ExecutionStressConfig(bootstrap_iters=config.defaults.bootstrap_iters),
        ) if not m5_candidates.is_empty() else []
        m5_detail = pl.DataFrame(m5_rows) if m5_rows else pl.DataFrame()
        if not m5_detail.is_empty():
            exit_optimization = optimize_underlying_exit(
                strategy_key=_strategy_key_for_name(str(row["strategy"])),
                symbol=str(row["ticker"]),
                direction=str(row["direction"]),
                strategy=strategy,
                enriched_frame=ticker_frames[row["ticker"]],
                holdout_start=config.defaults.holdout_start,
                holdout_end=config.defaults.holdout_end,
                catastrophe_exit_params=_catastrophe_exit_defaults(m5_detail.row(0, named=True)),
            )
            if exit_optimization is not None:
                write_exit_optimization_result(
                    exit_optimization,
                    path=artifact_dir / "m5_exit_optimization.json",
                )

        _write_frame_if_not_empty(m3_detail, artifact_dir / "m3_walk_forward_detail.csv")
        _write_frame_if_not_empty(m4_detail, artifact_dir / "m4_holdout_detail.csv")
        _write_frame_if_not_empty(m4_summary, artifact_dir / "m4_holdout_summary.csv")
        _write_frame_if_not_empty(m5_detail, artifact_dir / "m5_execution_mapping.csv")

        passes_m4 = (
            not m4_summary.is_empty()
            and bool(m4_summary.row(0, named=True).get("passes_holdout"))
        )
        passes_m5 = not m5_detail.is_empty()
        latest_stage_reached = "M5" if passes_m5 else "M4" if passes_m4 else "M3"
        latest_stage_decision = (
            "promote"
            if passes_m5
            else (
                str(m4_summary.row(0, named=True).get("decision"))
                if not m4_summary.is_empty()
                else "walk_forward_completed"
            )
        )
        return {
            "queue_status": QUEUE_STATUS_EXECUTED,
            "passes_m3": not m3_detail.is_empty(),
            "passes_m4": passes_m4,
            "passes_m5": passes_m5,
            "is_full_m1_m5_survivor": bool(row.get("passes_m1"))
            and bool(row.get("passes_m2"))
            and not m3_detail.is_empty()
            and passes_m4
            and passes_m5,
            "latest_stage_reached": latest_stage_reached,
            "latest_stage_decision": latest_stage_decision,
            "latest_artifact_dir": str(artifact_dir),
        }

    def _execute_retune(
        self,
        *,
        row: dict[str, Any],
        config: Any,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        strategy_name = self._catalog_strategy_name(row)
        base_config = self._queue_config_params(row)
        entry = self.registry.catalog_entry(strategy_name, base_config)
        neighbor_configs = build_neighbor_configs(entry, base_config, max_configs=8)
        windows = build_windows(
            config.defaults.start,
            config.defaults.end,
            config.defaults.train_months,
            config.defaults.test_months,
        )
        ratios = parse_csv_floats(config.defaults.ratios)
        costs = parse_csv_floats(config.defaults.cost_grid_bps)

        m1_records: list[dict[str, Any]] = []
        m2_frames: list[pl.DataFrame] = []
        for neighbor in neighbor_configs:
            strategy = self.registry.build(strategy_name, neighbor)
            candidate_frame = self._load_enriched_frame_for_strategy(
                ticker=str(row["ticker"]),
                strategy=strategy,
                start_date=config.defaults.start,
                end_date=config.defaults.end,
            )
            walk_rows = run_walk_forward_for_strategies(
                ticker=row["ticker"],
                df=candidate_frame,
                strategies=[strategy],
                windows=windows,
                ratios=ratios,
                metrics=self.metrics,
                min_signals=config.defaults.min_signals,
                cost_bps=config.defaults.m1_cost_bps,
            )
            if walk_rows:
                for item in aggregate_walk_forward(walk_rows).iter_rows(named=True):
                    if item["direction"] == row["direction"]:
                        m1_records.append({**item, **neighbor})
            for cost_bps in costs:
                cost_rows = run_walk_forward_for_strategies(
                    ticker=row["ticker"],
                    df=candidate_frame,
                    strategies=[strategy],
                    windows=windows,
                    ratios=ratios,
                    metrics=self.metrics,
                    min_signals=config.defaults.min_signals,
                    cost_bps=cost_bps,
                )
                if not cost_rows:
                    continue
                cost_agg = aggregate_walk_forward(cost_rows).filter(pl.col("direction") == row["direction"])
                if cost_agg.is_empty():
                    continue
                m2_frames.append(
                    cost_agg.with_columns(pl.lit(cost_bps).alias("cost_bps")).with_columns(
                        [pl.lit(value).alias(key) for key, value in neighbor.items()]
                    )
                )
        m1_ranked = pl.DataFrame(m1_records) if m1_records else pl.DataFrame()
        m2_gate = build_gate_report(
            combined=pl.concat(m2_frames, how="diagonal_relaxed") if m2_frames else pl.DataFrame(),
            cost_count=len(costs),
            gate_min_oos_windows=config.defaults.gate_min_oos_windows,
            gate_min_oos_signals=config.defaults.gate_min_oos_signals,
            gate_min_pct_positive=config.defaults.gate_min_pct_positive,
            gate_min_exp_r=config.defaults.gate_min_exp_r,
        ) if m2_frames else pl.DataFrame()
        _write_frame_if_not_empty(m1_ranked, artifact_dir / "retune_m1_ranked.csv")
        _write_frame_if_not_empty(m2_gate, artifact_dir / "retune_m2_gate_report.csv")
        return {
            "queue_status": QUEUE_STATUS_EXECUTED,
            "latest_stage_reached": "M2",
            "latest_stage_decision": "retune_completed",
            "latest_artifact_dir": str(artifact_dir),
        }

    def _execute_expand_symbols(
        self,
        *,
        row: dict[str, Any],
        config: Any,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        strategy = build_candidate_strategy(self._candidate_stage_payload(row))
        symbols = [
            symbol for symbol in getattr(config, "tier2_watchlist", [])
            if symbol.upper() != str(row["ticker"]).upper()
        ]
        if not symbols:
            return {
                "queue_status": QUEUE_STATUS_EXECUTED,
                "latest_stage_reached": "M2",
                "latest_stage_decision": "expand_symbols_skipped",
                "latest_artifact_dir": str(artifact_dir),
            }
        windows = build_windows(
            config.defaults.start,
            config.defaults.end,
            config.defaults.train_months,
            config.defaults.test_months,
        )
        ratios = parse_csv_floats(config.defaults.ratios)
        costs = parse_csv_floats(config.defaults.cost_grid_bps)

        ticker_frames = {
            symbol: self._load_enriched_frame_for_strategy(
                ticker=symbol,
                strategy=strategy,
                start_date=config.defaults.start,
                end_date=config.defaults.end,
            )
            for symbol in symbols
        }
        detail_rows: list[dict[str, Any]] = []
        convergence_frames: list[pl.DataFrame] = []
        params = candidate_params(row)
        for symbol, frame in ticker_frames.items():
            if frame.is_empty():
                continue
            walk_rows = run_walk_forward_for_strategies(
                ticker=symbol,
                df=frame,
                strategies=[strategy],
                windows=windows,
                ratios=ratios,
                metrics=self.metrics,
                min_signals=config.defaults.min_signals,
                cost_bps=config.defaults.m1_cost_bps,
            )
            if walk_rows:
                detail_rows.extend(
                    {**item, **params}
                    for item in walk_rows
                    if item["direction"] == row["direction"]
                )
            for cost_bps in costs:
                cost_rows = run_walk_forward_for_strategies(
                    ticker=symbol,
                    df=frame,
                    strategies=[strategy],
                    windows=windows,
                    ratios=ratios,
                    metrics=self.metrics,
                    min_signals=config.defaults.min_signals,
                    cost_bps=cost_bps,
                )
                if not cost_rows:
                    continue
                cost_agg = aggregate_walk_forward(cost_rows).filter(pl.col("direction") == row["direction"])
                if cost_agg.is_empty():
                    continue
                convergence_frames.append(
                    cost_agg.with_columns(pl.lit(cost_bps).alias("cost_bps")).with_columns(
                        [pl.lit(value).alias(key) for key, value in params.items()]
                    )
                )
        m3_detail = pl.DataFrame(detail_rows) if detail_rows else pl.DataFrame()
        m2_gate = build_gate_report(
            combined=pl.concat(convergence_frames, how="diagonal_relaxed") if convergence_frames else pl.DataFrame(),
            cost_count=len(costs),
            gate_min_oos_windows=config.defaults.gate_min_oos_windows,
            gate_min_oos_signals=config.defaults.gate_min_oos_signals,
            gate_min_pct_positive=config.defaults.gate_min_pct_positive,
            gate_min_exp_r=config.defaults.gate_min_exp_r,
        ) if convergence_frames else pl.DataFrame()
        _write_frame_if_not_empty(m3_detail, artifact_dir / "expand_symbols_walk_forward_detail.csv")
        _write_frame_if_not_empty(m2_gate, artifact_dir / "expand_symbols_m2_gate_report.csv")
        return {
            "queue_status": QUEUE_STATUS_EXECUTED,
            "latest_stage_reached": "M2",
            "latest_stage_decision": "expand_symbols_completed",
            "latest_artifact_dir": str(artifact_dir),
        }

    def _eligible_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        run_date: date,
    ) -> list[dict[str, Any]]:
        eligible = []
        for row in rows:
            normalized = self._normalize_row(dict(row), run_date=run_date, manual_override=False)
            if normalized.get("queue_status") != QUEUE_STATUS_PENDING:
                continue
            if normalized.get("human_decision") not in ACTIONABLE_HUMAN_DECISIONS:
                continue
            eligible.append(normalized)
        eligible.sort(
            key=lambda row: (
                int(_intish(row.get("priority"), default=0)),
                parse_iso_datetime(row.get("human_updated_at")) or datetime.min.replace(tzinfo=timezone.utc),
                parse_iso_date(row.get("last_seen_run_date")) or date.min,
                float(row.get("m2_score") or float("-inf")),
            ),
            reverse=True,
        )
        return eligible

    def _normalize_row(
        self,
        row: dict[str, Any],
        *,
        run_date: date,
        manual_override: bool,
    ) -> dict[str, Any]:
        normalized = dict(row)
        normalized["candidate_key"] = str(normalized.get("candidate_key", ""))
        normalized["human_decision"] = str(normalized.get("human_decision", "") or "").strip()
        normalized["human_notes"] = str(normalized.get("human_notes", "") or "").strip()
        normalized["queue_status"] = str(normalized.get("queue_status", "") or "").strip().upper() or QUEUE_STATUS_NEW
        normalized["priority"] = _intish(normalized.get("priority"), default=0)
        normalized["passes_m1"] = _boolish(normalized.get("passes_m1"))
        normalized["passes_m2"] = _boolish(normalized.get("passes_m2"))
        normalized["passes_m3"] = _boolish(normalized.get("passes_m3"))
        normalized["passes_m4"] = _boolish(normalized.get("passes_m4"))
        normalized["passes_m5"] = _boolish(normalized.get("passes_m5"))
        normalized["is_full_m1_m5_survivor"] = all(
            [
                normalized["passes_m1"],
                normalized["passes_m2"],
                normalized["passes_m3"],
                normalized["passes_m4"],
                normalized["passes_m5"],
            ]
        )
        normalized.setdefault("last_action_run_date", "")
        normalized.setdefault("human_updated_at", "")
        if normalized["human_decision"] in ACTIONABLE_HUMAN_DECISIONS and not normalized["human_updated_at"]:
            normalized["human_updated_at"] = datetime.combine(
                run_date,
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).isoformat()
        if normalized["queue_status"] in TERMINAL_QUEUE_STATUSES and manual_override:
            normalized["queue_status"] = QUEUE_STATUS_PENDING
        elif normalized["queue_status"] not in TERMINAL_QUEUE_STATUSES and normalized["human_decision"] in ACTIONABLE_HUMAN_DECISIONS:
            if normalized["queue_status"] in {QUEUE_STATUS_NEW, QUEUE_STATUS_STALE, QUEUE_STATUS_ERROR}:
                normalized["queue_status"] = QUEUE_STATUS_PENDING
        return normalized

    def _render_chart(
        self,
        *,
        family: str,
        candidate: dict[str, Any],
        config: Any,
        candidate_key: str,
    ) -> Path:
        try:
            import plotly.graph_objects as go
            from plotly.io import write_html
        except ModuleNotFoundError as exc:  # pragma: no cover - validated in environment/tests
            raise ModuleNotFoundError(
                "Plotly is required for M2 review charts. Install the 'plotly' dependency."
            ) from exc

        chart_path = self.paths.charts_dir / f"{candidate_key}.html"
        strategy = build_candidate_strategy(candidate)
        raw = self._load_raw_chart_frame(ticker=str(candidate["ticker"]), config=config)
        if raw.is_empty():
            chart_path.write_text(
                "<html><body><p>No bars available for chart generation.</p></body></html>\n",
                encoding="utf-8",
            )
            return chart_path
        enriched = self.enricher(raw.clone(), required_feature_union([strategy]))
        signal_frame = strategy.generate_signals(enriched.clone())
        recent = tail_trading_days(signal_frame, days=3)
        if recent.is_empty():
            recent = signal_frame.tail(500)

        figure = go.Figure()
        figure.add_trace(
            go.Candlestick(
                x=recent["timestamp"].to_list(),
                open=recent["open"].to_list(),
                high=recent["high"].to_list(),
                low=recent["low"].to_list(),
                close=recent["close"].to_list(),
                name=str(candidate["ticker"]),
            )
        )
        long_entries = recent.filter(pl.col("signal") & (pl.col("signal_direction") == "long"))
        short_entries = recent.filter(pl.col("signal") & (pl.col("signal_direction") == "short"))
        if not long_entries.is_empty():
            figure.add_trace(
                go.Scatter(
                    x=long_entries["timestamp"].to_list(),
                    y=long_entries["close"].to_list(),
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 11, "color": "#2e8b57"},
                    name="Long Entry",
                )
            )
        if not short_entries.is_empty():
            figure.add_trace(
                go.Scatter(
                    x=short_entries["timestamp"].to_list(),
                    y=short_entries["close"].to_list(),
                    mode="markers",
                    marker={"symbol": "triangle-down", "size": 11, "color": "#b22222"},
                    name="Short Entry",
                )
            )
        params = candidate_params(candidate)
        figure.update_layout(
            title=(
                f"{candidate['ticker']} | {candidate['strategy']} | {candidate['direction']} | {family}"
                f"<br><sup>{json.dumps(json_ready(params), sort_keys=True)}</sup>"
            ),
            xaxis_title="Timestamp",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
        )
        write_html(figure, file=str(chart_path), full_html=True, include_plotlyjs=True)
        return chart_path

    def _write_review_bundle(self, rows: list[dict[str, Any]]) -> None:
        queue_df = pd.DataFrame(self._sort_rows(rows))
        history_df = pd.DataFrame(self._load_rows(self.paths.history_path))
        recent_history_df = history_df.copy()
        if not recent_history_df.empty and "run_date" in recent_history_df.columns:
            run_dates = sorted({str(item) for item in recent_history_df["run_date"].tolist()}, reverse=True)[:3]
            recent_history_df = recent_history_df[recent_history_df["run_date"].isin(run_dates)]

        execution_queue_df = queue_df[
            queue_df["queue_status"].isin([QUEUE_STATUS_PENDING, QUEUE_STATUS_EXECUTING, QUEUE_STATUS_ERROR])
        ] if not queue_df.empty else queue_df.copy()
        full_survivors_df = queue_df[
            queue_df["is_full_m1_m5_survivor"].astype(str).str.lower().isin(["true", "1"])
        ] if not queue_df.empty and "is_full_m1_m5_survivor" in queue_df.columns else queue_df.iloc[0:0].copy()
        charts_index_df = queue_df[
            [
                column for column in [
                    "candidate_key",
                    "ticker",
                    "strategy",
                    "direction",
                    "strategy_family",
                    "chart_link",
                    "queue_status",
                    "human_decision",
                    "latest_stage_reached",
                    "latest_stage_decision",
                    "is_full_m1_m5_survivor",
                    "last_seen_run_date",
                ] if column in queue_df.columns
            ]
        ] if not queue_df.empty else queue_df.copy()

        views = {
            "M2 Review": queue_df,
            "Recent History": recent_history_df,
            "Execution Queue": execution_queue_df,
            "Full Survivors": full_survivors_df,
            "Charts Index": charts_index_df,
        }
        self.paths.review_bundle_dir.mkdir(parents=True, exist_ok=True)
        for sheet_name, frame in views.items():
            csv_name = sheet_name.lower().replace(" ", "_") + ".csv"
            frame.to_csv(self.paths.review_bundle_dir / csv_name, index=False)
        try:
            with pd.ExcelWriter(self.paths.workbook_path, engine="xlsxwriter") as writer:
                for sheet_name, frame in views.items():
                    frame.to_excel(writer, index=False, sheet_name=sheet_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env package
            raise ModuleNotFoundError(
                "xlsxwriter is required for the review workbook export."
            ) from exc

    def _append_history(self, observations: list[dict[str, Any]], *, run_date: date) -> None:
        existing = self._load_rows(self.paths.history_path)
        by_key = {
            (row.get("run_date"), row.get("candidate_key")): row
            for row in existing
        }
        for observation in observations:
            history_row = {
                **observation,
                "run_date": run_date.isoformat(),
            }
            by_key[(run_date.isoformat(), observation["candidate_key"])] = history_row
        merged = sorted(
            by_key.values(),
            key=lambda row: (row.get("run_date", ""), row.get("candidate_key", "")),
            reverse=True,
        )
        self._write_rows(self.paths.history_path, merged)

    def _load_rows(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        return frame.to_dict(orient="records")

    def _write_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered_rows = self._sort_rows(rows)
        columns = self._queue_columns(ordered_rows)
        normalized_rows = [
            {column: csv_ready_value(row.get(column, "")) for column in columns}
            for row in ordered_rows
        ]
        pd.DataFrame(normalized_rows, columns=columns).to_csv(path, index=False)

    def _load_snapshot(self) -> dict[str, dict[str, Any]]:
        if not self.paths.snapshot_path.exists():
            return {}
        return json.loads(self.paths.snapshot_path.read_text(encoding="utf-8"))

    def _write_snapshot(self, rows: list[dict[str, Any]]) -> None:
        payload = {
            row["candidate_key"]: {field: row.get(field, "") for field in MANUAL_QUEUE_FIELDS}
            for row in rows
        }
        self.paths.snapshot_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _sort_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                int(_intish(row.get("priority"), default=0)),
                parse_iso_date(row.get("last_seen_run_date")) or date.min,
                float(row.get("m2_score") or float("-inf")),
                row.get("candidate_key", ""),
            ),
            reverse=True,
        )

    def _queue_columns(self, rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "candidate_key",
            "strategy_family",
            "ticker",
            "strategy",
            "direction",
            "config_signature",
            "config_json",
            "research_slice_id",
            "chart_link",
            "human_decision",
            "human_notes",
            "priority",
            "human_updated_at",
            "queue_status",
            "latest_stage_reached",
            "latest_stage_decision",
            "passes_m1",
            "passes_m2",
            "passes_m3",
            "passes_m4",
            "passes_m5",
            "is_full_m1_m5_survivor",
            "last_seen_run_date",
            "last_action_run_date",
            "last_source_run_dir",
            "latest_artifact_dir",
        ]
        dynamic_columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in preferred and key not in dynamic_columns:
                    dynamic_columns.append(key)
        param_columns = sorted(
            [
                key for key in dynamic_columns
                if not key.startswith(OBSERVATION_COLUMN_PREFIXES)
                and key not in {"run_date"}
            ]
        )
        metric_columns = sorted(
            [key for key in dynamic_columns if key.startswith(OBSERVATION_COLUMN_PREFIXES)]
        )
        tail_columns = ["run_date"] if any("run_date" in row for row in rows) else []
        return [
            *preferred,
            *param_columns,
            *metric_columns,
            *tail_columns,
        ]

    def _is_manual_override(
        self,
        *,
        row: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> bool:
        if row.get("queue_status") not in TERMINAL_QUEUE_STATUSES:
            return False
        if not snapshot:
            return False
        return any(str(row.get(field, "")) != str(snapshot.get(field, "")) for field in MANUAL_QUEUE_FIELDS)

    def _load_enriched_candidate_frame(self, *, row: dict[str, Any], config: Any) -> pl.DataFrame:
        strategy = build_candidate_strategy(self._candidate_stage_payload(row))
        return self._load_enriched_frame_for_strategy(
            ticker=str(row["ticker"]),
            strategy=strategy,
            start_date=config.defaults.start,
            end_date=config.defaults.end,
        )

    def _load_enriched_frame_for_strategy(
        self,
        *,
        ticker: str,
        strategy: Any,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        return self._load_enriched_frame_for_strategies(
            ticker=ticker,
            strategies=[strategy],
            start_date=start_date,
            end_date=end_date,
        )

    def _load_enriched_frame_for_strategies(
        self,
        *,
        ticker: str,
        strategies: list[Any],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        raw = self.frame_loader(ticker, start_date, end_date)
        if raw.is_empty():
            return raw
        return self.enricher(raw, required_feature_union(strategies))

    def _load_raw_chart_frame(self, *, ticker: str, config: Any) -> pl.DataFrame:
        recent_start = config.defaults.end - timedelta(days=14)
        raw = self.frame_loader(ticker, recent_start, config.defaults.end)
        if raw.is_empty():
            raw = self.frame_loader(ticker, config.defaults.start, config.defaults.end)
        return raw

    def _build_slice_id(self, *, family: str, config: Any) -> str:
        payload = {
            "family": family,
            "watchlist": list(config.watchlist),
            "start": config.defaults.start.isoformat(),
            "end": config.defaults.end.isoformat(),
            "calibration_end": config.defaults.calibration_end.isoformat(),
            "holdout_start": config.defaults.holdout_start.isoformat(),
            "holdout_end": config.defaults.holdout_end.isoformat(),
        }
        return f"{family}-{stable_signature(payload)}"

    def _candidate_stage_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": row["ticker"],
            "strategy": row["strategy"],
            "direction": row["direction"],
            **self._queue_config_params(row),
        }

    def _queue_config_params(self, row: dict[str, Any]) -> dict[str, Any]:
        config_json = row.get("config_json")
        if config_json not in (None, ""):
            try:
                loaded = json.loads(str(config_json))
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(loaded, dict):
                    return {str(key): value for key, value in loaded.items()}
        return candidate_params(row)

    def _catalog_strategy_name(self, row: dict[str, Any]) -> str:
        family = str(row.get("strategy_family", "") or "").strip()
        return FAMILY_CATALOG_STRATEGY_NAMES.get(family, str(row["strategy"]))

    def _index_frame(
        self,
        frame: pl.DataFrame,
        *,
        research_slice_id: str,
        family: str,
    ) -> dict[str, dict[str, Any]]:
        if frame.is_empty():
            return {}
        indexed: dict[str, dict[str, Any]] = {}
        for row in frame.iter_rows(named=True):
            candidate_key = build_review_candidate_key(
                family=family,
                ticker=str(row["ticker"]),
                direction=str(row["direction"]),
                strategy=str(row["strategy"]),
                params=candidate_params(row),
                research_slice_id=research_slice_id,
            )
            indexed[candidate_key] = {**row, "candidate_key": candidate_key}
        return indexed


def build_review_candidate_key(
    *,
    family: str,
    ticker: str,
    direction: str,
    strategy: str,
    params: dict[str, Any],
    research_slice_id: str,
) -> str:
    payload = {
        "family": family,
        "ticker": ticker.upper(),
        "direction": direction,
        "strategy": strategy,
        "params": json_ready(params),
        "research_slice_id": research_slice_id,
    }
    return stable_signature(payload, length=24)


def build_neighbor_configs(
    entry: StrategyCatalogEntry,
    base_config: dict[str, Any],
    *,
    max_configs: int,
) -> list[dict[str, Any]]:
    if entry.search_spec is None:
        return [dict(base_config)]
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(config: dict[str, Any]) -> None:
        normalized = entry.search_spec.normalize_config(config, base_config=entry.strategy_config)
        if normalized.errors:
            return
        signature = stable_signature(normalized.config)
        if signature in seen:
            return
        seen.add(signature)
        configs.append(normalized.config)

    add(base_config)
    for parameter in entry.search_spec.parameters:
        current = base_config.get(parameter.name, parameter.default)
        legal_values = parameter.legal_values()
        if current not in legal_values:
            add({**base_config, parameter.name: parameter.default})
            continue
        indices: set[int] = {legal_values.index(current)}
        if parameter.type == "discrete":
            current_index = legal_values.index(current)
            if current_index > 0:
                indices.add(current_index - 1)
            if current_index + 1 < len(legal_values):
                indices.add(current_index + 1)
        else:
            indices.update(range(len(legal_values)))
        for index in sorted(indices):
            add({**base_config, parameter.name: legal_values[index]})
            if len(configs) >= max_configs:
                return configs[:max_configs]
    return configs[:max_configs]


def tail_trading_days(frame: pl.DataFrame, *, days: int) -> pl.DataFrame:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return frame
    trade_dates = (
        frame.select(pl.col("timestamp").dt.date().alias("trade_date"))
        .unique()
        .sort("trade_date")
        .get_column("trade_date")
        .to_list()
    )
    if not trade_dates:
        return frame
    selected_dates = trade_dates[-days:]
    return frame.filter(pl.col("timestamp").dt.date().is_in(selected_dates))


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Could not parse floats from: {value}")
    return values


def stable_signature(payload: Any, *, length: int = 16) -> str:
    canonical = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def prefix_keys(
    row: dict[str, Any],
    prefix: str,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude or set()
    return {
        f"{prefix}{key}": value
        for key, value in row.items()
        if key not in excluded
    }


def parse_iso_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def parse_iso_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def csv_ready_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    return value


def _latest_stage_state(
    *,
    passes_m1: bool,
    passes_m2: bool,
    passes_m3: bool,
    passes_m4: bool,
    passes_m5: bool,
    m2_decision: Any,
    m4_decision: Any,
) -> tuple[str, str]:
    if passes_m5:
        return "M5", "promote"
    if passes_m4:
        return "M4", str(m4_decision or "promote_to_execution_mapping")
    if passes_m3:
        return "M3", "walk_forward_completed"
    if passes_m2:
        return "M2", str(m2_decision or "promote_to_holdout")
    if passes_m1:
        return "M1", "selected_for_convergence"
    return "M0", "unseen"


def _strategy_key_for_name(strategy_name: str) -> str:
    return {
        "Market Impulse (Cross & Reclaim)": "market_impulse",
        "Jerk-Pivot Momentum (tight)": "jerk_pivot_momentum",
        "Elastic Band Reversion": "elastic_band_reversion",
        "Opening Drive Classifier": "opening_drive_classifier",
    }.get(strategy_name, strategy_name.lower().replace(" ", "_"))


def _catastrophe_exit_defaults(m5_row: dict[str, Any]) -> dict[str, Any]:
    risk_rule = str(m5_row.get("risk_rule") or "").lower()
    stop_loss_pct = 0.45
    if "-35%" in risk_rule:
        stop_loss_pct = 0.35
    elif "-45%" in risk_rule:
        stop_loss_pct = 0.45
    return {
        "stop_loss_pct": stop_loss_pct,
        "hard_flat_time_et": "15:55",
        "use_profit_target": False,
        "profit_target_multiple": None,
        "stop_to_breakeven_after_r_multiple": None,
    }


def _read_optional_csv(run_dir: Path, *names: str) -> pl.DataFrame:
    for name in names:
        path = run_dir / name
        if path.exists():
            try:
                return pl.read_csv(path)
            except NoDataError:
                return pl.DataFrame()
    return pl.DataFrame()


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def _intish(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _append_note(existing: str, note: str) -> str:
    return note if not existing else f"{existing}\n{note}"


def _has_followup_error_note(notes: Any) -> bool:
    return "follow-up error:" in str(notes or "")


def _strip_followup_error_notes(notes: Any) -> str:
    cleaned = [
        line.strip()
        for line in str(notes or "").splitlines()
        if "follow-up error:" not in line
    ]
    return "\n".join(line for line in cleaned if line)


def _write_frame_if_not_empty(frame: pl.DataFrame, path: Path) -> None:
    if frame.is_empty():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [
        {key: _csv_safe_value(value) for key, value in row.items()}
        for row in frame.iter_rows(named=True)
    ]
    pl.DataFrame(safe_rows).write_csv(path)


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(json_ready(value), sort_keys=True)
    return value


__all__ = [
    "ACTIONABLE_HUMAN_DECISIONS",
    "HumanReviewQueueManager",
    "QUEUE_STATUS_EXECUTED",
    "QUEUE_STATUS_KILLED",
    "QUEUE_STATUS_NEW",
    "QUEUE_STATUS_PENDING",
    "QUEUE_STATUS_STALE",
    "ReviewQueueArtifacts",
    "ReviewQueuePaths",
    "TERMINAL_QUEUE_STATUSES",
    "build_neighbor_configs",
    "build_review_candidate_key",
    "parse_csv_floats",
    "resolve_review_queue_paths",
    "stable_signature",
    "tail_trading_days",
]
