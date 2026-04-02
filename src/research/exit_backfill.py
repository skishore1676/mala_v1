"""One-time backfill utilities for optimized thesis exits on older M5 survivors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl

from src.chronos.storage import LocalStorage
from src.config import PROJECT_ROOT
from src.newton.engine import PhysicsEngine
from src.oracle.metrics import MetricsCalculator
from src.research.exit_optimizer import optimize_underlying_exit, write_exit_optimization_result
from src.research.playbooks import augment_playbook_catalog_from_queue
from src.research.review_queue import json_ready
from src.research.stages.candidates import build_candidate_strategy
from src.strategy.base import required_feature_union


@dataclass(slots=True, frozen=True)
class ExitBackfillResult:
    queue_path: Path
    processed: int
    optimized: int
    skipped_existing: int
    skipped_ineligible: int
    playbook_catalog_path: Path | None


def backfill_exit_optimizations(
    *,
    queue_path: str | Path,
    start_date: date,
    end_date: date,
    holdout_start: date,
    holdout_end: date,
    playbook_catalog_path: str | Path | None = None,
    playbook_projection_path: str | Path | None = None,
    force: bool = False,
    frame_loader: Callable[[str, date | None, date | None], pl.DataFrame] | None = None,
    enricher: Callable[[pl.DataFrame, set[str]], pl.DataFrame] | None = None,
) -> ExitBackfillResult:
    queue = Path(queue_path)
    rows = _load_rows(queue)
    storage = LocalStorage()
    physics = PhysicsEngine()
    _ = MetricsCalculator()
    resolved_frame_loader = frame_loader or storage.load_bars
    resolved_enricher = enricher or physics.enrich_for_features

    processed = 0
    optimized = 0
    skipped_existing = 0
    skipped_ineligible = 0

    for row in rows:
        if not _eligible_row(row):
            skipped_ineligible += 1
            continue
        artifact_dir = _artifact_dir(row)
        if artifact_dir is None:
            skipped_ineligible += 1
            continue
        output_path = artifact_dir / "m5_exit_optimization.json"
        if output_path.exists() and not force:
            skipped_existing += 1
            continue
        processed += 1
        strategy = build_candidate_strategy(_candidate_stage_payload(row))
        raw = resolved_frame_loader(str(row["ticker"]), start_date, end_date)
        if raw.is_empty():
            skipped_ineligible += 1
            continue
        enriched = resolved_enricher(raw, required_feature_union([strategy]))
        m5_row = _load_m5_row(artifact_dir)
        if m5_row is None:
            skipped_ineligible += 1
            continue
        result = optimize_underlying_exit(
            strategy_key=_strategy_key_for_name(str(row["strategy"])),
            symbol=str(row["ticker"]),
            direction=str(row["direction"]),
            strategy=strategy,
            enriched_frame=enriched,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            catastrophe_exit_params=_catastrophe_exit_defaults(m5_row),
        )
        if result is None:
            skipped_ineligible += 1
            continue
        write_exit_optimization_result(result, path=output_path)
        optimized += 1

    resolved_catalog: Path | None = None
    if playbook_catalog_path is not None:
        resolved_catalog, _ = augment_playbook_catalog_from_queue(
            playbook_catalog_path=playbook_catalog_path,
            queue_path=queue,
            playbook_projection_path=playbook_projection_path,
        )

    return ExitBackfillResult(
        queue_path=queue.resolve(),
        processed=processed,
        optimized=optimized,
        skipped_existing=skipped_existing,
        skipped_ineligible=skipped_ineligible,
        playbook_catalog_path=resolved_catalog.resolve() if resolved_catalog is not None else None,
    )


def _eligible_row(row: dict[str, Any]) -> bool:
    return _boolish(row.get("is_full_m1_m5_survivor")) and str(row.get("latest_stage_reached", "")).upper() == "M5"


def _artifact_dir(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("latest_artifact_dir") or row.get("last_source_run_dir") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _candidate_stage_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": row["ticker"],
        "strategy": row["strategy"],
        "direction": row["direction"],
        **_queue_config_params(row),
    }


def _queue_config_params(row: dict[str, Any]) -> dict[str, Any]:
    config_json = row.get("config_json")
    if config_json not in (None, ""):
        try:
            loaded = json.loads(str(config_json))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded
    return {}


def _load_m5_row(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / "m5_execution_mapping.csv"
    if not path.exists():
        return None
    frame = pl.read_csv(path)
    if frame.is_empty():
        return None
    return frame.row(0, named=True)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
    return json_ready(
        {
            "stop_loss_pct": stop_loss_pct,
            "hard_flat_time_et": "15:55",
            "use_profit_target": False,
            "profit_target_multiple": None,
            "stop_to_breakeven_after_r_multiple": None,
        }
    )
