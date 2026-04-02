"""Playbook catalog, bias routing, and armed deployment contracts for Mala."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any, Literal

import polars as pl
from polars.exceptions import NoDataError
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

from src.config import PROJECT_ROOT
from src.research.google_sheets import GoogleSheetTableClient, spreadsheet_id_from_url
from src.research.loop_contracts import PLAYBOOK_CATALOG_CONTRACT_NAME, build_contract_metadata
from src.research.loop_export import (
    LoopArtifactExporter,
    _SUPPORTED_EXECUTION_PRESETS,
    _bias_template,
    _jsonable,
    _maybe_float,
    _maybe_int,
    _parse_window,
    _strategy_descriptor,
)
from src.research.review_queue import QUEUE_STATUS_KILLED, QUEUE_STATUS_STALE, json_ready, stable_signature


PLAYBOOK_STATUS_ACTIVE = "active"
PLAYBOOK_STATUS_STALE = "stale"
PLAYBOOK_STATUS_RETIRED = "retired"

_PLAYBOOK_STATUS_PRIORITY = {
    PLAYBOOK_STATUS_ACTIVE: 2,
    PLAYBOOK_STATUS_STALE: 1,
    PLAYBOOK_STATUS_RETIRED: 0,
}

_SUPPORTED_STRATEGY_KEYS = set(_SUPPORTED_EXECUTION_PRESETS)

_DAILY_BIAS_MAP = {
    "bullish": "bullish",
    "bearish": "bearish",
}

_THESIS_TEMPLATE_MAP = {
    ("bullish", "mean_reversion"): "bullish_mean_reversion_intraday",
    ("bearish", "mean_reversion"): "bearish_mean_reversion_intraday",
    ("bullish", "chop_fade"): "bullish_mean_reversion_intraday",
    ("bearish", "chop_fade"): "bearish_mean_reversion_intraday",
    ("bullish", "opening_drive_failure"): "bullish_mean_reversion_intraday",
    ("bearish", "opening_drive_failure"): "bearish_mean_reversion_intraday",
    ("bullish", "trend_continuation"): "bullish_trend_intraday",
    ("bearish", "trend_continuation"): "bearish_trend_intraday",
    ("bullish", "opening_drive_followthrough"): "bullish_trend_intraday",
    ("bearish", "opening_drive_followthrough"): "bearish_trend_intraday",
}

_ALLOWED_DAILY_BIAS = {"Bullish", "Bearish"}
_ALLOWED_INTRADAY_THESIS = {
    "Mean_Reversion",
    "Trend_Continuation",
    "Chop_Fade",
    "Opening_Drive_Followthrough",
    "Opening_Drive_Failure",
}


class PlaybookRecord(BaseModel):
    playbook_id: str
    strategy_key: str
    strategy_family: str
    strategy_display_name: str
    symbol: str
    symbol_scope: list[str] = Field(default_factory=list)
    direction: str
    bias_template: str
    horizon: str = "intraday"
    regime_tags: list[str] = Field(default_factory=list)
    lifecycle_status: Literal["active", "stale", "retired"] = PLAYBOOK_STATUS_ACTIVE
    automation_status: str
    surface_class: str
    entry_params: dict[str, Any] = Field(default_factory=dict)
    exit_params: dict[str, Any] = Field(default_factory=dict)
    execution_mapping: dict[str, Any] = Field(default_factory=dict)
    risk_mapping: dict[str, Any] = Field(default_factory=dict)
    deployment_manifest_template: dict[str, Any] = Field(default_factory=dict)
    time_window: dict[str, Any] = Field(default_factory=dict)
    vehicle_mapping: dict[str, Any] = Field(default_factory=dict)
    passes_m1: bool = False
    passes_m2: bool = False
    passes_m3: bool = False
    passes_m4: bool = False
    passes_m5: bool = False
    is_full_m1_m5_survivor: bool = False
    expectancy: float | None = None
    confidence: float | None = None
    signal_count: int | None = None
    execution_robustness: float | None = None
    stress_metrics: dict[str, Any] = Field(default_factory=dict)
    last_validated_date: str
    research_slice_id: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    bhiksha_compatibility: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: Any) -> str:
        return str(value).upper()


class BiasInputRow(BaseModel):
    date: date
    symbol: str
    daily_bias: str
    intraday_thesis: str
    max_risk_usd: float
    translator_status: str = ""
    armed_playbook_id: str = ""
    notes: str = ""
    enabled: bool = True
    after_time_et: str | None = None
    only_if_price_crosses: str | None = None
    translator_notes: str = ""

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: Any) -> str:
        return str(value).upper()

    @field_validator("date", mode="before")
    @classmethod
    def parse_date_value(cls, value: Any) -> date:
        if isinstance(value, date):
            return value
        raw = str(value).strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%-m/%-d/%y", "%-m/%-d/%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unsupported Date format: {value!r}")

    @field_validator("daily_bias", mode="before")
    @classmethod
    def validate_daily_bias(cls, value: Any) -> str:
        normalized = str(value).strip().title().replace(" ", "_")
        normalized = normalized.replace("_", " ")
        titled = normalized.title()
        if titled not in _ALLOWED_DAILY_BIAS:
            raise ValueError(f"Unsupported Daily_Bias: {value!r}")
        return titled

    @field_validator("intraday_thesis", mode="before")
    @classmethod
    def validate_intraday_thesis(cls, value: Any) -> str:
        raw = str(value).strip()
        normalized = "_".join(part for part in raw.replace(" ", "_").split("_") if part)
        canonical = "_".join(word.capitalize() for word in normalized.split("_"))
        if canonical not in _ALLOWED_INTRADAY_THESIS:
            raise ValueError(f"Unsupported Intraday_Thesis: {value!r}")
        return canonical

    @field_validator("enabled", mode="before")
    @classmethod
    def parse_enabled(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return True
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    @model_validator(mode="after")
    def normalize_optional_text(self) -> "BiasInputRow":
        self.translator_status = str(self.translator_status or "").strip()
        self.armed_playbook_id = str(self.armed_playbook_id or "").strip()
        self.notes = str(self.notes or "").strip()
        self.after_time_et = _optional_text(self.after_time_et)
        self.only_if_price_crosses = _optional_text(self.only_if_price_crosses)
        self.translator_notes = str(self.translator_notes or "").strip()
        return self

    @property
    def bias_template(self) -> str:
        key = (
            _DAILY_BIAS_MAP[self.daily_bias.lower()],
            self.intraday_thesis.lower(),
        )
        try:
            return _THESIS_TEMPLATE_MAP[key]
        except KeyError as exc:
            raise ValueError(
                f"No bias template mapping for Daily_Bias={self.daily_bias!r}, "
                f"Intraday_Thesis={self.intraday_thesis!r}"
            ) from exc

    @property
    def horizon(self) -> str:
        return "intraday"

    @property
    def routing_key(self) -> str:
        return f"{self.symbol}|{self.bias_template}|{self.horizon}"


class TranslatorSelection(BaseModel):
    symbol: str
    bias_template: str
    horizon: str = "intraday"
    status: str
    selected_playbook_id: str | None = None
    reason: str = ""
    translator_notes: str = ""
    ranking_score: float | None = None
    deployment_id: str | None = None


class LiveObservationRecord(BaseModel):
    playbook_id: str
    deployment_id: str
    symbol: str
    armed: bool
    triggered: bool
    exit_exercised: bool
    realized_outcome_summary: dict[str, Any] = Field(default_factory=dict)
    armed_at: str | None = None
    triggered_at: str | None = None
    exit_at: str | None = None
    recorded_at: str | None = None


def augment_playbook_catalog_from_queue(
    *,
    playbook_catalog_path: str | Path,
    queue_path: str | Path,
    playbook_projection_path: str | Path | None = None,
    project_root: Path | None = None,
) -> tuple[Path, Path | None]:
    catalog_path = Path(playbook_catalog_path)
    queue_rows = _load_csv_rows(Path(queue_path))
    playbooks = build_playbook_records_from_queue(queue_rows, project_root=project_root)
    payload = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {
        **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
        "generated_at": datetime.now(UTC).isoformat(),
        "contexts": {},
    }
    payload["playbooks"] = [playbook.model_dump(mode="json") for playbook in playbooks]
    payload["playbook_count"] = len(playbooks)
    payload["playbook_status_counts"] = {
        status: sum(1 for playbook in playbooks if playbook.lifecycle_status == status)
        for status in (PLAYBOOK_STATUS_ACTIVE, PLAYBOOK_STATUS_STALE, PLAYBOOK_STATUS_RETIRED)
    }
    catalog_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    projection_path = Path(playbook_projection_path) if playbook_projection_path is not None else None
    if projection_path is not None:
        _write_playbook_projection_csv(projection_path, playbooks)
    return catalog_path, projection_path


def build_playbook_records_from_queue(
    queue_rows: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[PlaybookRecord]:
    exporter = LoopArtifactExporter(project_root=project_root)
    records_by_id: dict[str, PlaybookRecord] = {}
    for row in queue_rows:
        record = _playbook_from_queue_row(row, exporter=exporter)
        if record is None:
            continue
        existing = records_by_id.get(record.playbook_id)
        if existing is None or _playbook_sort_key(record) > _playbook_sort_key(existing):
            records_by_id[record.playbook_id] = record
    return sorted(
        records_by_id.values(),
        key=lambda record: (
            record.symbol,
            record.bias_template,
            -_PLAYBOOK_STATUS_PRIORITY[record.lifecycle_status],
            -(record.expectancy or float("-inf")),
            record.playbook_id,
        ),
    )


def route_bias_inputs(
    *,
    bias_inputs_path: str | Path,
    playbook_catalog_path: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path, list[TranslatorSelection]]:
    biases = load_bias_inputs_sheet(bias_inputs_path)
    routing_report_path, armed_payloads_path, selections, _ = route_bias_rows(
        biases=biases,
        playbook_catalog_path=playbook_catalog_path,
        out_dir=out_dir,
        bias_source=str(Path(bias_inputs_path).resolve()),
    )
    return routing_report_path, armed_payloads_path, selections


def route_bias_rows(
    *,
    biases: list[BiasInputRow],
    playbook_catalog_path: str | Path,
    out_dir: str | Path,
    bias_source: str = "in_memory",
) -> tuple[Path, Path, list[TranslatorSelection], list[dict[str, Any]]]:
    playbooks = load_playbook_records(playbook_catalog_path)
    grouped_playbooks = _group_playbooks(playbooks)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = target_dir / "bhiksha_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    selections: list[TranslatorSelection] = []
    updated_rows: list[dict[str, Any]] = []
    armed_payloads: list[dict[str, Any]] = []
    for bias in biases:
        selection, playbook = _select_playbook_for_bias(bias, grouped_playbooks)
        updated_rows.append(_bias_row_output(bias, selection))
        selections.append(selection)
        if playbook is None or selection.selected_playbook_id is None:
            continue
        deployment_manifest = build_armed_deployment_manifest(playbook, bias)
        manifest_path = manifests_dir / f"{deployment_manifest['deployment_id']}.yaml"
        manifest_path.write_text(yaml.safe_dump(deployment_manifest, sort_keys=False), encoding="utf-8")
        armed_payloads.append(
            {
                "playbook_id": playbook.playbook_id,
                "symbol": playbook.symbol,
                "bias_template": bias.bias_template,
                "deployment_id": deployment_manifest["deployment_id"],
                "manifest_path": str(manifest_path),
                "manifest": deployment_manifest,
            }
        )
        selection.deployment_id = deployment_manifest["deployment_id"]
        selection.translator_notes = _append_notes(
            selection.translator_notes,
            f"armed_manifest={manifest_path.name}",
        )

    routing_report_path = target_dir / "bias_routing_report.csv"
    armed_payloads_path = target_dir / "armed_playbooks.json"
    _write_csv_rows(routing_report_path, updated_rows)
    armed_payloads_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "bias_inputs_path": bias_source,
                "playbook_catalog_path": str(Path(playbook_catalog_path).resolve()),
                "armed_playbooks": armed_payloads,
                "selections": [selection.model_dump(mode="json") for selection in selections],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return routing_report_path, armed_payloads_path, selections, updated_rows


def route_google_sheet_bias_inputs(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    credentials_path: str | Path,
    playbook_catalog_path: str | Path,
    out_dir: str | Path,
    update_sheet: bool = True,
) -> tuple[Path, Path, list[TranslatorSelection]]:
    client = GoogleSheetTableClient(
        spreadsheet_id=spreadsheet_id,
        sheet_name=sheet_name,
        credentials_path=Path(credentials_path),
    )
    raw_rows = client.read_rows()
    biases = [_bias_row_from_sheet_row(row) for row in raw_rows]
    routing_report_path, armed_payloads_path, selections, updated_rows = route_bias_rows(
        biases=biases,
        playbook_catalog_path=playbook_catalog_path,
        out_dir=out_dir,
        bias_source=f"google_sheet:{spreadsheet_id_from_url(spreadsheet_id)}:{sheet_name}",
    )
    if update_sheet:
        sheet_updates = []
        for raw_row, updated_row in zip(raw_rows, updated_rows, strict=False):
            sheet_updates.append(
                {
                    "row_index": raw_row["row_index"],
                    "Translator_Status": updated_row.get("Translator_Status", ""),
                    "Armed_Playbook_ID": updated_row.get("Armed_Playbook_ID", ""),
                    "Translator_Notes": updated_row.get("Translator_Notes", ""),
                }
            )
        client.batch_update_rows(
            rows=sheet_updates,
            columns=["Translator_Status", "Armed_Playbook_ID", "Translator_Notes"],
        )
    return routing_report_path, armed_payloads_path, selections


def build_armed_deployment_manifest(playbook: PlaybookRecord, bias: BiasInputRow) -> dict[str, Any]:
    deployment_id = _deployment_id_from_playbook(playbook=playbook, bias=bias)
    manifest = json.loads(json.dumps(playbook.deployment_manifest_template))
    manifest["deployment_id"] = deployment_id
    manifest["enabled"] = True
    manifest["symbol"] = bias.symbol
    manifest.setdefault("risk", {})
    manifest["risk"]["max_trade_premium_usd"] = float(bias.max_risk_usd)
    manifest.setdefault("source", {})
    manifest["source"]["origin"] = "mala_bias_translator_v1"
    manifest["source"]["run_date"] = bias.date.isoformat()
    manifest["source"].setdefault("metadata", {})
    manifest["source"]["metadata"].update(
        {
            "playbook_id": playbook.playbook_id,
            "bias_template": bias.bias_template,
            "daily_bias": bias.daily_bias,
            "intraday_thesis": bias.intraday_thesis,
            "notes": bias.notes,
            "after_time_et": bias.after_time_et,
            "only_if_price_crosses": bias.only_if_price_crosses,
            "expectancy": playbook.expectancy,
            "confidence": playbook.confidence,
            "last_validated_date": playbook.last_validated_date,
        }
    )
    return manifest


def write_live_observation_records(records: list[LiveObservationRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "records": [record.model_dump(mode="json") for record in records],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_live_observation_records(path: str | Path) -> list[LiveObservationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LiveObservationRecord.model_validate(item) for item in payload.get("records", [])]


def load_playbook_records(path: str | Path) -> list[PlaybookRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PlaybookRecord.model_validate(item) for item in payload.get("playbooks", [])]


def load_bias_inputs_sheet(path: str | Path) -> list[BiasInputRow]:
    rows = _load_csv_rows(Path(path))
    parsed: list[BiasInputRow] = []
    for row in rows:
        parsed.append(_bias_row_from_sheet_row(row))
    return parsed


def _playbook_from_queue_row(
    row: dict[str, Any],
    *,
    exporter: LoopArtifactExporter,
) -> PlaybookRecord | None:
    if not _boolish(row.get("is_full_m1_m5_survivor")):
        return None
    artifact_dir_raw = str(row.get("latest_artifact_dir", "")).strip() or str(row.get("last_source_run_dir", "")).strip()
    if not artifact_dir_raw:
        return None
    artifact_dir = Path(artifact_dir_raw)
    m5_detail = _read_optional_csv(artifact_dir, "m5_execution_mapping.csv", "M5_execution_mapping.csv")
    if m5_detail.is_empty():
        return None
    m5_row = m5_detail.row(0, named=True)
    m4_summary = _read_optional_csv(artifact_dir, "m4_holdout_summary.csv", "M4_holdout_summary.csv")
    m4_row = m4_summary.row(0, named=True) if not m4_summary.is_empty() else {}

    descriptor = _strategy_descriptor(str(row["strategy"]))
    strategy_key = descriptor.strategy_key
    surface_class = "supported" if strategy_key in _SUPPORTED_STRATEGY_KEYS else "proposed"
    automation_status = "shadow_ready" if surface_class == "supported" else "manual_research_only"
    strategy_params = _queue_strategy_params(row)
    manifest = exporter._build_manifest(  # noqa: SLF001 - reuse loop-export manifest contract
        descriptor=descriptor,
        symbol=str(row["ticker"]),
        direction=str(row["direction"]),
        strategy_params=strategy_params,
        surface_class=surface_class,
        execution_row=m5_row,
    )
    manifest["source"] = {
        "origin": "mala_playbook_catalog_v1",
        "run_date": str(row.get("last_action_run_date") or row.get("last_seen_run_date") or ""),
        "artifact": str(artifact_dir),
        "metadata": {
            "candidate_key": row.get("candidate_key"),
            "research_slice_id": row.get("research_slice_id"),
            "latest_stage_reached": row.get("latest_stage_reached"),
            "queue_status": row.get("queue_status"),
        },
    }
    playbook_id = _build_playbook_id(
        strategy_key=strategy_key,
        symbol=str(row["ticker"]),
        direction=str(row["direction"]),
        entry_params=strategy_params,
        execution_mapping=manifest["execution"],
        exit_mapping=manifest["exit"],
    )
    time_start, time_end = _manifest_window(manifest)
    lifecycle_status = _lifecycle_status(row)
    bias_template = _bias_template(strategy_key, str(row["direction"]))
    confidence = _maybe_float(row.get("m2_mean_test_confidence")) or _maybe_float(row.get("m1_avg_test_confidence"))
    signal_count = _maybe_int(m5_row.get("holdout_trades")) or _maybe_int(row.get("m1_oos_signals"))
    expectancy = _maybe_float(m5_row.get("base_exp_r")) or _maybe_float(m4_row.get("mean_holdout_exp_r"))
    execution_robustness = _maybe_float(m5_row.get("mc_prob_positive_exp"))
    bhiksha_supported = surface_class == "supported"
    record = PlaybookRecord(
        playbook_id=playbook_id,
        strategy_key=strategy_key,
        strategy_family=str(row.get("strategy_family", strategy_key)),
        strategy_display_name=str(row["strategy"]),
        symbol=str(row["ticker"]),
        symbol_scope=[str(row["ticker"]).upper()],
        direction=str(row["direction"]),
        bias_template=bias_template,
        regime_tags=_regime_tags(bias_template, strategy_key),
        lifecycle_status=lifecycle_status,
        automation_status=automation_status,
        surface_class=surface_class,
        entry_params={**strategy_params, "direction": str(row["direction"])},
        exit_params=json_ready(manifest["exit"]),
        execution_mapping=json_ready(manifest["execution"]),
        risk_mapping=json_ready(manifest["risk"]),
        deployment_manifest_template=json_ready(manifest),
        time_window={
            "entry_window_start_et": time_start,
            "entry_window_end_et": time_end,
        },
        vehicle_mapping={
            "profile": manifest["execution"].get("profile"),
            "option_mapping": json_ready(manifest["execution"].get("option_mapping", {})),
            "dte_min": manifest["execution"].get("dte_min"),
            "dte_max": manifest["execution"].get("dte_max"),
        },
        passes_m1=_boolish(row.get("passes_m1")),
        passes_m2=_boolish(row.get("passes_m2")),
        passes_m3=_boolish(row.get("passes_m3")),
        passes_m4=_boolish(row.get("passes_m4")),
        passes_m5=_boolish(row.get("passes_m5")),
        is_full_m1_m5_survivor=_boolish(row.get("is_full_m1_m5_survivor")),
        expectancy=expectancy,
        confidence=confidence,
        signal_count=signal_count,
        execution_robustness=execution_robustness,
        stress_metrics={
            "mc_exp_r_mean": _maybe_float(m5_row.get("mc_exp_r_mean")),
            "mc_exp_r_p05": _maybe_float(m5_row.get("mc_exp_r_p05")),
            "mc_exp_r_p50": _maybe_float(m5_row.get("mc_exp_r_p50")),
            "mc_exp_r_p95": _maybe_float(m5_row.get("mc_exp_r_p95")),
            "mc_prob_positive_exp": execution_robustness,
            "mc_total_r_p05": _maybe_float(m5_row.get("mc_total_r_p05")),
            "mc_total_r_p50": _maybe_float(m5_row.get("mc_total_r_p50")),
            "mc_total_r_p95": _maybe_float(m5_row.get("mc_total_r_p95")),
            "mc_max_dd_p50": _maybe_float(m5_row.get("mc_max_dd_p50")),
        },
        last_validated_date=str(row.get("last_action_run_date") or row.get("last_seen_run_date") or ""),
        research_slice_id=_optional_text(row.get("research_slice_id")),
        source={
            "queue_candidate_key": row.get("candidate_key"),
            "artifact_dir": str(artifact_dir),
            "chart_link": row.get("chart_link", ""),
            "config_signature": row.get("config_signature", ""),
            "m4_summary": _jsonable(m4_row),
            "m5_execution": _jsonable(m5_row),
        },
        bhiksha_compatibility={
            "supported": bhiksha_supported,
            "required_capabilities": (
                [] if bhiksha_supported else ["strategy_plugin_not_yet_implemented_in_bhiksha"]
            ),
        },
    )
    return record


def _bias_row_from_sheet_row(row: dict[str, Any]) -> BiasInputRow:
    normalized = {
        "date": row.get("Date"),
        "symbol": row.get("Symbol"),
        "daily_bias": row.get("Daily_Bias"),
        "intraday_thesis": row.get("Intraday_Thesis"),
        "max_risk_usd": row.get("Max_Risk_USD"),
        "translator_status": row.get("Translator_Status", ""),
        "armed_playbook_id": row.get("Armed_Playbook_ID", ""),
        "notes": row.get("Notes", ""),
        "enabled": row.get("Enabled", True),
        "after_time_et": row.get("After_Time_ET"),
        "only_if_price_crosses": row.get("Only_If_Price_Crosses"),
        "translator_notes": row.get("Translator_Notes", ""),
    }
    return BiasInputRow.model_validate(normalized)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _write_playbook_projection_csv(path: Path, playbooks: list[PlaybookRecord]) -> None:
    rows = [
        {
            "playbook_id": playbook.playbook_id,
            "symbol": playbook.symbol,
            "bias_template": playbook.bias_template,
            "strategy_key": playbook.strategy_key,
            "direction": playbook.direction,
            "lifecycle_status": playbook.lifecycle_status,
            "automation_status": playbook.automation_status,
            "expectancy": playbook.expectancy,
            "confidence": playbook.confidence,
            "signal_count": playbook.signal_count,
            "execution_robustness": playbook.execution_robustness,
            "last_validated_date": playbook.last_validated_date,
            "is_full_m1_m5_survivor": playbook.is_full_m1_m5_survivor,
        }
        for playbook in playbooks
    ]
    _write_csv_rows(path, rows)


def _group_playbooks(playbooks: list[PlaybookRecord]) -> dict[str, list[PlaybookRecord]]:
    grouped: dict[str, list[PlaybookRecord]] = {}
    for playbook in playbooks:
        key = f"{playbook.symbol}|{playbook.bias_template}|{playbook.horizon}"
        grouped.setdefault(key, []).append(playbook)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=_playbook_ranking_key, reverse=True)
    return grouped


def _select_playbook_for_bias(
    bias: BiasInputRow,
    grouped_playbooks: dict[str, list[PlaybookRecord]],
) -> tuple[TranslatorSelection, PlaybookRecord | None]:
    if not bias.enabled:
        selection = TranslatorSelection(
            symbol=bias.symbol,
            bias_template=bias.bias_template,
            status="disabled",
            reason="bias_row_disabled",
            translator_notes="row disabled; no playbook armed",
        )
        return selection, None

    candidates = grouped_playbooks.get(bias.routing_key, [])
    active_candidates = [candidate for candidate in candidates if candidate.lifecycle_status == PLAYBOOK_STATUS_ACTIVE]
    if not active_candidates:
        selection = TranslatorSelection(
            symbol=bias.symbol,
            bias_template=bias.bias_template,
            status="no_match",
            reason="no_active_playbook_for_context",
            translator_notes=f"no active playbook for {bias.routing_key}",
        )
        return selection, None

    compatible = [
        candidate
        for candidate in active_candidates
        if candidate.is_full_m1_m5_survivor and candidate.bhiksha_compatibility.get("supported", False)
    ]
    if not compatible:
        selection = TranslatorSelection(
            symbol=bias.symbol,
            bias_template=bias.bias_template,
            status="unsupported",
            reason="matching_playbooks_not_bhiksha_ready",
            translator_notes=f"matching playbooks exist for {bias.routing_key} but none are Bhiksha-ready",
        )
        return selection, None

    selected = max(compatible, key=_playbook_ranking_key)
    score = _playbook_ranking_score(selected)
    selection = TranslatorSelection(
        symbol=bias.symbol,
        bias_template=bias.bias_template,
        status="armed",
        selected_playbook_id=selected.playbook_id,
        reason="selected_top_ranked_active_playbook",
        translator_notes=(
            f"selected {selected.strategy_key} "
            f"(expectancy={_format_metric(selected.expectancy)}, "
            f"confidence={_format_metric(selected.confidence)}, "
            f"robustness={_format_metric(selected.execution_robustness)})"
        ),
        ranking_score=score,
    )
    return selection, selected


def _bias_row_output(bias: BiasInputRow, selection: TranslatorSelection) -> dict[str, Any]:
    return {
        "Date": bias.date.isoformat(),
        "Symbol": bias.symbol,
        "Daily_Bias": bias.daily_bias,
        "Intraday_Thesis": bias.intraday_thesis,
        "Max_Risk_USD": bias.max_risk_usd,
        "Enabled": bias.enabled,
        "After_Time_ET": bias.after_time_et or "",
        "Only_If_Price_Crosses": bias.only_if_price_crosses or "",
        "Translator_Status": selection.status,
        "Armed_Playbook_ID": selection.selected_playbook_id or "",
        "Translator_Notes": selection.translator_notes,
        "Notes": bias.notes,
    }


def _playbook_ranking_key(playbook: PlaybookRecord) -> tuple[float, float, float, float, int, int]:
    return (
        _playbook_ranking_score(playbook),
        playbook.expectancy or float("-inf"),
        playbook.execution_robustness or float("-inf"),
        playbook.confidence or float("-inf"),
        playbook.signal_count or 0,
        _PLAYBOOK_STATUS_PRIORITY[playbook.lifecycle_status],
    )


def _playbook_ranking_score(playbook: PlaybookRecord) -> float:
    freshness_penalty = 0.0
    last_validated = _parse_iso_date(playbook.last_validated_date)
    if last_validated is not None:
        age_days = max((datetime.now(UTC).date() - last_validated).days, 0)
        freshness_penalty = min(age_days, 365) * 0.05
    return round(
        (1000.0 if playbook.is_full_m1_m5_survivor else 0.0)
        + (_PLAYBOOK_STATUS_PRIORITY[playbook.lifecycle_status] * 50.0)
        + ((playbook.execution_robustness or 0.0) * 100.0)
        + ((playbook.expectancy or 0.0) * 50.0)
        + ((playbook.confidence or 0.0) * 25.0)
        + min(playbook.signal_count or 0, 500) * 0.1
        - freshness_penalty,
        6,
    )


def _playbook_sort_key(playbook: PlaybookRecord) -> tuple[int, float, float]:
    return (
        _PLAYBOOK_STATUS_PRIORITY[playbook.lifecycle_status],
        playbook.expectancy or float("-inf"),
        playbook.execution_robustness or float("-inf"),
    )


def _build_playbook_id(
    *,
    strategy_key: str,
    symbol: str,
    direction: str,
    entry_params: dict[str, Any],
    execution_mapping: dict[str, Any],
    exit_mapping: dict[str, Any],
) -> str:
    digest = stable_signature(
        {
            "strategy_key": strategy_key,
            "symbol_scope": [symbol.upper()],
            "direction": direction,
            "entry": entry_params,
            "execution": execution_mapping,
            "exit": exit_mapping,
        },
        length=12,
    )
    return f"{strategy_key}_{symbol.lower()}_{direction}_{digest}"


def _deployment_id_from_playbook(*, playbook: PlaybookRecord, bias: BiasInputRow) -> str:
    suffix = stable_signature(
        {
            "playbook_id": playbook.playbook_id,
            "symbol": bias.symbol,
            "date": bias.date.isoformat(),
            "bias_template": bias.bias_template,
        },
        length=8,
    )
    return f"{playbook.strategy_key}_{bias.symbol.lower()}_{playbook.direction}_armed_{suffix}"


def _regime_tags(bias_template: str, strategy_key: str) -> list[str]:
    direction, thesis, horizon = bias_template.split("_", 2)
    return [direction, thesis, horizon, strategy_key]


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _manifest_window(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    execution = manifest.get("execution", {})
    start = execution.get("entry_window_start_et")
    end = execution.get("entry_window_end_et")
    if start or end:
        return _optional_text(start), _optional_text(end)
    return _parse_window(manifest.get("entry_window_et"))


def _lifecycle_status(row: dict[str, Any]) -> Literal["active", "stale", "retired"]:
    queue_status = str(row.get("queue_status", "")).strip()
    if queue_status == QUEUE_STATUS_KILLED:
        return PLAYBOOK_STATUS_RETIRED
    if queue_status == QUEUE_STATUS_STALE:
        return PLAYBOOK_STATUS_STALE
    return PLAYBOOK_STATUS_ACTIVE


def _queue_strategy_params(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("config_json")
    if raw not in (None, ""):
        try:
            loaded = json.loads(str(raw))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return {str(key): value for key, value in loaded.items()}
    params: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith(("m1_", "m2_")):
            continue
        if key in {
            "candidate_key",
            "strategy_family",
            "ticker",
            "strategy",
            "direction",
            "research_slice_id",
            "config_signature",
            "config_json",
            "chart_link",
            "passes_m1",
            "passes_m2",
            "passes_m3",
            "passes_m4",
            "passes_m5",
            "is_full_m1_m5_survivor",
            "latest_stage_reached",
            "latest_stage_decision",
            "last_seen_run_date",
            "last_source_run_dir",
            "human_decision",
            "human_notes",
            "priority",
            "human_updated_at",
            "queue_status",
            "last_action_run_date",
            "latest_artifact_dir",
        }:
            continue
        if value in ("", None):
            continue
        params[str(key)] = _typed_value(value)
    return params


def _typed_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    raw = str(value).strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _read_optional_csv(root: Path, *names: str) -> pl.DataFrame:
    for name in names:
        path = root / name
        if not path.exists():
            continue
        try:
            return pl.read_csv(path)
        except NoDataError:
            return pl.DataFrame()
    return pl.DataFrame()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _append_notes(existing: str, note: str) -> str:
    return note if not existing else f"{existing}; {note}"


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(json_ready(value), sort_keys=True)
    return _jsonable(value)


__all__ = [
    "BiasInputRow",
    "LiveObservationRecord",
    "PLAYBOOK_STATUS_ACTIVE",
    "PLAYBOOK_STATUS_RETIRED",
    "PLAYBOOK_STATUS_STALE",
    "PlaybookRecord",
    "TranslatorSelection",
    "augment_playbook_catalog_from_queue",
    "build_armed_deployment_manifest",
    "build_playbook_records_from_queue",
    "load_bias_inputs_sheet",
    "load_live_observation_records",
    "load_playbook_records",
    "route_bias_rows",
    "route_google_sheet_bias_inputs",
    "route_bias_inputs",
    "write_live_observation_records",
]
