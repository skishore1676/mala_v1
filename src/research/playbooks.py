"""Playbook catalog, bias routing, and armed deployment contracts for Mala."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any, Literal

import polars as pl
from polars.exceptions import NoDataError
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

from src.config import PROJECT_ROOT, settings
from src.research.exit_optimizer import load_exit_optimization_result
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
MASTER_PLAYBOOK_STALE_DAYS = 60
MASTER_PLAYBOOK_CATALOG_CONTRACT_NAME = "master_playbook_catalog"
MASTER_PLAYBOOK_OVERRIDE_RETIRED = "retired"

_MASTER_PLAYBOOK_SHEET_HEADERS = [
    "catalog_key",
    "playbook_id",
    "symbol",
    "bias_template",
    "strategy_key",
    "strategy_family",
    "direction",
    "lifecycle_status",
    "operator_status_override",
    "operator_notes",
    "bionic_ready",
    "first_validated_date",
    "last_validated_date",
    "validation_count",
    "expectancy",
    "confidence",
    "signal_count",
    "execution_robustness",
    "thesis_exit_policy",
    "playbook_summary_json",
]

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
    catalog_key: str | None = None
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
    operator_status_override: str = ""
    operator_notes: str = ""
    automation_status: str
    surface_class: str
    entry_params: dict[str, Any] = Field(default_factory=dict)
    exit_params: dict[str, Any] = Field(default_factory=dict)
    thesis_exit_anchor: str | None = None
    thesis_exit_policy: str | None = None
    thesis_exit_params: dict[str, Any] = Field(default_factory=dict)
    catastrophe_exit_anchor: str | None = None
    catastrophe_exit_params: dict[str, Any] = Field(default_factory=dict)
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
    first_validated_date: str | None = None
    last_validated_date: str
    validation_count: int = 1
    research_slice_id: str | None = None
    bionic_ready: bool = False
    source: dict[str, Any] = Field(default_factory=dict)
    bhiksha_compatibility: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: Any) -> str:
        return str(value).upper()

    @model_validator(mode="after")
    def populate_catalog_defaults(self) -> "PlaybookRecord":
        if not self.catalog_key:
            self.catalog_key = _build_catalog_key(
                strategy_key=self.strategy_key,
                symbol=self.symbol,
                direction=self.direction,
                entry_params=self.entry_params,
                thesis_exit_policy=self.thesis_exit_policy,
                thesis_exit_params=self.thesis_exit_params,
                catastrophe_exit_params=self.catastrophe_exit_params,
                execution_mapping=self.execution_mapping,
            )
        if not self.first_validated_date:
            self.first_validated_date = self.last_validated_date
        self.operator_status_override = str(self.operator_status_override or "").strip().lower()
        self.operator_notes = str(self.operator_notes or "").strip()
        return self


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


def default_master_playbook_catalog_path() -> Path:
    configured = Path(settings.master_playbook_catalog_path)
    if configured.is_absolute():
        return configured
    return (PROJECT_ROOT / configured).resolve()


def default_master_playbook_projection_path() -> Path:
    configured = Path(settings.master_playbook_projection_path)
    if configured.is_absolute():
        return configured
    return (PROJECT_ROOT / configured).resolve()


def merge_master_playbook_catalog(
    *,
    new_playbooks: list[PlaybookRecord],
    catalog_path: str | Path | None = None,
    projection_path: str | Path | None = None,
    sync_sheet: bool = True,
) -> tuple[Path, Path | None, list[PlaybookRecord]]:
    resolved_catalog = Path(catalog_path) if catalog_path is not None else default_master_playbook_catalog_path()
    resolved_projection = (
        Path(projection_path) if projection_path is not None else default_master_playbook_projection_path()
    )
    existing = load_master_playbook_records(resolved_catalog)
    sheet_overrides = load_master_playbook_sheet_overrides()
    merged = _merge_playbook_records(existing=existing, incoming=new_playbooks, sheet_overrides=sheet_overrides)
    _write_master_playbook_catalog(
        resolved_catalog,
        merged,
        projection_path=resolved_projection,
    )
    if sync_sheet:
        sync_master_playbook_catalog_sheet(playbooks=merged)
    return resolved_catalog, resolved_projection, merged


def refresh_master_playbook_catalog_statuses(
    *,
    catalog_path: str | Path | None = None,
    projection_path: str | Path | None = None,
    sync_sheet: bool = True,
) -> tuple[Path, Path | None, list[PlaybookRecord]]:
    resolved_catalog = Path(catalog_path) if catalog_path is not None else default_master_playbook_catalog_path()
    resolved_projection = (
        Path(projection_path) if projection_path is not None else default_master_playbook_projection_path()
    )
    existing = load_master_playbook_records(resolved_catalog)
    sheet_overrides = load_master_playbook_sheet_overrides()
    refreshed = _merge_playbook_records(existing=existing, incoming=[], sheet_overrides=sheet_overrides)
    _write_master_playbook_catalog(resolved_catalog, refreshed, projection_path=resolved_projection)
    if sync_sheet:
        sync_master_playbook_catalog_sheet(playbooks=refreshed)
    return resolved_catalog, resolved_projection, refreshed


def seed_master_playbook_catalog_from_queue(
    *,
    queue_path: str | Path,
    catalog_path: str | Path | None = None,
    projection_path: str | Path | None = None,
    sync_sheet: bool = True,
    project_root: Path | None = None,
) -> tuple[Path, Path | None, list[PlaybookRecord]]:
    rows = _load_csv_rows(Path(queue_path))
    playbooks = build_playbook_records_from_queue(rows, project_root=project_root)
    return merge_master_playbook_catalog(
        new_playbooks=playbooks,
        catalog_path=catalog_path,
        projection_path=projection_path,
        sync_sheet=sync_sheet,
    )


def seed_master_playbook_catalog_from_catalog(
    *,
    source_catalog_path: str | Path,
    catalog_path: str | Path | None = None,
    projection_path: str | Path | None = None,
    sync_sheet: bool = True,
) -> tuple[Path, Path | None, list[PlaybookRecord]]:
    playbooks = load_playbook_records(source_catalog_path)
    return merge_master_playbook_catalog(
        new_playbooks=playbooks,
        catalog_path=catalog_path,
        projection_path=projection_path,
        sync_sheet=sync_sheet,
    )


def load_master_playbook_records(path: str | Path | None = None) -> list[PlaybookRecord]:
    resolved = Path(path) if path is not None else default_master_playbook_catalog_path()
    if not resolved.exists():
        return []
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return [PlaybookRecord.model_validate(item) for item in payload.get("playbooks", [])]


def load_master_playbook_sheet_overrides(
    *,
    spreadsheet_id: str | None = None,
    sheet_name: str | None = None,
    credentials_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    resolved_sheet_id = (spreadsheet_id or settings.master_playbook_sheet_id or settings.bionic_sheet_id).strip()
    if not resolved_sheet_id:
        return {}
    resolved_sheet_name = (sheet_name or settings.master_playbook_sheet_name).strip()
    resolved_credentials = str(credentials_path or settings.google_api_credentials_path).strip()
    if not resolved_credentials:
        return {}
    try:
        client = GoogleSheetTableClient(
            spreadsheet_id=resolved_sheet_id,
            sheet_name=resolved_sheet_name,
            credentials_path=Path(resolved_credentials),
        )
        rows = client.read_rows()
    except Exception:
        return {}
    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        catalog_key = str(row.get("catalog_key", "")).strip()
        playbook_id = str(row.get("playbook_id", "")).strip()
        key = catalog_key or playbook_id
        if not key:
            continue
        overrides[key] = {
            "operator_status_override": str(row.get("operator_status_override", "")).strip().lower(),
            "operator_notes": str(row.get("operator_notes", "")).strip(),
        }
    return overrides


def sync_master_playbook_catalog_sheet(
    *,
    playbooks: list[PlaybookRecord],
    spreadsheet_id: str | None = None,
    sheet_name: str | None = None,
    credentials_path: str | Path | None = None,
) -> None:
    resolved_sheet_id = (spreadsheet_id or settings.master_playbook_sheet_id or settings.bionic_sheet_id).strip()
    if not resolved_sheet_id:
        return
    resolved_sheet_name = (sheet_name or settings.master_playbook_sheet_name).strip()
    resolved_credentials = str(credentials_path or settings.google_api_credentials_path).strip()
    if not resolved_credentials:
        return
    client = GoogleSheetTableClient(
        spreadsheet_id=resolved_sheet_id,
        sheet_name=resolved_sheet_name,
        credentials_path=Path(resolved_credentials),
    )
    rows = [_master_playbook_sheet_row(playbook) for playbook in playbooks]
    client.overwrite_table(headers=_MASTER_PLAYBOOK_SHEET_HEADERS, rows=rows)


def resolve_playbook_catalog_path(playbook_catalog_path: str | Path | None = None) -> Path:
    if playbook_catalog_path is not None:
        return Path(playbook_catalog_path)
    master = default_master_playbook_catalog_path()
    if master.exists():
        return master
    raise FileNotFoundError(
        "No playbook catalog path provided and master_playbook_catalog.json does not exist yet."
    )


def route_bias_inputs(
    *,
    bias_inputs_path: str | Path,
    playbook_catalog_path: str | Path | None = None,
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
    playbook_catalog_path: str | Path | None = None,
    out_dir: str | Path,
    bias_source: str = "in_memory",
) -> tuple[Path, Path, list[TranslatorSelection], list[dict[str, Any]]]:
    resolved_catalog = resolve_playbook_catalog_path(playbook_catalog_path)
    playbooks = load_playbook_records(resolved_catalog)
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
                "playbook_catalog_path": str(resolved_catalog.resolve()),
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
    playbook_catalog_path: str | Path | None = None,
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
            "bionic_ready": playbook.bionic_ready,
            "thesis_exit_policy": playbook.thesis_exit_policy,
            "thesis_exit_anchor": playbook.thesis_exit_anchor,
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


def load_playbook_records(path: str | Path | None = None) -> list[PlaybookRecord]:
    resolved = resolve_playbook_catalog_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
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
    exit_optimization_path = artifact_dir / "m5_exit_optimization.json"
    exit_optimization = (
        load_exit_optimization_result(exit_optimization_path)
        if exit_optimization_path.exists()
        else None
    )
    if exit_optimization is not None:
        manifest.setdefault("exit", {})
        manifest["exit"].update(
            {
                "use_algorithmic_exit": False,
                "thesis_exit_anchor": exit_optimization.thesis_exit_anchor,
                "thesis_exit_policy": exit_optimization.thesis_exit_policy,
                "thesis_exit_params": json_ready(exit_optimization.thesis_exit_params),
                "catastrophe_exit_anchor": exit_optimization.catastrophe_exit_anchor,
                "catastrophe_exit_params": json_ready(exit_optimization.catastrophe_exit_params),
                "stop_loss_pct": exit_optimization.catastrophe_exit_params.get(
                    "stop_loss_pct",
                    manifest["exit"].get("stop_loss_pct"),
                ),
                "hard_flat_time_et": exit_optimization.catastrophe_exit_params.get(
                    "hard_flat_time_et",
                    manifest["exit"].get("hard_flat_time_et"),
                ),
            }
        )
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
    bias_template = _bias_template(
        strategy_key,
        str(row["direction"]),
        strategy_params=strategy_params,
    )
    confidence = _maybe_float(row.get("m2_mean_test_confidence")) or _maybe_float(row.get("m1_avg_test_confidence"))
    signal_count = _maybe_int(m5_row.get("holdout_trades")) or _maybe_int(row.get("m1_oos_signals"))
    expectancy = _maybe_float(m5_row.get("base_exp_r")) or _maybe_float(m4_row.get("mean_holdout_exp_r"))
    execution_robustness = _maybe_float(m5_row.get("mc_prob_positive_exp"))
    bhiksha_supported = surface_class == "supported"
    has_optimized_exit = exit_optimization is not None
    bionic_ready = bhiksha_supported and has_optimized_exit
    validated_date = str(row.get("last_action_run_date") or row.get("last_seen_run_date") or "")
    catalog_key = _build_catalog_key(
        strategy_key=strategy_key,
        symbol=str(row["ticker"]),
        direction=str(row["direction"]),
        entry_params={**strategy_params, "direction": str(row["direction"])},
        thesis_exit_policy=exit_optimization.thesis_exit_policy if exit_optimization is not None else None,
        thesis_exit_params=(
            json_ready(exit_optimization.thesis_exit_params) if exit_optimization is not None else {}
        ),
        catastrophe_exit_params=(
            json_ready(exit_optimization.catastrophe_exit_params)
            if exit_optimization is not None
            else {
                "stop_loss_pct": manifest["exit"].get("stop_loss_pct"),
                "hard_flat_time_et": manifest["exit"].get("hard_flat_time_et"),
            }
        ),
        execution_mapping=json_ready(manifest["execution"]),
    )
    record = PlaybookRecord(
        playbook_id=playbook_id,
        catalog_key=catalog_key,
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
        thesis_exit_anchor=exit_optimization.thesis_exit_anchor if exit_optimization is not None else None,
        thesis_exit_policy=exit_optimization.thesis_exit_policy if exit_optimization is not None else None,
        thesis_exit_params=(
            json_ready(exit_optimization.thesis_exit_params) if exit_optimization is not None else {}
        ),
        catastrophe_exit_anchor=(
            exit_optimization.catastrophe_exit_anchor if exit_optimization is not None else "option_premium"
        ),
        catastrophe_exit_params=(
            json_ready(exit_optimization.catastrophe_exit_params)
            if exit_optimization is not None
            else {
                "stop_loss_pct": manifest["exit"].get("stop_loss_pct"),
                "hard_flat_time_et": manifest["exit"].get("hard_flat_time_et"),
            }
        ),
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
        first_validated_date=validated_date,
        last_validated_date=validated_date,
        validation_count=1,
        research_slice_id=_optional_text(row.get("research_slice_id")),
        bionic_ready=bionic_ready,
        source={
            "queue_candidate_key": row.get("candidate_key"),
            "artifact_dir": str(artifact_dir),
            "chart_link": row.get("chart_link", ""),
            "config_signature": row.get("config_signature", ""),
            "m4_summary": _jsonable(m4_row),
            "m5_execution": _jsonable(m5_row),
            "exit_optimization": (
                exit_optimization.model_dump(mode="json") if exit_optimization is not None else None
            ),
        },
        bhiksha_compatibility={
            "supported": bhiksha_supported,
            "has_optimized_thesis_exit": has_optimized_exit,
            "bionic_ready": bionic_ready,
            "required_capabilities": (
                []
                if bionic_ready
                else (
                    ["optimized_underlying_thesis_exit_missing"]
                    if bhiksha_supported
                    else ["strategy_plugin_not_yet_implemented_in_bhiksha"]
                )
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
            "catalog_key": playbook.catalog_key,
            "playbook_id": playbook.playbook_id,
            "symbol": playbook.symbol,
            "bias_template": playbook.bias_template,
            "strategy_key": playbook.strategy_key,
            "direction": playbook.direction,
            "lifecycle_status": playbook.lifecycle_status,
            "operator_status_override": playbook.operator_status_override,
            "operator_notes": playbook.operator_notes,
            "automation_status": playbook.automation_status,
            "bionic_ready": playbook.bionic_ready,
            "expectancy": playbook.expectancy,
            "confidence": playbook.confidence,
            "signal_count": playbook.signal_count,
            "execution_robustness": playbook.execution_robustness,
            "first_validated_date": playbook.first_validated_date,
            "last_validated_date": playbook.last_validated_date,
            "validation_count": playbook.validation_count,
            "is_full_m1_m5_survivor": playbook.is_full_m1_m5_survivor,
        }
        for playbook in playbooks
    ]
    _write_csv_rows(path, rows)


def _write_master_playbook_catalog(
    path: Path,
    playbooks: list[PlaybookRecord],
    *,
    projection_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **build_contract_metadata(MASTER_PLAYBOOK_CATALOG_CONTRACT_NAME),
        "generated_at": datetime.now(UTC).isoformat(),
        "playbook_count": len(playbooks),
        "playbook_status_counts": {
            status: sum(1 for playbook in playbooks if playbook.lifecycle_status == status)
            for status in (PLAYBOOK_STATUS_ACTIVE, PLAYBOOK_STATUS_STALE, PLAYBOOK_STATUS_RETIRED)
        },
        "playbooks": [playbook.model_dump(mode="json") for playbook in playbooks],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if projection_path is not None:
        _write_playbook_projection_csv(projection_path, playbooks)


def _merge_playbook_records(
    *,
    existing: list[PlaybookRecord],
    incoming: list[PlaybookRecord],
    sheet_overrides: dict[str, dict[str, str]],
) -> list[PlaybookRecord]:
    by_key: dict[str, PlaybookRecord] = {}
    for playbook in existing:
        by_key[playbook.catalog_key or playbook.playbook_id] = playbook
    for playbook in incoming:
        key = playbook.catalog_key or playbook.playbook_id
        current = by_key.get(key)
        if current is None:
            updated = playbook.model_copy(
                update={
                    "first_validated_date": playbook.first_validated_date or playbook.last_validated_date,
                    "validation_count": max(playbook.validation_count, 1),
                }
            )
        else:
            updated = playbook.model_copy(
                update={
                    "first_validated_date": current.first_validated_date or playbook.first_validated_date or playbook.last_validated_date,
                    "validation_count": max(current.validation_count, 1) + 1,
                    "operator_status_override": current.operator_status_override,
                    "operator_notes": current.operator_notes,
                }
            )
        by_key[key] = updated

    merged: list[PlaybookRecord] = []
    for key, playbook in by_key.items():
        override = sheet_overrides.get(key) or sheet_overrides.get(playbook.playbook_id) or {}
        effective = playbook.model_copy(
            update={
                "operator_status_override": override.get("operator_status_override", playbook.operator_status_override),
                "operator_notes": override.get("operator_notes", playbook.operator_notes),
            }
        )
        effective = effective.model_copy(update={"lifecycle_status": _effective_lifecycle_status(effective)})
        merged.append(effective)
    return sorted(
        merged,
        key=lambda record: (
            record.symbol,
            record.bias_template,
            -_PLAYBOOK_STATUS_PRIORITY[record.lifecycle_status],
            -(record.expectancy or float("-inf")),
            record.playbook_id,
        ),
    )


def _effective_lifecycle_status(playbook: PlaybookRecord) -> Literal["active", "stale", "retired"]:
    override = (playbook.operator_status_override or "").strip().lower()
    if override == MASTER_PLAYBOOK_OVERRIDE_RETIRED:
        return PLAYBOOK_STATUS_RETIRED
    last_validated = _parse_iso_date(playbook.last_validated_date)
    if last_validated is not None and (datetime.now(UTC).date() - last_validated) > timedelta(days=MASTER_PLAYBOOK_STALE_DAYS):
        return PLAYBOOK_STATUS_STALE
    return PLAYBOOK_STATUS_ACTIVE


def _master_playbook_sheet_row(playbook: PlaybookRecord) -> dict[str, Any]:
    return {
        "catalog_key": playbook.catalog_key or "",
        "playbook_id": playbook.playbook_id,
        "symbol": playbook.symbol,
        "bias_template": playbook.bias_template,
        "strategy_key": playbook.strategy_key,
        "strategy_family": playbook.strategy_family,
        "direction": playbook.direction,
        "lifecycle_status": playbook.lifecycle_status,
        "operator_status_override": playbook.operator_status_override,
        "operator_notes": playbook.operator_notes,
        "bionic_ready": playbook.bionic_ready,
        "first_validated_date": playbook.first_validated_date or "",
        "last_validated_date": playbook.last_validated_date,
        "validation_count": playbook.validation_count,
        "expectancy": playbook.expectancy,
        "confidence": playbook.confidence,
        "signal_count": playbook.signal_count,
        "execution_robustness": playbook.execution_robustness,
        "thesis_exit_policy": playbook.thesis_exit_policy or "",
        "playbook_summary_json": json.dumps(
            {
                "entry_params": playbook.entry_params,
                "thesis_exit_params": playbook.thesis_exit_params,
                "catastrophe_exit_params": playbook.catastrophe_exit_params,
                "vehicle_mapping": playbook.vehicle_mapping,
                "bhiksha_compatibility": playbook.bhiksha_compatibility,
            },
            sort_keys=True,
        ),
    }


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
        if candidate.is_full_m1_m5_survivor and candidate.bhiksha_compatibility.get("bionic_ready", False)
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


def _build_catalog_key(
    *,
    strategy_key: str,
    symbol: str,
    direction: str,
    entry_params: dict[str, Any],
    thesis_exit_policy: str | None,
    thesis_exit_params: dict[str, Any],
    catastrophe_exit_params: dict[str, Any],
    execution_mapping: dict[str, Any],
) -> str:
    digest = stable_signature(
        {
            "strategy_key": strategy_key,
            "symbol": symbol.upper(),
            "direction": direction,
            "entry": entry_params,
            "thesis_exit_policy": thesis_exit_policy,
            "thesis_exit_params": thesis_exit_params,
            "catastrophe_exit_params": catastrophe_exit_params,
            "execution_profile": {
                "profile": execution_mapping.get("profile"),
                "option_mapping": execution_mapping.get("option_mapping"),
                "dte_min": execution_mapping.get("dte_min"),
                "dte_max": execution_mapping.get("dte_max"),
            },
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
    "MASTER_PLAYBOOK_CATALOG_CONTRACT_NAME",
    "MASTER_PLAYBOOK_STALE_DAYS",
    "PLAYBOOK_STATUS_ACTIVE",
    "PLAYBOOK_STATUS_RETIRED",
    "PLAYBOOK_STATUS_STALE",
    "PlaybookRecord",
    "TranslatorSelection",
    "augment_playbook_catalog_from_queue",
    "build_armed_deployment_manifest",
    "build_playbook_records_from_queue",
    "default_master_playbook_catalog_path",
    "default_master_playbook_projection_path",
    "load_bias_inputs_sheet",
    "load_live_observation_records",
    "load_master_playbook_records",
    "load_playbook_records",
    "merge_master_playbook_catalog",
    "refresh_master_playbook_catalog_statuses",
    "resolve_playbook_catalog_path",
    "route_bias_rows",
    "route_google_sheet_bias_inputs",
    "route_bias_inputs",
    "seed_master_playbook_catalog_from_queue",
    "seed_master_playbook_catalog_from_catalog",
    "sync_master_playbook_catalog_sheet",
    "write_live_observation_records",
]
