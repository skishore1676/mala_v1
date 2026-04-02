"""Compile one unified Bhiksha session payload from playbooks and manual entries."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.research.google_sheets import GoogleSheetTableClient, spreadsheet_id_from_url
from src.research.playbooks import BiasInputRow, _bias_row_from_sheet_row, route_bias_rows


ACTIVE_SESSION_CONTRACT_NAME = "active_session"
ACTIVE_SESSION_SCHEMA_VERSION = 1

_PENDING_STATUSES = {"", "PENDING", "BLOCKED", "PROCESSING"}
_TERMINAL_STATUSES = {"TRIGGERED", "EXECUTED", "DONE", "CLOSED", "CANCELLED", "ERROR"}
_TRIGGER_DIRECTIONS = {"ABOVE", "BELOW", "CLOSE_BY"}


class ManualEntryRow(BaseModel):
    row_index: int | None = None
    stock: str
    trigger_price: float
    trigger_direction: str = "ABOVE"
    direction: str
    expected_move: str = "2%"
    option_week_to_play: str = "1"
    status: str = ""
    is_signal_active: str = "1"
    trade_id: str = ""
    notes: str = ""
    idea_date: date | None = None
    execute_after: str | None = None
    day: str = ""
    max_risk_usd: float | None = None
    stop_loss_pct: float | None = None
    profit_target_multiple: float | None = None
    hard_flat_time_et: str | None = None

    @field_validator("stock", mode="before")
    @classmethod
    def normalize_stock(cls, value: Any) -> str:
        return str(value).strip().upper()

    @field_validator("trigger_direction", mode="before")
    @classmethod
    def normalize_trigger_direction(cls, value: Any) -> str:
        normalized = str(value or "ABOVE").strip().upper()
        if normalized not in _TRIGGER_DIRECTIONS:
            raise ValueError(f"Unsupported trigger_direction: {value!r}")
        return normalized

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"c", "call", "long"}:
            return "long"
        if normalized in {"p", "put", "short"}:
            return "short"
        raise ValueError(f"Unsupported manual direction: {value!r}")

    @field_validator("idea_date", mode="before")
    @classmethod
    def parse_idea_date(cls, value: Any) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date):
            return value
        raw = str(value).strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unsupported idea_date format: {value!r}")

    @model_validator(mode="after")
    def normalize_optional_values(self) -> "ManualEntryRow":
        self.status = str(self.status or "").strip().upper()
        self.trade_id = str(self.trade_id or "").strip()
        self.notes = str(self.notes or "").strip()
        self.execute_after = _optional_text(self.execute_after)
        self.day = str(self.day or "").strip()
        return self

    @property
    def symbol(self) -> str:
        return self.stock

    @property
    def active_mode(self) -> str:
        return str(self.is_signal_active or "").strip() or "1"

    def is_session_eligible(self, *, today: date) -> bool:
        if self.status in _TERMINAL_STATUSES:
            return False
        if self.status not in _PENDING_STATUSES:
            return False
        if self.active_mode in {"", "0"}:
            return False
        if self.active_mode == "2":
            return self.idea_date is None or today == self.idea_date
        if self.idea_date is not None and today < self.idea_date:
            return False
        return True


def compile_active_session_from_rows(
    *,
    biases: list[BiasInputRow],
    manual_entries: list[ManualEntryRow],
    playbook_catalog_path: str | Path,
    out_dir: str | Path,
    bias_source: str = "in_memory",
    manual_source: str = "in_memory",
    session_date: date | None = None,
    live_authorized: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    routing_dir = target_dir / "bias_routing"
    routing_report_path, armed_payloads_path, selections, _ = route_bias_rows(
        biases=biases,
        playbook_catalog_path=playbook_catalog_path,
        out_dir=routing_dir,
        bias_source=bias_source,
    )
    armed_payloads = json.loads(armed_payloads_path.read_text(encoding="utf-8"))

    effective_date = session_date or datetime.now(UTC).date()
    eligible_manual_entries = [entry for entry in manual_entries if entry.is_session_eligible(today=effective_date)]
    manual_by_symbol: dict[str, ManualEntryRow] = {}
    suppressed: list[dict[str, Any]] = []
    for entry in eligible_manual_entries:
        previous = manual_by_symbol.get(entry.symbol)
        if previous is not None:
            suppressed.append(
                {
                    "symbol": entry.symbol,
                    "suppressed_origin": "operator_manual",
                    "reason": "replaced_by_later_manual_row",
                    "previous_row_index": previous.row_index,
                    "row_index": entry.row_index,
                }
            )
        manual_by_symbol[entry.symbol] = entry

    deployments: list[dict[str, Any]] = []
    for entry in manual_by_symbol.values():
        deployments.append(_apply_session_authorization(build_manual_trigger_manifest(entry), live_authorized=live_authorized))

    for item in armed_payloads.get("armed_playbooks", []):
        manifest = dict(item["manifest"])
        symbol = str(manifest["symbol"]).upper()
        if symbol in manual_by_symbol:
            suppressed.append(
                {
                    "symbol": symbol,
                    "suppressed_origin": "mala_playbook",
                    "reason": "manual_override",
                    "playbook_id": item.get("playbook_id"),
                    "deployment_id": manifest.get("deployment_id"),
                }
            )
            continue
        manifest.setdefault("source", {})
        manifest["source"]["origin"] = "mala_playbook"
        deployments.append(_apply_session_authorization(manifest, live_authorized=live_authorized))

    payload = {
        "contract_name": ACTIVE_SESSION_CONTRACT_NAME,
        "schema_version": ACTIVE_SESSION_SCHEMA_VERSION,
        "session_id": f"active_session_{effective_date.isoformat()}",
        "session_date": effective_date.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "authorization_mode": "live" if live_authorized else "shadow",
        "source": {
            "bionic": bias_source,
            "manual": manual_source,
            "playbook_catalog_path": str(Path(playbook_catalog_path).resolve()),
            "routing_report_path": str(routing_report_path.resolve()),
        },
        "summary": {
            "deployment_count": len(deployments),
            "manual_deployment_count": sum(1 for deployment in deployments if deployment.get("source", {}).get("origin") == "operator_manual"),
            "playbook_deployment_count": sum(1 for deployment in deployments if deployment.get("source", {}).get("origin") == "mala_playbook"),
            "armed_bias_count": sum(1 for selection in selections if selection.status == "armed"),
            "eligible_manual_row_count": len(eligible_manual_entries),
            "suppressed_count": len(suppressed),
            "live_authorized_deployment_count": sum(
                1 for deployment in deployments if deployment.get("execution", {}).get("shadow_only") is False
            ),
        },
        "suppressed": suppressed,
        "deployments": deployments,
    }
    session_path = target_dir / "active_session.json"
    session_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "session_path": str(session_path.resolve()),
        "bias_routing_report_path": str(routing_report_path.resolve()),
        "manual_source": manual_source,
        "bionic_source": bias_source,
        "authorization_mode": payload["authorization_mode"],
        "selections": [selection.model_dump(mode="json") for selection in selections],
        "eligible_manual_entries": [entry.model_dump(mode="json") for entry in eligible_manual_entries],
        "suppressed": suppressed,
    }
    report_path = target_dir / "active_session_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return session_path, report_path, payload


def compile_active_session_from_google_sheets(
    *,
    bionic_spreadsheet_id: str,
    bionic_sheet_name: str,
    manual_spreadsheet_id: str,
    manual_sheet_name: str,
    credentials_path: str | Path,
    playbook_catalog_path: str | Path,
    out_dir: str | Path,
    update_bionic_sheet: bool = True,
    live_authorized: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    credentials = Path(credentials_path)
    bionic_client = GoogleSheetTableClient(
        spreadsheet_id=bionic_spreadsheet_id,
        sheet_name=bionic_sheet_name,
        credentials_path=credentials,
    )
    manual_client = GoogleSheetTableClient(
        spreadsheet_id=manual_spreadsheet_id,
        sheet_name=manual_sheet_name,
        credentials_path=credentials,
    )

    raw_bias_rows = bionic_client.read_rows()
    biases = [_bias_row_from_sheet_row(row) for row in raw_bias_rows]
    raw_manual_rows = manual_client.read_rows()
    manual_entries = _parse_manual_rows(raw_manual_rows)

    session_path, report_path, payload = compile_active_session_from_rows(
        biases=biases,
        manual_entries=manual_entries,
        playbook_catalog_path=playbook_catalog_path,
        out_dir=out_dir,
        bias_source=f"google_sheet:{spreadsheet_id_from_url(bionic_spreadsheet_id)}:{bionic_sheet_name}",
        manual_source=f"google_sheet:{spreadsheet_id_from_url(manual_spreadsheet_id)}:{manual_sheet_name}",
        live_authorized=live_authorized,
    )

    if update_bionic_sheet:
        # Re-read the routing CSV so we can preserve the exact writeback behavior.
        routing_rows = _load_csv_rows(Path(out_dir) / "bias_routing" / "bias_routing_report.csv")
        updates = []
        for raw_row, updated_row in zip(raw_bias_rows, routing_rows, strict=False):
            updates.append(
                {
                    "row_index": raw_row["row_index"],
                    "Translator_Status": updated_row.get("Translator_Status", ""),
                    "Armed_Playbook_ID": updated_row.get("Armed_Playbook_ID", ""),
                    "Translator_Notes": updated_row.get("Translator_Notes", ""),
                }
            )
        bionic_client.batch_update_rows(
            rows=updates,
            columns=["Translator_Status", "Armed_Playbook_ID", "Translator_Notes"],
        )

    return session_path, report_path, payload


def load_manual_entries_sheet(path: str | Path) -> list[ManualEntryRow]:
    return _parse_manual_rows(_load_csv_rows(Path(path)))


def build_manual_trigger_manifest(row: ManualEntryRow) -> dict[str, Any]:
    dte_min, dte_max = _manual_dte_range(row)
    delta_min, delta_max = _manual_delta_range(row.expected_move)
    stop_loss_pct = float(row.stop_loss_pct if row.stop_loss_pct is not None else 0.45)
    profit_target_multiple = float(row.profit_target_multiple if row.profit_target_multiple is not None else 1.5)
    hard_flat_time_et = row.hard_flat_time_et or "15:55"
    max_risk_usd = float(row.max_risk_usd if row.max_risk_usd is not None else 300.0)

    signature = hashlib.sha256(
        json.dumps(
            {
                "symbol": row.symbol,
                "direction": row.direction,
                "trigger_price": row.trigger_price,
                "trigger_direction": row.trigger_direction,
                "expected_move": row.expected_move,
                "option_week_to_play": row.option_week_to_play,
                "day": row.day,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    deployment_id = f"manual_trigger_{row.symbol.lower()}_{row.direction}_{signature}"
    return {
        "deployment_id": deployment_id,
        "enabled": True,
        "symbol": row.symbol,
        "strategy": {
            "key": "manual_trigger",
            "version": 1,
            "params": {
                "direction": row.direction,
                "trigger_price": row.trigger_price,
                "trigger_direction": row.trigger_direction,
                "close_by_factor": 0.001,
                "after_time_et": row.execute_after,
            },
        },
        "execution": {
            "profile": "single_leg_long_premium_v1",
            "option_mapping": {
                "long_signal": "CALL",
                "short_signal": "PUT",
            },
            "dte_min": dte_min,
            "dte_max": dte_max,
            "target_abs_delta_min": delta_min,
            "target_abs_delta_max": delta_max,
            "min_open_interest": 100,
            "max_bid_ask_spread_pct": 0.20,
        },
        "risk": {
            "profile": "manual_trigger_v1",
            "max_trade_premium_usd": max_risk_usd,
            "hard_flat_time_et": hard_flat_time_et,
            "stop_loss_pct": stop_loss_pct,
        },
        "exit": {
            "profile": "manual_trigger_exit_v1",
            "use_algorithmic_exit": False,
            "use_profit_target": True,
            "profit_target_multiple": profit_target_multiple,
            "stop_loss_pct": stop_loss_pct,
            "hard_flat_time_et": hard_flat_time_et,
            "thesis_exit_anchor": None,
            "thesis_exit_policy": None,
            "thesis_exit_params": {},
            "catastrophe_exit_anchor": "option_premium",
            "catastrophe_exit_params": {
                "stop_loss_pct": stop_loss_pct,
                "hard_flat_time_et": hard_flat_time_et,
            },
        },
        "source": {
            "origin": "operator_manual",
            "run_date": (row.idea_date.isoformat() if row.idea_date is not None else datetime.now(UTC).date().isoformat()),
            "artifact": "entry_v1",
            "metadata": {
                "row_index": row.row_index,
                "expected_move": row.expected_move,
                "option_week_to_play": row.option_week_to_play,
                "day": row.day,
                "notes": row.notes,
                "trade_id": row.trade_id,
                "status": row.status,
                "execute_after": row.execute_after,
            },
        },
    }


def publish_active_session_to_bhiksha(
    *,
    session_payload_path: str | Path,
    bhiksha_root: str | Path,
) -> Path:
    root = Path(bhiksha_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    target = root / "artifacts" / "playbook" / "active_session.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(session_payload_path).resolve()
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _manual_row_from_sheet_row(row: dict[str, Any]) -> ManualEntryRow:
    normalized = {
        "row_index": row.get("row_index"),
        "stock": row.get("stock") or row.get("Stock") or row.get("Symbol"),
        "trigger_price": row.get("trigger_price") or row.get("Trigger_Price"),
        "trigger_direction": row.get("trigger_direction") or row.get("Trigger_Direction") or "ABOVE",
        "direction": row.get("direction") or row.get("Direction"),
        "expected_move": row.get("expected_move") or row.get("Expected_Move") or "2%",
        "option_week_to_play": row.get("option_week_to_play") or row.get("Option_Week_To_Play") or "1",
        "status": row.get("status") or row.get("Status") or "",
        "is_signal_active": row.get("is_signal_active") or row.get("Is_Signal_Active") or "1",
        "trade_id": row.get("trade_id") or row.get("Trade_ID") or "",
        "notes": row.get("notes") or row.get("Notes") or "",
        "idea_date": row.get("idea_date") or row.get("Idea_Date"),
        "execute_after": row.get("execute_after") or row.get("Execute_After"),
        "day": row.get("day") or row.get("Day") or "",
        "max_risk_usd": row.get("Max_Risk_USD"),
        "stop_loss_pct": row.get("Stop_Loss_Pct"),
        "profit_target_multiple": row.get("Profit_Target_Multiple"),
        "hard_flat_time_et": row.get("Hard_Flat_Time_ET"),
    }
    return ManualEntryRow.model_validate(normalized)


def _parse_manual_rows(rows: list[dict[str, Any]]) -> list[ManualEntryRow]:
    parsed: list[ManualEntryRow] = []
    for row in rows:
        try:
            parsed.append(_manual_row_from_sheet_row(row))
        except (ValidationError, ValueError, TypeError):
            continue
    return parsed


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _manual_dte_range(row: ManualEntryRow) -> tuple[int, int]:
    try:
        horizon = max(int(str(row.option_week_to_play).strip() or "1"), 1)
    except ValueError:
        horizon = 1
    if str(row.day).strip() in {"1", "true", "TRUE", "yes"}:
        return max(horizon - 1, 0), horizon + 1
    upper = max(horizon * 7, 7)
    lower = max(upper - 7, 0)
    return lower, upper


def _manual_delta_range(expected_move: str) -> tuple[float, float]:
    text = str(expected_move or "").strip().replace("%", "")
    try:
        pct = abs(float(text)) / 100.0
    except ValueError:
        pct = 0.02
    if pct <= 0.01:
        return 0.45, 0.60
    if pct <= 0.02:
        return 0.35, 0.55
    return 0.25, 0.45


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _apply_session_authorization(manifest: dict[str, Any], *, live_authorized: bool) -> dict[str, Any]:
    payload = json.loads(json.dumps(manifest))
    payload.setdefault("execution", {})
    payload["execution"]["shadow_only"] = not live_authorized
    payload.setdefault("source", {})
    payload["source"].setdefault("metadata", {})
    payload["source"]["metadata"]["authorization_mode"] = "live" if live_authorized else "shadow"
    return payload


__all__ = [
    "ACTIVE_SESSION_CONTRACT_NAME",
    "ACTIVE_SESSION_SCHEMA_VERSION",
    "ManualEntryRow",
    "build_manual_trigger_manifest",
    "compile_active_session_from_google_sheets",
    "compile_active_session_from_rows",
    "load_manual_entries_sheet",
    "publish_active_session_to_bhiksha",
]
