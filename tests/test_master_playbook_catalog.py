from __future__ import annotations

from pathlib import Path

from src.config import settings
import src.research.playbooks as playbooks_module
from src.research.playbooks import (
    BiasInputRow,
    PlaybookRecord,
    PLAYBOOK_STATUS_ACTIVE,
    PLAYBOOK_STATUS_RETIRED,
    PLAYBOOK_STATUS_STALE,
    merge_master_playbook_catalog,
    refresh_master_playbook_catalog_statuses,
    route_bias_rows,
)


def test_master_catalog_merge_preserves_existing_and_updates_same_playbook(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "master_playbook_catalog.json"
    projection_path = tmp_path / "master_playbook_catalog.csv"
    monkeypatch.setattr(settings, "master_playbook_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "master_playbook_projection_path", str(projection_path))

    spy = _playbook(
        symbol="SPY",
        playbook_id="market_impulse_spy_short_v1",
        expectancy=0.35,
        last_validated_date="2026-04-01",
    )
    tsla = _playbook(
        symbol="TSLA",
        playbook_id="market_impulse_tsla_short_v1",
        expectancy=0.18,
        last_validated_date="2026-04-01",
    )

    _, _, seeded = merge_master_playbook_catalog(
        new_playbooks=[spy],
        sync_sheet=False,
    )
    assert [playbook.symbol for playbook in seeded] == ["SPY"]
    _, _, merged = merge_master_playbook_catalog(
        new_playbooks=[spy.model_copy(update={"expectancy": 0.41}), tsla],
        sync_sheet=False,
    )

    assert {playbook.symbol for playbook in merged} == {"SPY", "TSLA"}
    updated_spy = next(playbook for playbook in merged if playbook.symbol == "SPY")
    assert updated_spy.validation_count == 2
    assert updated_spy.first_validated_date == "2026-04-01"
    assert updated_spy.expectancy == 0.41


def test_master_catalog_refresh_marks_stale_and_respects_retired_override(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "master_playbook_catalog.json"
    projection_path = tmp_path / "master_playbook_catalog.csv"
    monkeypatch.setattr(settings, "master_playbook_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "master_playbook_projection_path", str(projection_path))

    old_active = _playbook(
        symbol="SPY",
        playbook_id="market_impulse_spy_short_v1",
        last_validated_date="2026-01-01",
    )
    retired = _playbook(
        symbol="TSLA",
        playbook_id="market_impulse_tsla_short_v1",
        last_validated_date="2026-04-01",
        operator_status_override="retired",
    )
    merge_master_playbook_catalog(new_playbooks=[old_active, retired], sync_sheet=False)

    _, _, refreshed = refresh_master_playbook_catalog_statuses(sync_sheet=False)

    assert next(playbook for playbook in refreshed if playbook.symbol == "SPY").lifecycle_status == PLAYBOOK_STATUS_STALE
    assert next(playbook for playbook in refreshed if playbook.symbol == "TSLA").lifecycle_status == PLAYBOOK_STATUS_RETIRED


def test_master_catalog_applies_sheet_retired_override(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "master_playbook_catalog.json"
    projection_path = tmp_path / "master_playbook_catalog.csv"
    monkeypatch.setattr(settings, "master_playbook_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "master_playbook_projection_path", str(projection_path))
    monkeypatch.setattr(settings, "master_playbook_sheet_id", "sheet-1")
    monkeypatch.setattr(settings, "google_api_credentials_path", "/tmp/fake-credentials.json")

    class FakeSheetClient:
        def __init__(self, **kwargs):
            pass

        def read_rows(self, **kwargs):
            return [
                {
                    "catalog_key": "",
                    "playbook_id": "market_impulse_spy_short_v1",
                    "operator_status_override": "retired",
                    "operator_notes": "manual live override",
                }
            ]

    monkeypatch.setattr(playbooks_module, "GoogleSheetTableClient", FakeSheetClient)

    spy = _playbook(
        symbol="SPY",
        playbook_id="market_impulse_spy_short_v1",
    )
    _, _, merged = merge_master_playbook_catalog(
        new_playbooks=[spy],
        sync_sheet=False,
    )

    assert merged[0].lifecycle_status == PLAYBOOK_STATUS_RETIRED
    assert merged[0].operator_notes == "manual live override"


def test_route_bias_rows_defaults_to_master_catalog(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "master_playbook_catalog.json"
    projection_path = tmp_path / "master_playbook_catalog.csv"
    monkeypatch.setattr(settings, "master_playbook_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "master_playbook_projection_path", str(projection_path))

    spy = _playbook(
        symbol="SPY",
        playbook_id="market_impulse_spy_short_v1",
        bias_template="bearish_trend_intraday",
        bionic_ready=True,
        is_full_m1_m5_survivor=True,
        lifecycle_status=PLAYBOOK_STATUS_ACTIVE,
    )
    merge_master_playbook_catalog(new_playbooks=[spy], sync_sheet=False)

    biases = [
        BiasInputRow.model_validate(
            {
                "date": "2026-04-02",
                "symbol": "SPY",
                "daily_bias": "Bearish",
                "intraday_thesis": "Trend_Continuation",
                "max_risk_usd": 500,
            }
        )
    ]
    routing_report_path, armed_payloads_path, selections, _ = route_bias_rows(
        biases=biases,
        out_dir=tmp_path / "routing",
    )
    assert routing_report_path.exists()
    assert armed_payloads_path.exists()
    assert selections[0].status == "armed"
    assert selections[0].selected_playbook_id == "market_impulse_spy_short_v1"


def _playbook(
    *,
    symbol: str,
    playbook_id: str,
    bias_template: str = "bearish_trend_intraday",
    expectancy: float = 0.2,
    last_validated_date: str = "2026-04-01",
    operator_status_override: str = "",
    bionic_ready: bool = True,
    is_full_m1_m5_survivor: bool = True,
    lifecycle_status: str = PLAYBOOK_STATUS_ACTIVE,
) -> PlaybookRecord:
    payload = {
        "playbook_id": playbook_id,
        "strategy_key": "market_impulse",
        "strategy_family": "market_impulse",
        "strategy_display_name": "Market Impulse (Cross & Reclaim)",
        "symbol": symbol,
        "symbol_scope": [symbol],
        "direction": "short",
        "bias_template": bias_template,
        "horizon": "intraday",
        "regime_tags": ["bearish", "trend", "intraday", "market_impulse"],
        "lifecycle_status": lifecycle_status,
        "operator_status_override": operator_status_override,
        "operator_notes": "",
        "automation_status": "shadow_ready",
        "surface_class": "supported",
        "entry_params": {"direction": "short", "entry_buffer_minutes": 3, "entry_window_minutes": 45, "regime_timeframe": "30m"},
        "exit_params": {"thesis_exit_policy": "fixed_rr_underlying"},
        "thesis_exit_anchor": "underlying",
        "thesis_exit_policy": "fixed_rr_underlying",
        "thesis_exit_params": {"stop_loss_underlying_pct": 0.005, "take_profit_underlying_r_multiple": 2.0},
        "catastrophe_exit_anchor": "option_premium",
        "catastrophe_exit_params": {"stop_loss_pct": 0.45, "hard_flat_time_et": "15:55"},
        "execution_mapping": {
            "profile": "single_leg_long_premium_v1",
            "option_mapping": {"long_signal": "CALL", "short_signal": "PUT"},
            "dte_min": 0,
            "dte_max": 7,
        },
        "risk_mapping": {"profile": "conservative_day1", "max_trade_premium_usd": 500},
        "deployment_manifest_template": {
            "symbol": symbol,
            "strategy": {"key": "market_impulse", "version": 1, "params": {"direction": "short"}},
            "execution": {
                "profile": "single_leg_long_premium_v1",
                "shadow_only": True,
                "option_mapping": {"long_signal": "CALL", "short_signal": "PUT"},
                "dte_min": 0,
                "dte_max": 7,
            },
            "risk": {"profile": "conservative_day1", "max_trade_premium_usd": 500},
            "exit": {"stop_loss_pct": 0.45, "hard_flat_time_et": "15:55"},
        },
        "time_window": {},
        "vehicle_mapping": {"profile": "single_leg_long_premium_v1"},
        "passes_m1": True,
        "passes_m2": True,
        "passes_m3": True,
        "passes_m4": True,
        "passes_m5": True,
        "is_full_m1_m5_survivor": is_full_m1_m5_survivor,
        "expectancy": expectancy,
        "confidence": 0.6,
        "signal_count": 50,
        "execution_robustness": 0.7,
        "stress_metrics": {},
        "first_validated_date": last_validated_date,
        "last_validated_date": last_validated_date,
        "validation_count": 1,
        "research_slice_id": "market_impulse-slice",
        "bionic_ready": bionic_ready,
        "source": {},
        "bhiksha_compatibility": {"bionic_ready": bionic_ready},
    }
    return PlaybookRecord.model_validate(payload)
