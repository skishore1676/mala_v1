from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from src.research.loop_contracts import PLAYBOOK_CATALOG_CONTRACT_NAME, build_contract_metadata
from src.research.playbooks import (
    BiasInputRow,
    LiveObservationRecord,
    PLAYBOOK_STATUS_ACTIVE,
    augment_playbook_catalog_from_queue,
    build_playbook_records_from_queue,
    load_bias_inputs_sheet,
    load_live_observation_records,
    load_playbook_records,
    route_bias_rows,
    route_google_sheet_bias_inputs,
    route_bias_inputs,
    write_live_observation_records,
)
from src.research.google_sheets import spreadsheet_id_from_url
from src.research.bhiksha_bridge import publish_armed_playbooks_to_bhiksha


def test_build_playbook_records_from_full_survivor_queue_rows(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Market Impulse (Cross & Reclaim)",
        ticker="IWM",
        direction="short",
    )
    _write_queue_rows(
        queue_path,
        [
            _full_survivor_row(
                ticker="IWM",
                strategy="Market Impulse (Cross & Reclaim)",
                family="market_impulse",
                direction="short",
                artifact_dir=artifact_dir,
                config_json={
                    "entry_buffer_minutes": 5,
                    "entry_window_minutes": 60,
                    "regime_timeframe": "1h",
                },
            )
        ],
    )

    records = build_playbook_records_from_queue(_read_csv_rows(queue_path))

    assert len(records) == 1
    record = records[0]
    assert record.strategy_key == "market_impulse"
    assert record.playbook_id.startswith("market_impulse_iwm_short_")
    assert record.lifecycle_status == PLAYBOOK_STATUS_ACTIVE
    assert record.bias_template == "bearish_trend_intraday"
    assert record.execution_mapping["profile"] == "single_leg_long_premium_v1"
    assert record.exit_params["thesis_exit_policy"] == "trailing_vma_underlying"
    assert record.thesis_exit_policy == "trailing_vma_underlying"
    assert record.catastrophe_exit_params["stop_loss_pct"] == 0.45
    assert record.deployment_manifest_template["strategy"]["params"]["direction"] == "short"
    assert record.is_full_m1_m5_survivor is True
    assert record.bhiksha_compatibility["supported"] is True
    assert record.bionic_ready is True


def test_augment_playbook_catalog_from_queue_adds_first_class_records(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Elastic Band Reversion",
        ticker="NVDA",
        direction="long",
    )
    _write_queue_rows(
        queue_path,
        [
            _full_survivor_row(
                ticker="NVDA",
                strategy="Elastic Band Reversion",
                family="elastic_band_reversion",
                direction="long",
                artifact_dir=artifact_dir,
                config_json={
                    "z_score_threshold": 3.0,
                    "z_score_window": 120,
                    "use_directional_mass": True,
                },
            )
        ],
    )
    catalog_path = tmp_path / "playbook_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
                "generated_at": "2026-04-01T00:00:00+00:00",
                "contexts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    augment_playbook_catalog_from_queue(
        playbook_catalog_path=catalog_path,
        queue_path=queue_path,
        playbook_projection_path=tmp_path / "playbook_catalog.csv",
    )

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert payload["playbook_count"] == 1
    assert payload["playbooks"][0]["playbook_id"].startswith("elastic_band_reversion_nvda_long_")
    assert payload["playbooks"][0]["automation_status"] == "shadow_ready"
    assert payload["playbooks"][0]["bhiksha_compatibility"]["supported"] is True
    assert payload["playbooks"][0]["bionic_ready"] is False
    projection_rows = _read_csv_rows(tmp_path / "playbook_catalog.csv")
    assert projection_rows[0]["symbol"] == "NVDA"


def test_load_bias_inputs_sheet_and_route_bias_inputs(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Market Impulse (Cross & Reclaim)",
        ticker="IWM",
        direction="short",
    )
    _write_queue_rows(
        queue_path,
        [
            _full_survivor_row(
                ticker="IWM",
                strategy="Market Impulse (Cross & Reclaim)",
                family="market_impulse",
                direction="short",
                artifact_dir=artifact_dir,
                config_json={
                    "entry_buffer_minutes": 5,
                    "entry_window_minutes": 60,
                    "regime_timeframe": "1h",
                },
            )
        ],
    )
    catalog_path = tmp_path / "playbook_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
                "generated_at": "2026-04-01T00:00:00+00:00",
                "contexts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    augment_playbook_catalog_from_queue(playbook_catalog_path=catalog_path, queue_path=queue_path)

    bias_path = tmp_path / "bias_sheet.csv"
    _write_queue_rows(
        bias_path,
        [
            {
                "Date": "4/2/26",
                "Symbol": "IWM",
                "Daily_Bias": "Bearish",
                "Intraday_Thesis": "Trend_Continuation",
                "Max_Risk_USD": "500",
                "Translator_Status": "",
                "Armed_Playbook_ID": "",
                "Notes": "risk-off follow-through",
            },
            {
                "Date": "4/2/26",
                "Symbol": "SPY",
                "Daily_Bias": "Bullish",
                "Intraday_Thesis": "Mean_Reversion",
                "Max_Risk_USD": "400",
                "Translator_Status": "",
                "Armed_Playbook_ID": "",
                "Notes": "",
            },
        ],
    )

    biases = load_bias_inputs_sheet(bias_path)
    assert biases[0].bias_template == "bearish_trend_intraday"
    assert biases[1].bias_template == "bullish_mean_reversion_intraday"

    routing_report_path, armed_payloads_path, selections = route_bias_inputs(
        bias_inputs_path=bias_path,
        playbook_catalog_path=catalog_path,
        out_dir=tmp_path / "routing",
    )

    report_rows = _read_csv_rows(routing_report_path)
    assert report_rows[0]["Translator_Status"] == "armed"
    assert report_rows[0]["Armed_Playbook_ID"]
    assert report_rows[1]["Translator_Status"] == "no_match"

    armed_payloads = json.loads(armed_payloads_path.read_text(encoding="utf-8"))
    assert len(armed_payloads["armed_playbooks"]) == 1
    manifest_path = Path(armed_payloads["armed_playbooks"][0]["manifest_path"])
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "max_trade_premium_usd: 500.0" in manifest_text
    assert "thesis_exit_policy: trailing_vma_underlying" in manifest_text
    assert selections[0].status == "armed"
    assert selections[1].status == "no_match"


def test_route_bias_inputs_outputs_bhiksha_compatible_manifest(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Jerk-Pivot Momentum (tight)",
        ticker="TSLA",
        direction="long",
        confidence=0.62,
    )
    _write_queue_rows(
        queue_path,
        [
            _full_survivor_row(
                ticker="TSLA",
                strategy="Jerk-Pivot Momentum (tight)",
                family="jerk_pivot_momentum",
                direction="long",
                artifact_dir=artifact_dir,
                config_json={
                    "vpoc_proximity_pct": 0.002,
                    "jerk_lookback": 10,
                    "volume_multiplier": 1.3,
                    "volume_ma_period": 20,
                    "use_volume_filter": True,
                    "use_time_filter": True,
                    "session_start": "09:35",
                    "session_end": "15:30",
                },
            )
        ],
    )
    catalog_path = tmp_path / "playbook_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
                "generated_at": "2026-04-01T00:00:00+00:00",
                "contexts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    augment_playbook_catalog_from_queue(playbook_catalog_path=catalog_path, queue_path=queue_path)
    bias_path = tmp_path / "bias_sheet.csv"
    _write_queue_rows(
        bias_path,
        [
            {
                "Date": "2026-04-02",
                "Symbol": "TSLA",
                "Daily_Bias": "Bullish",
                "Intraday_Thesis": "Trend_Continuation",
                "Max_Risk_USD": "1000",
                "Translator_Status": "",
                "Armed_Playbook_ID": "",
                "Notes": "AI infra strength",
            }
        ],
    )

    _, armed_payloads_path, _ = route_bias_inputs(
        bias_inputs_path=bias_path,
        playbook_catalog_path=catalog_path,
        out_dir=tmp_path / "routing",
    )
    manifest_path = Path(json.loads(armed_payloads_path.read_text(encoding="utf-8"))["armed_playbooks"][0]["manifest_path"])

    bhiksha_root = Path(__file__).resolve().parents[2] / "bhiksha" / "src"
    sys.path.insert(0, str(bhiksha_root))
    try:
        from bhiksha.config.models import DeploymentManifest

        manifest_payload = yaml_safe_load(manifest_path)
        validated = DeploymentManifest.model_validate(manifest_payload)
        assert validated.symbol == "TSLA"
        assert validated.risk.max_trade_premium_usd == 1000.0
        assert validated.exit.thesis_exit_policy == "fixed_rr_underlying"
    finally:
        sys.path.remove(str(bhiksha_root))


def test_live_observation_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "observations.json"
    records = [
        LiveObservationRecord(
            playbook_id="market_impulse_iwm_short_abc123",
            deployment_id="market_impulse_iwm_short_armed_deadbeef",
            symbol="IWM",
            armed=True,
            triggered=False,
            exit_exercised=False,
            realized_outcome_summary={"realized_r": 0.0},
            recorded_at="2026-04-01T12:00:00+00:00",
        )
    ]

    write_live_observation_records(records, path)
    loaded = load_live_observation_records(path)

    assert loaded == records


def test_spreadsheet_id_from_url_extracts_id() -> None:
    url = "https://docs.google.com/spreadsheets/d/1cJPWfkQB6pp91TAFNT86R5Pi1cUfzCgT3bUWgjY6rbc/edit?gid=1907235657#gid=1907235657"
    assert spreadsheet_id_from_url(url) == "1cJPWfkQB6pp91TAFNT86R5Pi1cUfzCgT3bUWgjY6rbc"
    assert spreadsheet_id_from_url("abc123") == "abc123"


def test_route_google_sheet_bias_inputs_updates_machine_columns(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Market Impulse (Cross & Reclaim)",
        ticker="SPY",
        direction="short",
    )
    _write_queue_rows(
        queue_path,
        [
            _full_survivor_row(
                ticker="SPY",
                strategy="Market Impulse (Cross & Reclaim)",
                family="market_impulse",
                direction="short",
                artifact_dir=artifact_dir,
                config_json={
                    "entry_buffer_minutes": 3,
                    "entry_window_minutes": 45,
                    "regime_timeframe": "1h",
                },
            )
        ],
    )
    catalog_path = tmp_path / "playbook_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                **build_contract_metadata(PLAYBOOK_CATALOG_CONTRACT_NAME),
                "generated_at": "2026-04-01T00:00:00+00:00",
                "contexts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    augment_playbook_catalog_from_queue(playbook_catalog_path=catalog_path, queue_path=queue_path)

    rows = [
        {
            "Date": "2026-04-02",
            "Symbol": "SPY",
            "Daily_Bias": "Bearish",
            "Intraday_Thesis": "Trend_Continuation",
            "Max_Risk_USD": "500",
            "Enabled": "TRUE",
            "Translator_Status": "",
            "Armed_Playbook_ID": "",
            "Translator_Notes": "",
            "Notes": "",
            "row_index": 2,
        }
    ]
    updates: list[dict[str, object]] = []

    class FakeSheetClient:
        def __init__(self, **_: object) -> None:
            pass

        def read_rows(self) -> list[dict[str, object]]:
            return rows

        def batch_update_rows(self, *, rows: list[dict[str, object]], columns: list[str]) -> dict[str, object]:
            updates.extend(rows)
            assert columns == ["Translator_Status", "Armed_Playbook_ID", "Translator_Notes"]
            return {"updatedRows": len(rows)}

    monkeypatch.setattr("src.research.playbooks.GoogleSheetTableClient", FakeSheetClient)

    _, _, selections = route_google_sheet_bias_inputs(
        spreadsheet_id="1cJPWfkQB6pp91TAFNT86R5Pi1cUfzCgT3bUWgjY6rbc",
        sheet_name="Bionic_Loop",
        credentials_path=tmp_path / "creds.json",
        playbook_catalog_path=catalog_path,
        out_dir=tmp_path / "routing",
        update_sheet=True,
    )

    assert selections[0].status == "armed"
    assert updates
    assert updates[0]["row_index"] == 2
    assert updates[0]["Translator_Status"] == "armed"
    assert str(updates[0]["Armed_Playbook_ID"]).startswith("market_impulse_spy_short_")


def test_publish_armed_playbooks_to_bhiksha_copies_generated_manifests(tmp_path: Path) -> None:
    bhiksha_root = tmp_path / "bhiksha"
    generated_dir = bhiksha_root / "config" / "deployments" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    armed_manifest = tmp_path / "market_impulse_spy_short_armed_deadbeef.yaml"
    armed_manifest.write_text(
        "\n".join(
            [
                "deployment_id: market_impulse_spy_short_armed_deadbeef",
                "symbol: SPY",
                "strategy:",
                "  key: market_impulse",
                "  version: 1",
                "  params: {}",
                "execution:",
                "  profile: single_leg_long_premium_v1",
                "risk:",
                "  profile: conservative_day1",
                "exit:",
                "  profile: market_impulse_exit_v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    armed_payloads_path = tmp_path / "armed_playbooks.json"
    armed_payloads_path.write_text(
        json.dumps(
            {
                "armed_playbooks": [
                    {
                        "manifest_path": str(armed_manifest),
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = publish_armed_playbooks_to_bhiksha(
        armed_playbooks_path=armed_payloads_path,
        bhiksha_root=bhiksha_root,
    )

    published = generated_dir / armed_manifest.name
    assert published.exists()
    assert report.published_count == 1
    assert Path(report.import_report_path).exists()


def _write_full_survivor_artifacts(
    artifact_dir: Path,
    *,
    strategy: str,
    ticker: str,
    direction: str,
    confidence: float = 0.55,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if strategy.startswith("Market Impulse"):
        m4_header = "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n"
        m4_row = f"{ticker},{strategy},{direction},5,60,1h,3,117,0.3195,0.3786,true,true,promote_to_execution_mapping\n"
        m5_header = "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n"
        m5_row = f"{ticker},{strategy},{direction},5,60,1h,2.0,117,0.5043,0.4328,117.0,0.205046,-0.016752,0.206823,0.425053,0.936,-1.960005,24.198289,49.73124,10.485098,put_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n"
    elif strategy.startswith("Jerk-Pivot Momentum"):
        m4_header = "ticker,strategy,direction,vpoc_proximity_pct,jerk_lookback,volume_multiplier,volume_ma_period,use_volume_filter,use_time_filter,session_start,session_end,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n"
        m4_row = f"{ticker},{strategy},{direction},0.002,10,1.3,20,true,true,09:35,15:30,3,64,0.2211,0.3012,true,true,promote_to_execution_mapping\n"
        m5_header = "ticker,strategy,direction,vpoc_proximity_pct,jerk_lookback,volume_multiplier,volume_ma_period,use_volume_filter,use_time_filter,session_start,session_end,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n"
        m5_row = f"{ticker},{strategy},{direction},0.002,10,1.3,20,true,true,09:35,15:30,1.5,64,0.6021,0.4111,64.0,0.298,-0.022,0.301,0.544,0.811,-1.1,12.2,26.4,4.8,long_call,7-21,0.35-0.55,09:45-14:30,2.0R,-35% premium\n"
    else:
        m4_header = "ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n"
        m4_row = f"{ticker},{strategy},{direction},3.0,120,true,3,42,0.1313,0.1621,true,true,promote_to_execution_mapping\n"
        m5_header = "ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n"
        m5_row = f"{ticker},{strategy},{direction},3.0,120,true,1.25,42,0.5476,0.1521,42.0,-0.020538,-0.310285,-0.021474,0.25991,0.45275,-13.03198,-0.90191,10.916207,7.684631,call_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n"
    (artifact_dir / "m4_holdout_summary.csv").write_text(m4_header + m4_row, encoding="utf-8")
    (artifact_dir / "m5_execution_mapping.csv").write_text(m5_header + m5_row, encoding="utf-8")
    if strategy.startswith("Market Impulse"):
        optimization = {
            "generated_at": "2026-04-01T00:00:00+00:00",
            "strategy_key": "market_impulse",
            "symbol": ticker,
            "direction": direction,
            "selection_metric": "expectancy",
            "selection_slice": {"holdout_start": "2026-03-01", "holdout_end": "2026-03-31"},
            "selected_policy_name": "trailing_vma_underlying:vma_10",
            "thesis_exit_anchor": "underlying",
            "thesis_exit_policy": "trailing_vma_underlying",
            "thesis_exit_params": {"vma_col": "vma_10"},
            "catastrophe_exit_anchor": "option_premium",
            "catastrophe_exit_params": {"stop_loss_pct": 0.45, "hard_flat_time_et": "15:55"},
            "selected_metrics": {"trade_count": 12, "expectancy": 0.41, "win_rate": 0.58, "profit_factor": 1.4, "total_pnl": 4.9},
            "candidate_policies": [],
        }
        (artifact_dir / "m5_exit_optimization.json").write_text(json.dumps(optimization, indent=2) + "\n", encoding="utf-8")
    elif strategy.startswith("Jerk-Pivot Momentum"):
        optimization = {
            "generated_at": "2026-04-01T00:00:00+00:00",
            "strategy_key": "jerk_pivot_momentum",
            "symbol": ticker,
            "direction": direction,
            "selection_metric": "expectancy",
            "selection_slice": {"holdout_start": "2026-03-01", "holdout_end": "2026-03-31"},
            "selected_policy_name": "fixed_rr_underlying:0.0035x1.50",
            "thesis_exit_anchor": "underlying",
            "thesis_exit_policy": "fixed_rr_underlying",
            "thesis_exit_params": {"stop_loss_underlying_pct": 0.0035, "take_profit_underlying_r_multiple": 1.5},
            "catastrophe_exit_anchor": "option_premium",
            "catastrophe_exit_params": {"stop_loss_pct": 0.35, "hard_flat_time_et": "15:55"},
            "selected_metrics": {"trade_count": 9, "expectancy": 0.22, "win_rate": 0.56, "profit_factor": 1.3, "total_pnl": 2.0},
            "candidate_policies": [],
        }
        (artifact_dir / "m5_exit_optimization.json").write_text(json.dumps(optimization, indent=2) + "\n", encoding="utf-8")


def _full_survivor_row(
    *,
    ticker: str,
    strategy: str,
    family: str,
    direction: str,
    artifact_dir: Path,
    config_json: dict[str, object],
) -> dict[str, object]:
    return {
        "candidate_key": f"{family}_{ticker}_{direction}",
        "strategy_family": family,
        "ticker": ticker,
        "strategy": strategy,
        "direction": direction,
        "research_slice_id": f"{family}-slice-abc123",
        "config_signature": "deadbeefcafebabe",
        "config_json": json.dumps(config_json, sort_keys=True),
        "chart_link": str((artifact_dir / "chart.html").resolve()),
        "human_decision": "promote_to_m3",
        "human_notes": "",
        "priority": 1,
        "human_updated_at": "2026-04-01T01:00:00+00:00",
        "queue_status": "EXECUTED",
        "latest_stage_reached": "M5",
        "latest_stage_decision": "promote",
        "passes_m1": True,
        "passes_m2": True,
        "passes_m3": True,
        "passes_m4": True,
        "passes_m5": True,
        "is_full_m1_m5_survivor": True,
        "last_seen_run_date": "2026-04-01",
        "last_action_run_date": "2026-04-01",
        "last_source_run_dir": str((artifact_dir / "run").resolve()),
        "latest_artifact_dir": str(artifact_dir.resolve()),
        "m1_oos_signals": 140,
        "m1_avg_test_confidence": 0.58,
        "m2_mean_test_confidence": 0.61,
    }


def _write_queue_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def yaml_safe_load(path: Path) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
