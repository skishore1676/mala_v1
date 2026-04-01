from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.loop_contracts import (
    DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
    PLAYBOOK_CATALOG_CONTRACT_NAME,
    LOOP_ARTIFACT_SCHEMA_VERSION,
    validate_contract_metadata,
)
from src.research.loop_export import LoopArtifactExporter


def test_loop_export_builds_supported_and_proposed_artifacts(tmp_path: Path) -> None:
    market_run = tmp_path / "market_impulse_run"
    elastic_run = tmp_path / "elastic_band_run"
    market_run.mkdir()
    elastic_run.mkdir()

    _write_run(
        market_run,
        strategy="Market Impulse (Cross & Reclaim)",
        m5_decision="promote",
        m4_summary_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n",
        m4_summary_row="QQQ,Market Impulse (Cross & Reclaim),short,5,60,1h,3,117,0.3195,0.3786,true,true,promote_to_execution_mapping\n",
        m4_detail_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,cost_bps,selected_ratio,calib_signals,calib_exp_r,holdout_signals,holdout_confidence,holdout_exp_r,passes_cost_gate\n",
        m4_detail_rows=[
            "QQQ,Market Impulse (Cross & Reclaim),short,5,60,1h,5.0,2.0,595,0.0533,117,0.5043,0.4323,true\n",
            "QQQ,Market Impulse (Cross & Reclaim),short,5,60,1h,8.0,2.0,595,0.0197,117,0.5043,0.3839,true\n",
        ],
        m5_detail_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n",
        m5_detail_row="QQQ,Market Impulse (Cross & Reclaim),short,5,60,1h,2.0,117,0.5043,0.4328,117.0,0.205046,-0.016752,0.206823,0.425053,0.936,-1.960005,24.198289,49.73124,10.485098,put_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n",
    )
    _write_run(
        elastic_run,
        strategy="Elastic Band Reversion",
        m5_decision="gather_more_evidence",
        m4_summary_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n",
        m4_summary_row="NVDA,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,3,42,0.1313,0.1621,true,true,promote_to_execution_mapping\n",
        m4_detail_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,cost_bps,selected_ratio,calib_signals,calib_exp_r,holdout_signals,holdout_confidence,holdout_exp_r,passes_cost_gate\n",
        m4_detail_rows=[
            "NVDA,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,5.0,1.25,321,0.0812,42,0.5476,0.1901,true\n",
            "NVDA,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,8.0,1.25,321,0.0612,42,0.5476,0.1649,true\n",
        ],
        m5_detail_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n",
        m5_detail_row="NVDA,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,1.25,42,0.5476,0.1521,42.0,-0.020538,-0.310285,-0.021474,0.25991,0.45275,-13.03198,-0.90191,10.916207,7.684631,call_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n",
    )

    out_dir = tmp_path / "loop_artifacts"
    exporter = LoopArtifactExporter()
    candidates_path, playbook_path = exporter.export_runs(
        [market_run, elastic_run],
        out_dir=out_dir,
        watchlist=["QQQ", "NVDA", "IWM"],
        enabled_strategy_families=["market_impulse", "jerk_pivot_momentum", "elastic_band_reversion"],
    )

    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    playbook_payload = json.loads(playbook_path.read_text(encoding="utf-8"))

    validate_contract_metadata(
        candidates_payload,
        expected_contract_name=DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
    )
    validate_contract_metadata(
        playbook_payload,
        expected_contract_name=PLAYBOOK_CATALOG_CONTRACT_NAME,
    )
    assert candidates_payload["schema_version"] == LOOP_ARTIFACT_SCHEMA_VERSION
    assert playbook_payload["schema_version"] == LOOP_ARTIFACT_SCHEMA_VERSION
    assert candidates_payload["watchlist"] == ["IWM", "NVDA", "QQQ"]
    assert playbook_payload["watchlist"] == ["IWM", "NVDA", "QQQ"]

    assert {candidate["surface_class"] for candidate in candidates_payload["candidates"]} == {"supported", "proposed"}

    market_candidate = next(
        candidate for candidate in candidates_payload["candidates"] if candidate["strategy_key"] == "market_impulse"
    )
    assert market_candidate["automation_status"] == "shadow_ready"
    assert market_candidate["bias_template"] == "bearish_trend_intraday"
    assert market_candidate["manifest"]["execution"]["profile"] == "single_leg_long_premium_v1"
    assert market_candidate["manifest"]["execution"]["shadow_only"] is True
    assert market_candidate["manifest"]["source"]["metadata"]["candidate_id"] == market_candidate["candidate_id"]

    elastic_candidate = next(
        candidate for candidate in candidates_payload["candidates"] if candidate["strategy_key"] == "elastic_band_reversion"
    )
    assert elastic_candidate["automation_status"] == "manual_research_only"
    assert elastic_candidate["bias_template"] == "bullish_mean_reversion_intraday"
    assert elastic_candidate["manifest"]["execution"]["profile"] == "manual_research_only"
    assert "Spread-aware execution stress and live monitoring" in elastic_candidate["manifest"]["source"]["metadata"]["required_bhiksha_capabilities"]

    market_context = playbook_payload["contexts"]["QQQ|bearish_trend_intraday|intraday"]
    assert market_context["coverage_status"] == "researched_with_survivors"
    assert market_context["covered_by_strategy_families"] == ["market_impulse", "jerk_pivot_momentum"]
    assert market_context["supported_candidates"][0]["candidate_id"] == market_candidate["candidate_id"]
    assert market_context["proposed_candidates"] == []

    elastic_context = playbook_payload["contexts"]["NVDA|bullish_mean_reversion_intraday|intraday"]
    assert elastic_context["coverage_status"] == "researched_with_survivors"
    assert elastic_context["covered_by_strategy_families"] == ["elastic_band_reversion"]
    assert elastic_context["supported_candidates"] == []
    assert elastic_context["proposed_candidates"][0]["candidate_id"] == elastic_candidate["candidate_id"]

    empty_trend_context = playbook_payload["contexts"]["IWM|bullish_trend_intraday|intraday"]
    assert empty_trend_context["coverage_status"] == "researched_no_survivors"
    assert empty_trend_context["supported_candidates"] == []
    assert empty_trend_context["proposed_candidates"] == []

    empty_reversion_context = playbook_payload["contexts"]["QQQ|bullish_mean_reversion_intraday|intraday"]
    assert empty_reversion_context["coverage_status"] == "researched_no_survivors"
    assert empty_reversion_context["covered_by_strategy_families"] == ["elastic_band_reversion"]


def test_validate_contract_metadata_rejects_missing_or_wrong_fields() -> None:
    with pytest.raises(ValueError, match="Missing required contract_name"):
        validate_contract_metadata({"schema_version": LOOP_ARTIFACT_SCHEMA_VERSION})

    with pytest.raises(ValueError, match="Unexpected contract_name"):
        validate_contract_metadata(
            {
                "contract_name": PLAYBOOK_CATALOG_CONTRACT_NAME,
                "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
            },
            expected_contract_name=DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
        )

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        validate_contract_metadata(
            {
                "contract_name": DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
                "schema_version": 1,
            },
            expected_contract_name=DEPLOYMENT_CANDIDATES_CONTRACT_NAME,
        )


def _write_run(
    run_dir: Path,
    *,
    strategy: str,
    m5_decision: str,
    m4_summary_header: str,
    m4_summary_row: str,
    m4_detail_header: str,
    m4_detail_rows: list[str],
    m5_detail_header: str,
    m5_detail_row: str,
) -> None:
    (run_dir / "M4_holdout_validation_summary.csv").write_text(
        m4_summary_header + m4_summary_row,
        encoding="utf-8",
    )
    (run_dir / "M4_holdout_validation_detail.csv").write_text(
        m4_detail_header + "".join(m4_detail_rows),
        encoding="utf-8",
    )
    (run_dir / "M5_execution_mapping_detail.csv").write_text(
        m5_detail_header + m5_detail_row,
        encoding="utf-8",
    )
    manifest = {
        "stages": [
            {
                "stage": "M4",
                "decision": "promote",
                "recorded_at": "2026-03-30T19:55:58.630829+00:00",
                "artifacts": {
                    "summary": str(run_dir / "M4_holdout_validation_summary.csv"),
                    "detail": str(run_dir / "M4_holdout_validation_detail.csv"),
                },
                "context": {"strategy_family": strategy},
            },
            {
                "stage": "M5",
                "decision": m5_decision,
                "recorded_at": "2026-03-30T19:55:58.631909+00:00",
                "artifacts": {
                    "detail": str(run_dir / "M5_execution_mapping_detail.csv"),
                },
                "context": {"strategy_family": strategy},
            },
        ]
    }
    (run_dir / "research_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
