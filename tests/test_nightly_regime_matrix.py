from __future__ import annotations

import csv
import json
from pathlib import Path

from src.research.loop_contracts import LOOP_ARTIFACT_SCHEMA_VERSION
from src.research.nightly_matrix import (
    NightlyRegimeMatrixConfig,
    _run_family_research,
    load_nightly_regime_matrix_config,
    run_nightly_regime_matrix,
)


def test_load_nightly_regime_matrix_config_and_run_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "nightly_regime_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "output_root: data/results/nightly_regime_matrix",
                "watchlist:",
                "  - iwm",
                "  - tsla",
                "enabled_strategy_families:",
                "  - market_impulse",
                "  - jerk_pivot_momentum",
                "  - elastic_band_reversion",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_nightly_regime_matrix_config(config_path)
    config.research_control_root = str(tmp_path / "control")
    assert config.watchlist == ["IWM", "TSLA"]

    def fake_family_runner(
        family: str,
        loaded_config: NightlyRegimeMatrixConfig,
        bundle_dir: Path,
    ) -> tuple[Path, Path]:
        run_dir = bundle_dir / family
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = bundle_dir / "logs" / f"{family}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"OUT_DIR={run_dir}\n", encoding="utf-8")
        if family == "market_impulse":
            _write_run(
                run_dir,
                strategy="Market Impulse (Cross & Reclaim)",
                ticker="IWM",
                direction="short",
                m5_decision="promote",
                m4_summary_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n",
                m4_summary_row="IWM,Market Impulse (Cross & Reclaim),short,5,60,1h,3,117,0.3195,0.3786,true,true,promote_to_execution_mapping\n",
                m4_detail_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,cost_bps,selected_ratio,calib_signals,calib_exp_r,holdout_signals,holdout_confidence,holdout_exp_r,passes_cost_gate\n",
                m4_detail_rows=[
                    "IWM,Market Impulse (Cross & Reclaim),short,5,60,1h,5.0,2.0,595,0.0533,117,0.5043,0.4323,true\n",
                ],
                m5_detail_header="ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n",
                m5_detail_row="IWM,Market Impulse (Cross & Reclaim),short,5,60,1h,2.0,117,0.5043,0.4328,117.0,0.205046,-0.016752,0.206823,0.425053,0.936,-1.960005,24.198289,49.73124,10.485098,put_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n",
            )
        elif family == "jerk_pivot_momentum":
            _write_run(
                run_dir,
                strategy="Jerk-Pivot Momentum (tight)",
                ticker="TSLA",
                direction="long",
                m5_decision="promote",
                m4_summary_header="ticker,strategy,direction,vpoc_proximity_pct,jerk_lookback,volume_multiplier,volume_ma_period,use_volume_filter,use_time_filter,session_start,session_end,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n",
                m4_summary_row="TSLA,Jerk-Pivot Momentum (tight),long,0.002,10,1.3,20,true,true,09:35,15:30,3,64,0.2211,0.3012,true,true,promote_to_execution_mapping\n",
                m4_detail_header="ticker,strategy,direction,vpoc_proximity_pct,jerk_lookback,volume_multiplier,volume_ma_period,use_volume_filter,use_time_filter,session_start,session_end,cost_bps,selected_ratio,calib_signals,calib_exp_r,holdout_signals,holdout_confidence,holdout_exp_r,passes_cost_gate\n",
                m4_detail_rows=[
                    "TSLA,Jerk-Pivot Momentum (tight),long,0.002,10,1.3,20,true,true,09:35,15:30,5.0,1.5,222,0.1811,64,0.6021,0.3111,true\n",
                ],
                m5_detail_header="ticker,strategy,direction,vpoc_proximity_pct,jerk_lookback,volume_multiplier,volume_ma_period,use_volume_filter,use_time_filter,session_start,session_end,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n",
                m5_detail_row="TSLA,Jerk-Pivot Momentum (tight),long,0.002,10,1.3,20,true,true,09:35,15:30,1.5,64,0.6021,0.4111,64.0,0.298,-0.022,0.301,0.544,0.811,-1.1,12.2,26.4,4.8,long_call,7-21,0.35-0.55,09:45-14:30,2.0R,-35% premium\n",
            )
        else:
            _write_run(
                run_dir,
                strategy="Elastic Band Reversion",
                ticker="IWM",
                direction="long",
                m5_decision="gather_more_evidence",
                m4_summary_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n",
                m4_summary_row="IWM,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,3,42,0.1313,0.1621,true,true,promote_to_execution_mapping\n",
                m4_detail_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,cost_bps,selected_ratio,calib_signals,calib_exp_r,holdout_signals,holdout_confidence,holdout_exp_r,passes_cost_gate\n",
                m4_detail_rows=[
                    "IWM,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,5.0,1.25,321,0.0812,42,0.5476,0.1901,true\n",
                ],
                m5_detail_header="ticker,strategy,direction,z_score_threshold,z_score_window,use_directional_mass,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n",
                m5_detail_row="IWM,Elastic Band z=3.0/w=120+dm,long,3.0,120,true,1.25,42,0.5476,0.1521,42.0,-0.020538,-0.310285,-0.021474,0.25991,0.45275,-13.03198,-0.90191,10.916207,7.684631,call_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n",
            )
        return run_dir, log_path

    result = run_nightly_regime_matrix(
        config,
        bundle_dir=tmp_path / "bundle",
        family_runner=fake_family_runner,
    )

    deployment_candidates = json.loads(result.deployment_candidates_path.read_text(encoding="utf-8"))
    playbook_catalog = json.loads(result.playbook_catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert deployment_candidates["schema_version"] == LOOP_ARTIFACT_SCHEMA_VERSION
    assert playbook_catalog["schema_version"] == LOOP_ARTIFACT_SCHEMA_VERSION
    assert sorted(result.run_dirs) == [
        "elastic_band_reversion",
        "jerk_pivot_momentum",
        "market_impulse",
    ]
    assert manifest["config_watchlist"] == ["IWM", "TSLA"]
    assert manifest["contracts"]["deployment_candidates"]["schema_version"] == LOOP_ARTIFACT_SCHEMA_VERSION
    assert manifest["family_logs"]["market_impulse"].endswith("logs/market_impulse.log")
    assert result.review_queue_path.exists()
    assert result.review_history_path.exists()
    assert result.review_workbook_path.exists()
    assert playbook_catalog["contexts"]["TSLA|bullish_trend_intraday|intraday"]["coverage_status"] == "researched_with_survivors"
    assert playbook_catalog["contexts"]["IWM|bullish_mean_reversion_intraday|intraday"]["proposed_candidates"]
    assert playbook_catalog["contexts"]["TSLA|bearish_mean_reversion_intraday|intraday"]["coverage_status"] == "researched_no_survivors"


def test_run_family_research_streams_output_to_log(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "fake_family.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                "assert '--max-stage' in sys.argv",
                "assert sys.argv[sys.argv.index('--max-stage') + 1] == 'M2'",
                "print('line-1')",
                "print('line-2')",
                "out_dir = Path('family_output').resolve()",
                "out_dir.mkdir(parents=True, exist_ok=True)",
                "print(f'OUT_DIR={out_dir}')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import src.research.nightly_matrix as nightly_matrix

    monkeypatch.setattr(nightly_matrix, "PROJECT_ROOT", project_root)
    monkeypatch.setitem(nightly_matrix._FAMILY_TO_SCRIPT, "market_impulse", "scripts/fake_family.py")

    config = NightlyRegimeMatrixConfig(enabled_strategy_families=["market_impulse"])
    run_dir, log_path = _run_family_research("market_impulse", config, tmp_path / "bundle")

    assert run_dir == (project_root / "family_output").resolve()
    assert log_path == (tmp_path / "bundle" / "logs" / "market_impulse.log").resolve()
    log_text = log_path.read_text(encoding="utf-8")
    assert "line-1" in log_text
    assert f"OUT_DIR={run_dir}" in log_text


def test_run_nightly_regime_matrix_merges_m2_only_scout_into_queue(tmp_path: Path) -> None:
    config = NightlyRegimeMatrixConfig(enabled_strategy_families=["market_impulse"])
    config.research_control_root = str(tmp_path / "control")

    def fake_family_runner(
        family: str,
        loaded_config: NightlyRegimeMatrixConfig,
        bundle_dir: Path,
    ) -> tuple[Path, Path]:
        run_dir = bundle_dir / family
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = bundle_dir / "logs" / f"{family}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"OUT_DIR={run_dir}\n", encoding="utf-8")
        (run_dir / "m1_top_candidates.csv").write_text(
            "\n".join(
                [
                    "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,avg_test_exp_r,pct_positive_oos_windows,oos_windows,oos_signals",
                    "SPY,Market Impulse (Cross & Reclaim),short,5,60,1h,0.12,0.67,7,140",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "m2_gate_report.csv").write_text(
            "\n".join(
                [
                    "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,passes_all_gates,decision,observed_cost_points,min_oos_windows,min_oos_signals,min_pct_positive_oos_windows,min_avg_test_exp_r,mean_avg_test_exp_r,mean_pct_positive_oos_windows,mean_test_confidence",
                    "SPY,Market Impulse (Cross & Reclaim),short,5,60,1h,true,promote_to_holdout,3,7,140,0.67,0.11,0.14,0.70,0.55",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "stages": [
                {
                    "stage": "M1",
                    "decision": "promote",
                    "recorded_at": "2026-04-01T01:00:00+00:00",
                    "artifacts": {
                        "top_candidates": str(run_dir / "m1_top_candidates.csv"),
                    },
                    "context": {"strategy_family": "Market Impulse (Cross & Reclaim)"},
                },
                {
                    "stage": "M2",
                    "decision": "promote",
                    "recorded_at": "2026-04-01T01:10:00+00:00",
                    "artifacts": {
                        "gate_report": str(run_dir / "m2_gate_report.csv"),
                    },
                    "context": {"strategy_family": "Market Impulse (Cross & Reclaim)"},
                },
            ]
        }
        (run_dir / "research_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return run_dir, log_path

    result = run_nightly_regime_matrix(
        config,
        bundle_dir=tmp_path / "bundle",
        family_runner=fake_family_runner,
    )

    deployment_candidates = json.loads(result.deployment_candidates_path.read_text(encoding="utf-8"))
    queue_rows = _read_csv_rows(result.review_queue_path)

    assert deployment_candidates["candidates"] == []
    assert len(queue_rows) == 1
    assert queue_rows[0]["ticker"] == "SPY"
    assert queue_rows[0]["queue_status"] == "NEW"
    assert queue_rows[0]["passes_m1"] == "True"
    assert queue_rows[0]["passes_m2"] == "True"
    assert queue_rows[0]["passes_m3"] == "False"
    assert queue_rows[0]["latest_stage_reached"] == "M2"


def test_run_family_research_failure_references_log(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / "fake_family_fail.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "print('bad-line')",
                "raise SystemExit(3)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import src.research.nightly_matrix as nightly_matrix

    monkeypatch.setattr(nightly_matrix, "PROJECT_ROOT", project_root)
    monkeypatch.setitem(nightly_matrix._FAMILY_TO_SCRIPT, "market_impulse", "scripts/fake_family_fail.py")

    config = NightlyRegimeMatrixConfig(enabled_strategy_families=["market_impulse"])

    try:
        _run_family_research("market_impulse", config, tmp_path / "bundle")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected _run_family_research to fail")

    assert "See log:" in message
    assert "bad-line" in message
    assert "STDOUT:" not in message


def test_nightly_regime_matrix_default_watchlist_includes_tier1_single_names() -> None:
    config = NightlyRegimeMatrixConfig()

    assert {"SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL"} <= set(config.watchlist)
    assert config.defaults.broad_scout_max_stage == "M2"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_run(
    run_dir: Path,
    *,
    strategy: str,
    ticker: str,
    direction: str,
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
                "recorded_at": "2026-03-31T19:55:58.630829+00:00",
                "artifacts": {
                    "summary": str(run_dir / "M4_holdout_validation_summary.csv"),
                    "detail": str(run_dir / "M4_holdout_validation_detail.csv"),
                },
                "context": {"strategy_family": strategy, "ticker": ticker, "direction": direction},
            },
            {
                "stage": "M5",
                "decision": m5_decision,
                "recorded_at": "2026-03-31T19:55:58.631909+00:00",
                "artifacts": {
                    "detail": str(run_dir / "M5_execution_mapping_detail.csv"),
                },
                "context": {"strategy_family": strategy, "ticker": ticker, "direction": direction},
            },
        ]
    }
    (run_dir / "research_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
