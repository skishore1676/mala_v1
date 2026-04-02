from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path

from src.research.loop_contracts import PLAYBOOK_CATALOG_CONTRACT_NAME, build_contract_metadata
from src.research.playbooks import BiasInputRow, augment_playbook_catalog_from_queue
from src.research.session_payloads import (
    ManualEntryRow,
    compile_active_session_from_rows,
    publish_active_session_to_bhiksha,
)


def test_compile_active_session_manual_override_suppresses_playbook(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    artifact_dir = tmp_path / "followup"
    _write_full_survivor_artifacts(
        artifact_dir,
        strategy="Market Impulse (Cross & Reclaim)",
        ticker="SPY",
        direction="short",
    )
    _write_rows(
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

    bias = BiasInputRow.model_validate(
        {
            "date": "2026-04-02",
            "symbol": "SPY",
            "daily_bias": "Bearish",
            "intraday_thesis": "Trend_Continuation",
            "max_risk_usd": 500,
        }
    )
    manual = ManualEntryRow.model_validate(
        {
            "stock": "SPY",
            "trigger_price": 505.5,
            "trigger_direction": "BELOW",
            "direction": "p",
            "expected_move": "2%",
            "option_week_to_play": "1",
            "status": "PENDING",
            "is_signal_active": "1",
            "idea_date": "2026-04-02",
            "execute_after": "09:45",
            "notes": "manual override",
        }
    )

    session_path, report_path, payload = compile_active_session_from_rows(
        biases=[bias],
        manual_entries=[manual],
        playbook_catalog_path=catalog_path,
        out_dir=tmp_path / "session",
        session_date=date(2026, 4, 2),
    )

    assert session_path.exists()
    assert report_path.exists()
    assert payload["summary"]["deployment_count"] == 1
    deployment = payload["deployments"][0]
    assert deployment["strategy"]["key"] == "manual_trigger"
    assert deployment["source"]["origin"] == "operator_manual"
    assert payload["suppressed"][0]["reason"] == "manual_override"


def test_publish_active_session_to_bhiksha_copies_payload(tmp_path: Path) -> None:
    session_path = tmp_path / "active_session.json"
    session_path.write_text(
        json.dumps(
            {
                "contract_name": "active_session",
                "schema_version": 1,
                "session_id": "active_session_2026-04-02",
                "deployments": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bhiksha_root = tmp_path / "bhiksha"

    published = publish_active_session_to_bhiksha(
        session_payload_path=session_path,
        bhiksha_root=bhiksha_root,
    )

    assert published.exists()
    assert json.loads(published.read_text(encoding="utf-8"))["session_id"] == "active_session_2026-04-02"


def _write_full_survivor_artifacts(
    artifact_dir: Path,
    *,
    strategy: str,
    ticker: str,
    direction: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "m4_holdout_summary.csv").write_text(
        "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,observed_cost_points,min_holdout_signals,min_holdout_exp_r,mean_holdout_exp_r,passes_all_cost_gates,passes_holdout,decision\n"
        f"{ticker},{strategy},{direction},3,45,1h,3,117,0.31,0.37,true,true,promote_to_execution_mapping\n",
        encoding="utf-8",
    )
    (artifact_dir / "m5_execution_mapping.csv").write_text(
        "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule\n"
        f"{ticker},{strategy},{direction},3,45,1h,2.0,117,0.5043,0.4328,117.0,0.205046,-0.016752,0.206823,0.425053,0.936,-1.960005,24.198289,49.73124,10.485098,put_debit_spread,7-21,long 0.30-0.45 / short 0.10-0.25,09:45-14:30,50-70% spread value,hard stop at -45% premium\n",
        encoding="utf-8",
    )
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
        "priority": 1,
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


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
