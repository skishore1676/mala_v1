from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from src.research.exit_backfill import backfill_exit_optimizations
from src.research.loop_contracts import PLAYBOOK_CATALOG_CONTRACT_NAME, build_contract_metadata


def test_backfill_exit_optimizations_writes_artifact_and_refreshes_catalog(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "followup"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "m5_execution_mapping.csv").write_text(
        "\n".join(
            [
                "ticker,strategy,direction,entry_buffer_minutes,entry_window_minutes,regime_timeframe,selected_ratio,holdout_trades,holdout_win_rate,base_exp_r,trades,mc_exp_r_mean,mc_exp_r_p05,mc_exp_r_p50,mc_exp_r_p95,mc_prob_positive_exp,mc_total_r_p05,mc_total_r_p50,mc_total_r_p95,mc_max_dd_p50,structure,dte,delta_plan,entry_window_et,profit_take,risk_rule",
                "SPY,Market Impulse (Cross & Reclaim),short,3,45,1h,1.5,25,0.60,0.20,25,0.15,0.05,0.16,0.21,0.82,-1.0,4.0,6.0,1.5,single_option,7-21,0.30-0.45,09:45-14:30,2.0R,hard stop at -45% premium",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.csv"
    _write_rows(
        queue_path,
        [
            {
                "candidate_key": "spy_market_impulse_short",
                "ticker": "SPY",
                "strategy": "Market Impulse (Cross & Reclaim)",
                "direction": "short",
                "config_json": json.dumps(
                    {"entry_buffer_minutes": 3, "entry_window_minutes": 45, "regime_timeframe": "1h"},
                    sort_keys=True,
                ),
                "latest_stage_reached": "M5",
                "is_full_m1_m5_survivor": "true",
                "latest_artifact_dir": str(artifact_dir),
                "queue_status": "EXECUTED",
                "last_action_run_date": "2026-04-01",
                "last_seen_run_date": "2026-04-01",
                "passes_m1": "true",
                "passes_m2": "true",
                "passes_m3": "true",
                "passes_m4": "true",
                "passes_m5": "true",
                "strategy_family": "market_impulse",
            }
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

    def frame_loader(ticker: str, start: date | None, end: date | None) -> pl.DataFrame:
        timestamps = [
            datetime(2026, 3, 16, 13, 35, tzinfo=timezone.utc),
            datetime(2026, 3, 16, 13, 36, tzinfo=timezone.utc),
            datetime(2026, 3, 16, 13, 37, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 13, 35, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 13, 36, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 13, 37, tzinfo=timezone.utc),
        ]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [ticker] * len(timestamps),
                "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
                "high": [100.2, 100.3, 100.4, 100.6, 100.7, 100.8],
                "low": [99.8, 99.9, 100.0, 100.1, 100.2, 100.3],
                "close": [99.9, 100.0, 100.1, 100.2, 100.3, 100.4],
                "volume": [1000.0] * len(timestamps),
                "vma_10": [100.0, 100.0, 100.05, 100.2, 100.25, 100.3],
                "impulse_regime_1h": ["bearish"] * len(timestamps),
            }
        )

    result = backfill_exit_optimizations(
        queue_path=queue_path,
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 31),
        holdout_start=date(2026, 3, 16),
        holdout_end=date(2026, 3, 17),
        playbook_catalog_path=catalog_path,
        playbook_projection_path=tmp_path / "playbook_catalog.csv",
        frame_loader=frame_loader,
        enricher=lambda frame, required: frame,
    )

    assert result.optimized == 1
    optimization_path = artifact_dir / "m5_exit_optimization.json"
    assert optimization_path.exists()
    payload = json.loads(optimization_path.read_text(encoding="utf-8"))
    assert payload["thesis_exit_anchor"] == "underlying"
    refreshed_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert refreshed_catalog["playbooks"][0]["bionic_ready"] is True


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
