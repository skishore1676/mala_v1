from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_agentic_jerk_pivot_pass.py"
    spec = spec_from_file_location("run_agentic_jerk_pivot_pass", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_m4_m5_uses_shared_holdout_contract(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_run_holdout_validation_for_candidates(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "run_holdout_validation_for_candidates", fake_run_holdout_validation_for_candidates)

    holdout_detail, holdout_summary, execution_df = module.run_m4_m5(
        frames={},
        metrics=object(),
        promoted_candidates=pl.DataFrame(),
        start=date(2024, 1, 2),
        calibration_end=date(2025, 11, 30),
        holdout_start=date(2025, 12, 1),
        holdout_end=date(2026, 2, 28),
        ratios=[1.0, 1.25],
        cost_grid_bps=[5.0, 8.0],
        min_calibration_signals=40,
        min_holdout_signals=20,
        base_cost_r=0.08,
        bootstrap_iters=100,
    )

    assert captured["start_date"] == date(2024, 1, 2)
    assert captured["costs"] == [5.0, 8.0]
    assert captured["min_holdout_signals"] == 20
    assert "calibration_start" not in captured
    assert "cost_bps_grid" not in captured
    assert holdout_detail.is_empty()
    assert holdout_summary.is_empty()
    assert execution_df.is_empty()
