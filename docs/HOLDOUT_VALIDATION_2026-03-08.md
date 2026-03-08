# Holdout Validation (M5) — 2026-03-08

## Objective

Validate only promoted candidates from M4 on a holdout segment that is not used for ratio fitting.

## Setup

- Source candidates:
  - `data/results/convergence_gate_report_2026-03-08_15-48-19.csv`
  - Filter: `decision == promote_to_holdout`
- Candidates tested:
  - `AAPL | Elastic Band Reversion | long`
  - `AAPL | Elastic Band Reversion | combined`
  - `META | Elastic Band Reversion | short`

Time split:
- Calibration: `2024-01-01` → `2025-11-30`
- Holdout: `2025-12-01` → `2026-02-28`

Stress assumptions:
- `cost_r`: `0.05`, `0.08`, `0.12`
- Ratio grid: `1.0, 1.25, 1.5, 2.0`
- Minimum calibration signals: `200`
- Minimum holdout signals: `500`

## Command

```bash
python scripts/run_holdout_validation.py
```

## Artifacts

- `data/results/holdout_validation_detail_2026-03-08_16-04-50.csv`
- `data/results/holdout_validation_summary_2026-03-08_16-04-50.csv`

## Result

All 3 candidates passed holdout gates across all cost points:
- `META | Elastic Band Reversion | short` (min holdout Exp(R): `+0.139`, min N: `796`)
- `AAPL | Elastic Band Reversion | combined` (min holdout Exp(R): `+0.063`, min N: `1410`)
- `AAPL | Elastic Band Reversion | long` (min holdout Exp(R): `+0.044`, min N: `763`)

## Interpretation

- The shortlisted Elastic Band variants retained positive expectancy under harsher friction in holdout.
- This is strong enough to move to M6:
  - map directional signals to concrete options structures,
  - include conservative execution/theta assumptions before any live pilot.
