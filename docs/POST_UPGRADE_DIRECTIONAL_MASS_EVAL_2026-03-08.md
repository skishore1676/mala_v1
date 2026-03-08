# Post-Upgrade Evaluation (Directional Mass + Dynamic Elasticity) — 2026-03-08

## Scope

Evaluated the new Elastic Band implementation using:
- **Directional Mass** (`internal_strength`, `directional_mass`)
- **Dynamic Elasticity** (rolling z-score stretch vs VPOC)

Test universe and window remained unchanged for comparability:
- Tickers: `SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL`
- Range: `2024-01-01` to `2026-02-28`
- Cost grid for convergence: `0.05, 0.08, 0.12`

## Comparable Runs

Pre-upgrade baseline:
- `data/results/novel_strategy_robustness_2026-03-08_15-21-51.csv`
- `data/results/convergence_gate_report_2026-03-08_15-48-19.csv`

Post-upgrade:
- `data/results/novel_strategy_robustness_2026-03-08_16-16-01.csv`
- `data/results/convergence_gate_report_2026-03-08_16-19-45.csv`
- `data/results/convergence_shortlist_2026-03-08_16-19-45.md`

## What Changed

### Signal density
- Elastic Band signals dropped sharply:
  - from `143,256` to `18,849` (all tickers, all tested ratios aggregate)
  - about **-86.8%**

Interpretation:
- New logic is much more selective.
- It filters aggressively, especially on directional mass polarity and z-score extremes.

### Robustness / expectancy
- Elastic average expectancy by ratio decreased vs baseline (mean deltas):
  - `1.0`: `-0.016`
  - `1.25`: `-0.017`
  - `1.5`: `-0.017`
  - `2.0`: `-0.020`

### Convergence promotion outcome (strict M4 gate)
- Previous config: **3 promoted**
- New config: **0 promoted**

## Interesting pockets

With strict gates, no promotion.

With a relaxed M4 signal gate (`min_oos_signals >= 500`), positive short-side pockets appeared:
- `IWM | Elastic Band | short`
- `TSLA | Elastic Band | short`
- `META | Elastic Band | short`
- `QQQ | Elastic Band | short`

Relaxed report:
- `data/results/convergence_gate_report_relaxed500_2026-03-08.csv`

## Relaxed holdout check

Command:
- `python scripts/run_holdout_validation.py --gate-report data/results/convergence_gate_report_relaxed500_2026-03-08.csv --min-calibration-signals 150 --min-holdout-signals 200`

Result:
- No candidate passed promoted holdout gate (sample still small; mixed holdout expectancy).
- Best two by min holdout expectancy:
  - `IWM short`: `+0.1566` (min holdout signals `94`)
  - `QQQ short`: `+0.1022` (min holdout signals `135`)

Artifacts:
- `data/results/holdout_validation_summary_2026-03-08_16-21-16.csv`
- `data/results/holdout_validation_detail_2026-03-08_16-21-16.csv`

## End-of-day conclusion

- The upgrade is promising conceptually but currently too restrictive under the existing gating standards.
- Net effect today: **better selectivity, weaker aggregate robustness**.
- Next cycle should retune the new Elastic hyperparameters:
  - `z_score_threshold`
  - `z_score_window`
  - optional directional-mass strength filter (magnitude threshold, not just sign)
