# M6 Execution Mapping + Opening Drive v2 (2026-03-08)

## Scope Completed

1. Added M6 execution mapping script:
- `scripts/run_execution_mapping.py`
- Uses holdout-promoted candidates and maps to option structures (DTE, delta, risk rules).
- Runs Monte Carlo execution stress via:
  - `src/oracle/monte_carlo.py`

2. Added strategy factory for script reuse:
- `src/strategy/factory.py`
- `scripts/run_holdout_validation.py` now uses the shared factory.

3. Added Opening Drive v2 capability:
- `src/strategy/opening_drive_classifier.py` now supports:
  - direction filters (`allow_long`, `allow_short`)
  - mode filters (`enable_continue`, `enable_fail`)
  - custom labels (`strategy_label`)
- Added v2 variant to research pipelines:
  - `scripts/run_novel_ideas.py`
  - `scripts/run_walk_forward_novel.py`

4. Results DB mapping improved:
- `src/oracle/results_db.py` now maps additional fields:
  - `min_oos_signals`, `holdout_trades`
  - `base_exp_r`, `mc_exp_r_p50`
  - `holdout_win_rate`

## Key Outputs

Execution mapping run:
- `data/results/execution_mapping_summary_2026-03-08_17-55-53.csv`
- `data/results/execution_mapping_summary_2026-03-08_17-55-53.md`

Opening Drive v2 run:
- `data/results/opening_drive_summary_2026-03-08_17-56-12.csv`
- `data/results/opening_drive_robustness_2026-03-08_17-56-12.csv`
- `data/results/opening_drive_mode_summary_2026-03-08_17-56-12.csv`

Focused walk-forward probe:
- `data/results/walk_forward_novel_summary_2026-03-08_odv2_probe.csv`
- `data/results/walk_forward_novel_detail_2026-03-08_odv2_probe.csv`

## Quick Read of Results

### Execution Mapping (holdout-promoted candidates)
- QQQ Elastic short:
  - Base expectancy remains positive.
  - Stress test shows meaningful degradation risk under conservative fill/theta assumptions.
- TSLA Compression short:
  - Stronger trade count.
  - Stress test still vulnerable to cost/fill assumptions.

Interpretation:
- Both candidates survive directional holdout.
- Execution assumptions are likely the dominant next bottleneck.

### Opening Drive v2 (short continuation only)
- QQQ:
  - 55 short signals
  - MFE/MAE ~1.37 (promising shape)
- NVDA:
  - 47 short signals
  - MFE/MAE ~0.99 (borderline)

Interpretation:
- v2 refinement improved signal quality on QQQ vs broad opening-drive baseline.
- Keep v2 in convergence pipeline and require full multi-cost + holdout confirmation before promotion.

Walk-forward probe note:
- `QQQ | Opening Drive v2 (Short Continue) | short` showed positive OOS expectancy in the probe run.
- `NVDA | Opening Drive v2 (Short Continue)` was not robust OOS in the same probe.

## Validation

Full tests:
- `34 passed`
