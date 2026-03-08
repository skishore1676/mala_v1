# Strategy Convergence Plan (2026-03-08)

## Goal

Converge to a small set of strategies with **repeatable out-of-sample edge** that can be moved toward live options execution rules.

## Milestones (Measurable)

## M1: Data + Session Integrity (Pass/Fail)
- Pass criteria:
  - UTC storage + ET session filtering verified.
  - No mixed-schema failures while loading cached bars.
- Current status: **Pass**

## M2: Baseline Strategy Set (Pass/Fail)
- Scope:
  - Elastic Band Reversion
  - Kinematic Ladder
  - Compression Breakout
  - Regime Router
- Pass criteria:
  - Strategies produce directional signals and can be evaluated by walk-forward.
  - Tests pass.
- Current status: **Pass**

## M3: Expanded-Universe Discovery (Scored)
- Scope:
  - Tickers: `SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL`
  - Window: `2024-01-01` to `2026-02-28`
- Metrics:
  - OOS windows count
  - OOS signals
  - Avg OOS expectancy
  - % positive OOS windows
- Current status: **Pass (initial)**

## M4: Robustness Under Friction Stress (Primary Gate)
- Scope:
  - Re-run walk-forward for `cost_r` grid: `0.05, 0.08, 0.12`
- Candidate gate (must pass all):
  - `oos_windows >= 6`
  - `oos_signals >= 3000`
  - `pct_positive_oos_windows >= 0.67`
  - `avg_test_exp_r >= 0` at every cost point
- Output:
  - Ranked shortlist of robust candidates.
- Current status: **In progress (implemented now, running each cycle)**

## M5: Holdout Confirmation (Promotion Gate)
- Scope:
  - Freeze top candidates from M4.
  - Evaluate on untouched holdout window.
- Pass criteria:
  - Positive expectancy and acceptable stability on holdout.
- Current status: **Pass (initial promoted set)**

## M6: Execution Mapping For Options (Pre-Live Gate)
- Scope:
  - Map directional entries to option structures (DTE, delta, spread rules).
  - Add conservative execution costs/theta assumptions.
- Pass criteria:
  - Positive expectancy after execution assumptions.
  - Clear risk limits and invalidation rules.
- Current status: **Pending**

## Decision Rules

- If a candidate fails M4, it is not promoted to holdout.
- If no candidate passes M4, iterate parameters only on best-in-class family (currently Elastic Band and selective short-side variants).
- Promote at most 3 candidates to M5 to avoid research sprawl.

## Immediate Next Action

Run automated convergence pipeline:
1. Execute walk-forward over friction grid.
2. Generate gate report and shortlist.
3. Promote only candidates that pass M4.

## Latest Execution (2026-03-08)

Run:
```bash
python scripts/run_convergence_pipeline.py \
  --tickers SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL \
  --start 2024-01-01 --end 2026-02-28 \
  --cost-grid 0.05,0.08,0.12 \
  --gate-min-oos-windows 6 \
  --gate-min-oos-signals 3000 \
  --gate-min-pct-positive 0.67 \
  --gate-min-exp-r 0.0
```

Artifacts:
- `data/results/convergence_cost_summary_2026-03-08_15-48-19.csv`
- `data/results/convergence_gate_report_2026-03-08_15-48-19.csv`
- `data/results/convergence_shortlist_2026-03-08_15-48-19.md`

M4 result:
- Candidates passing all gates: **3**
  - `AAPL | Elastic Band Reversion | long`
  - `AAPL | Elastic Band Reversion | combined`
  - `META | Elastic Band Reversion | short`

Updated status:
- M4: **Pass (first batch)**
- M5: **Pass (2025-12-01 to 2026-02-28 holdout)**

See holdout details:
- `docs/HOLDOUT_VALIDATION_2026-03-08.md`
