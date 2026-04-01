# Script Support Matrix

This repo is now intentionally API-first.

## Top-Level Scripts

Only these files should be treated as active entrypoints under `scripts/`:

| script | status | reason |
|---|---|---|
| `run_research_orchestrator.py` | keep | Only supported top-level orchestration CLI for inspection and stage-safe planning. |
| `query_results_db.py` | keep | Utility for querying stored research results. |

## Quarantined Legacy Runners

All runner-style CLIs now live under `scripts/legacy/`. They are preserved for provenance, replay, or comparison only.

This is now an agent-driven, API-first repo:

- New research logic should land in `src/research/`, `src/oracle/`, or `src/newton/`.
- New orchestration should go through `ResearchOrchestrator` and `src/research/tools.py`.
- Legacy runners should be treated as archived, not as alternative active flows.
- If a legacy script exposes behavior we still need, migrate that behavior into reusable modules instead of extending the script.

| script | status | note |
|---|---|---|
| `legacy/run_elastic_grid.py` | quarantined | Historical M1 discovery wrapper; superseded by `parameter_sweep`. |
| `legacy/run_walk_forward_novel.py` | quarantined | Historical walk-forward CLI; reusable logic lives in `src/research/stages/walk_forward.py`. |
| `legacy/run_targeted_retune.py` | quarantined | Historical bounded-retune CLI; move future work into research tools. |
| `legacy/run_convergence_pipeline.py` | quarantined | Historical convergence wrapper; do not treat as canonical orchestration. |
| `legacy/run_holdout_validation.py` | quarantined | Historical holdout wrapper over reusable stage logic. |
| `legacy/run_execution_mapping.py` | quarantined | Historical execution-mapping wrapper over reusable stage logic. |
| `legacy/run_market_impulse.py` | quarantined | Specialized trade-simulator runner preserved for comparison until fully absorbed into the API surface. |
| `legacy/run_opening_drive_classifier.py` | quarantined | Strategy-specific reporting harness preserved for reference. |
| `legacy/run_measurement_sensitivity.py` | quarantined | Superseded by policy-driven ratio evaluation and modern research tooling. |
| `legacy/run_sweep.py` | quarantined | Old opening-bell experiment harness. |
| `legacy/run_stage_flip.py` | quarantined | Exploratory Market Impulse stage-flip simulator. |
| `legacy/run_jerk_pivot_backtest.py` | quarantined | Strategy-specific exploratory backtest with synthetic-data mode. |
| `legacy/run_novel_ideas.py` | quarantined | Batch comparison harness kept for provenance. |
| `legacy/run_p1_evaluation.py` | quarantined | Historical bundled experiment pack kept for provenance. |
