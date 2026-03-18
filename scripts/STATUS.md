# Script Support Matrix

This repo is intentionally moving away from "every script is equally official."

Use the categories below when deciding where to add work.

## Canonical Research Runners

These are the supported entrypoints for the deterministic M1-M5 flow and results inspection.

| script | status | reason |
|---|---|---|
| `run_research_orchestrator.py` | keep | Canonical inspection entrypoint for tracked strategies, validation fixtures, and next allowed actions. |
| `run_walk_forward_novel.py` | keep | Canonical M3-style reusable walk-forward runner. |
| `run_convergence_pipeline.py` | keep | Canonical convergence gate runner until orchestration is fully library-driven. |
| `run_holdout_validation.py` | keep | Canonical holdout validation runner. |
| `run_execution_mapping.py` | keep | Canonical execution mapping runner. |
| `query_results_db.py` | keep | Canonical artifact/result inspection utility. |

## Supported Specialized Runners

These are still supported because they cover behavior not yet fully absorbed into the research workflow.

| script | status | reason |
|---|---|---|
| `run_market_impulse.py` | keep_specialized | Owns trade-simulator workflow and Market Impulse execution path that is not yet part of the M1-M5 research stack. |

## Migrate Next

These still provide useful behavior, but new work should move into `src/research/` tools/stages instead of extending these scripts directly.

| script | status | migration target |
|---|---|---|
| `run_novel_ideas.py` | migrate | Fold ratio-grid robustness and directional summary reporting into callable research tools. |
| `run_elastic_grid.py` | migrate | Replace with `parameter_sweep` / M1 discovery tooling over reusable stages. |
| `run_targeted_retune.py` | migrate | Replace with bounded `retune` tool flow for tracked strategies. |
| `run_p1_evaluation.py` | migrate | Split into reusable ablation and grid-search tool calls. |
| `run_opening_drive_classifier.py` | migrate | Move strategy-specific robustness reporting onto the shared research tool surface. |

## Archived Legacy Scripts

These are historical experiments or one-off harnesses that have been moved under `scripts/legacy/`. Avoid adding new logic here unless preserving provenance is the specific goal.

| script | status | note |
|---|---|---|
| `legacy/run_measurement_sensitivity.py` | archived | Superseded by policy-driven ratio evaluation and modern research runners. |
| `legacy/run_sweep.py` | archived | Old opening-bell experiment harness; not part of the current research workflow. |
| `legacy/run_stage_flip.py` | archived | Exploratory Market Impulse stage-flip simulator, not a canonical strategy pipeline. |
| `legacy/run_jerk_pivot_backtest.py` | archived | Strategy-specific exploratory backtest with synthetic-data mode; preserve only if needed for provenance. |

## Deletion Rule

Delete or move a script out of the top-level `scripts/` surface only when all three are true:

1. Its unique logic exists in `src/research/` or another reusable module.
2. The refactor validation set still passes through the supported path.
3. Results inspection remains possible through the supported artifacts or `results.db`.
