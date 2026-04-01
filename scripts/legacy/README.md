# Legacy Scripts

These runners are archived for historical provenance and occasional comparison work.

They are not part of the canonical research workflow and should not receive new productized logic unless the goal is explicitly to preserve or replay a historical experiment.

For active development, treat `src/research/`, `src/oracle/`, and `src/newton/` as the canonical agent-driven surface.

Policy for future changes:

- Do not add new research features here.
- Do not fix bugs here first if the canonical path can be fixed instead.
- Only touch these files to preserve compatibility, replay a historical run, or document provenance.
- If a legacy behavior is still needed, migrate the behavior into reusable modules rather than extending the legacy runner.

Moved here during the refactor:

- `run_elastic_grid.py`
- `run_walk_forward_novel.py`
- `run_targeted_retune.py`
- `run_convergence_pipeline.py`
- `run_holdout_validation.py`
- `run_execution_mapping.py`
- `run_market_impulse.py`
- `run_opening_drive_classifier.py`
- `run_measurement_sensitivity.py`
- `run_sweep.py`
- `run_stage_flip.py`
- `run_jerk_pivot_backtest.py`
- `run_novel_ideas.py`
- `run_p1_evaluation.py`
