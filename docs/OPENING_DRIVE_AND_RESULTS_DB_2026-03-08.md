# Opening Drive Classifier + Results DB (2026-03-08)

## What Was Added

1. **Opening Drive Classifier strategy**
- File: `src/strategy/opening_drive_classifier.py`
- Exported from: `src/strategy/__init__.py`
- Core logic:
  - Classify opening drive using first 25 minutes after open.
  - `continue` entries: break in direction of opening drive.
  - `fail` entries: reclaim through opening midpoint against opening drive.
  - Uses acceleration, jerk, volume-vs-opening baseline, and directional mass filter.

2. **Dedicated run script**
- File: `scripts/run_opening_drive_classifier.py`
- Produces:
  - `opening_drive_summary_*.csv`
  - `opening_drive_robustness_*.csv`
  - `opening_drive_mode_summary_*.csv`

3. **Local SQLite results store**
- File: `src/oracle/results_db.py`
- Database path: `data/results/results.db`
- Tables:
  - `runs`
  - `artifact_rows`
- Added ingestion from core scripts:
  - `scripts/run_novel_ideas.py`
  - `scripts/run_walk_forward_novel.py`
  - `scripts/run_convergence_pipeline.py`
  - `scripts/run_holdout_validation.py`
  - `scripts/run_opening_drive_classifier.py`

4. **DB query helper**
- File: `scripts/query_results_db.py`
- Example:
  - `./.venv/bin/python scripts/query_results_db.py --artifact-type walk_forward_novel_summary --limit 20`

5. **uv project metadata**
- Added `pyproject.toml` for `uv sync` / `uv run` workflow.
- Existing `requirements.txt` remains available.

## Tests Added

- `tests/test_results_db.py`
- `tests/test_strategy_novel.py` now includes `TestOpeningDriveClassifierStrategy`.

## Validation Done

- Command:
  - `./.venv/bin/pytest tests/test_results_db.py tests/test_strategy_novel.py -q`
- Result: **12 passed**

## Smoke Backtest Snapshot

Run:
- `./.venv/bin/python scripts/run_opening_drive_classifier.py --tickers SPY --start 2025-12-01 --end 2026-02-28 --bootstrap-iters 500`

Observed from run output:
- Signals: 17 total (7 long, 10 short; 14 continue, 3 fail)
- Combined MFE/MAE: 0.82x
- Short MFE/MAE: 0.94x
- Not positive expectancy yet on this slice, but pipeline and DB persistence are functioning.
