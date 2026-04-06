# M1 - parameter_sweep

- Decision: `retune`
- Next action: `Retune opening-drive thresholds and optional filters within M1 while keeping regime confluence fixed at 5m.`

## Rationale
Bounded M1 sweep executed with the new 15-minute opening window and 5-minute regime confluence surface; signal quality stayed below discovery thresholds, so another bounded retune is required before any stage advance.

## Summary
- `strategy`: `Opening Drive Classifier`
- `config_count`: `8`
- `max_configs`: `8`
- `ticker_count`: `1`
- `detail_rows`: `14`
- `aggregate_rows`: `8`
- `parameter_count`: `12`
- `requested_config_count`: `8`
- `duplicate_config_count`: `0`
- `invalid_config_count`: `0`
- `stage_objective`: `find_edge_anywhere`

## Context
- `hypothesis`: `iwm-opening-range-regime-continuation`
- `ticker_scope`: `['IWM']`
- `repo_change_policy`: `implement_research_surface`

## Artifacts
- `detail`: `research/hypotheses/artifacts/iwm-opening-range-regime-continuation/2026-04-02T191830/M1_parameter_sweep_detail.csv`
- `aggregate`: `research/hypotheses/artifacts/iwm-opening-range-regime-continuation/2026-04-02T191830/M1_parameter_sweep_aggregate.csv`
