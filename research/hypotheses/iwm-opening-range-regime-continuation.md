# IWM Opening Range Regime Continuation

## Status
- state: `pending`
- allowed_values: `pending | running | retune | blocked | completed`
- owner: `codex`
- created_at: `2026-04-02`
- last_run_at: `2026-04-02T11:03:49-0500`
- next_action: `create a fresh codex/ branch, implement the smallest opening-range + regime-confluence research-surface increment with 15m support, then rerun bounded M1`

## Repo Change Policy
- repo_change_policy: `implement_research_surface`
- allowed_values: `propose | implement_research_surface`
- branch_rule: `if repo changes are allowed, create a fresh codex/ branch first and do not merge automatically`

## Hypothesis
If IWM crosses the 15-minute market open boundary in one direction, that move should continue when it is aligned with the 5-minute directional regime.

## Constraints
- symbol_scope: `IWM`
- preferred_strategy_family: `opening range continuation` or nearest existing implementation
- deterministic_gate_limit: `M1-M5`
- max_stage_this_run: `smallest valid stage sequence`
- allowed_parameter_budget: `bounded; prefer the smallest honest search space`
- date_range: `use repo defaults unless the chosen tool requires a narrower slice`
- notes:
  - prefer an existing strategy or template before proposing implementation work
  - respect all current deterministic gates and holdout boundaries
  - repo changes are allowed only for the smallest honest Mala research-surface increment needed to test this hypothesis
  - if repo edits are needed, create a fresh `codex/` branch first and do not merge automatically

## Mapping Hints
- nearest_existing_strategy: `Opening Drive Classifier`, `Market Impulse`, or a clearly named opening-range template if those do not fit exactly
- trigger_event: `cross of the 15-minute opening range high/low`
- directional_thesis: `continuation`
- timeframe_alignment: `5-minute regime must agree with trade direction`
- invalidation_conditions:
  - no stable M1 edge
  - no M2 plateau under friction
  - regime alignment does not improve the continuation thesis

## Output Requirements
- Update this file in place.
- Use the `Agent Report` section for the latest run only.
- Include:
  - hypothesis card
  - chosen implementation path
  - experiments run
  - stage outcomes
  - key metrics
  - disposition
  - next step
  - explicit artifact paths

## Agent Report
### Run Timestamp
`2026-04-02T11:03:49-0500`

### Hypothesis Card
- idea: Trade IWM continuation after a 15-minute opening-range boundary cross.
- target_behavior: Opening move continuation after range break.
- expected_edge: Continuation quality improves when trade direction matches 5-minute regime.
- invalidation_signals:
  - no stable M1 edge
  - no M2 plateau under friction
  - regime alignment does not improve continuation quality

### Chosen Implementation Path
- primary mapping attempt: `Opening Drive Classifier` (nearest opening-range continuation template).
- secondary mapping attempt: `Market Impulse (Cross & Reclaim)` (has explicit `regime_timeframe`, including `5m`).
- outcome: No honest single-strategy mapping exists in current surface for `opening-range cross` + `5m regime alignment` together.

### Experiments Run
- M1 feasibility gate (surface-fit audit only; no PnL sweep executed).
- Checked strategy parameter surfaces and tracked scope from registry-backed catalog entries.

### Stage Outcomes
- M1 (discovery): `blocked`
  - reason: required trigger/filter conjunction cannot be expressed with one existing strategy family under current repo surface.
  - deterministic gate progression: stopped at M1 by design; M2-M5 not eligible.

### Key Metrics
- opening-range window candidates in nearest opening-range strategy: `opening_window_minutes=[20, 25, 30]` (missing required `15`).
- regime filter in nearest opening-range strategy: `absent` (no `regime_timeframe` knob).
- regime filter availability in nearest regime-aware strategy: `regime_timeframe=['5m','15m','30m','1h']`, but opening-range-cross trigger is absent.

### Surface Outcome
- surface classification: `blocked by current Mala research surface`.
- capability-gap note:
  - Need a strategy surface that combines:
    - opening-range boundary crossing trigger
    - explicit regime-timeframe alignment filter (`5m`)
    - continuation-only mode
    - configurable opening window including `15m`

### Disposition
- decision: `blocked`
- rationale: The hypothesis cannot be tested honestly within current repo strategy surfaces and declared parameter constraints without inventing unsupported logic during evaluation.

### Exact Repo Changes Required Before Rerun
1. Extend `Opening Drive Classifier` (or add a dedicated opening-range continuation strategy) to include an explicit regime filter parameter (e.g., `regime_timeframe`/`regime_col`) and enforce directional confluence in signal generation.
2. Add `15` to the opening-range window domain (or add a dedicated `opening_range_minutes` parameter with legal value `15`) so the trigger matches the hypothesis.
3. Register the new/extended surface in research and, if execution-facing, reflect capability status in `docs/strategy_surface_proposed.yaml`.

### Smallest Next Change
- Implement one minimal strategy-surface increment: add `regime_timeframe` confluence gating plus `15m` opening-window support to `Opening Drive Classifier`, then run a bounded M1 parameter sweep on `IWM` only.

### Explicit Artifact Paths
- hypothesis file updated: `/Users/suman/kg_env/projects/mala_v1/research/hypotheses/iwm-opening-range-regime-continuation.md`
- strategy surface reference: `/Users/suman/kg_env/projects/mala_v1/docs/strategy_surface.yaml`
- proposed-surface reference: `/Users/suman/kg_env/projects/mala_v1/docs/strategy_surface_proposed.yaml`
- nearest-template implementation reviewed: `/Users/suman/kg_env/projects/mala_v1/src/strategy/opening_drive_classifier.py`
- regime-template implementation reviewed: `/Users/suman/kg_env/projects/mala_v1/src/strategy/market_impulse.py`
