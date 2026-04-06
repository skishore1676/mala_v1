# IWM Opening Range Regime Continuation

## Status
- state: `kill`
- allowed_values: `pending | running | retune | blocked | completed`
- owner: `codex`
- created_at: `2026-04-02`
- last_run_at: `2026-04-06T02:06:25+0000`
- next_action: `hypothesis killed — insufficient edge across all gates`

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
`2026-04-05T210611`

### Stages Executed
`M1`

### Stage Outcomes
- M1 (discovery/retune): `no positive configs`
- M2 promoted: `0`
- M4 holdout promoted: `0`
- M5 execution promoted: `0`

### Notes
- M1 gate failed: no positive configs found.
- No positive exp_r found in any config. Recommend kill.

### Disposition
- decision: `kill`

### Artifact Directory
`/sessions/gracious-eloquent-clarke/mnt/mala_v1/data/results/hypothesis_runs/iwm-opening-range-regime-continuation/2026-04-05T210611`
