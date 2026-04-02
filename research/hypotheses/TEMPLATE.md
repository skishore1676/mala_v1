# Hypothesis Title

## Status
- state: `pending`
- allowed_values: `pending | running | retune | blocked | completed`
- owner: `codex`
- created_at: `YYYY-MM-DD`
- last_run_at: ``
- next_action: ``

## Repo Change Policy
- repo_change_policy: `propose`
- allowed_values: `propose | implement_research_surface`
- branch_rule: `if repo changes are allowed, create a fresh codex/ branch first and do not merge automatically`

## Hypothesis
Write the plain-English market thesis here.

## Constraints
- symbol_scope:
- preferred_strategy_family:
- deterministic_gate_limit: `M1-M5`
- max_stage_this_run:
- allowed_parameter_budget:
- date_range:
- notes:

## Mapping Hints
- nearest_existing_strategy:
- trigger_event:
- directional_thesis:
- timeframe_alignment:
- invalidation_conditions:

## Output Requirements
- Update this file in place.
- Use the `Agent Report` section for the latest run only unless the user explicitly asks for history.
- Include explicit artifact paths.
- End with one of: `promote`, `retune`, `gather_more_evidence`, `kill`.
- Status rules:
  - use `pending` for a fresh hypothesis
  - use `running` only while an agent is actively working it
  - use `retune` when another bounded pass is the correct next step
  - use `blocked` when the repo cannot test the idea honestly without code or surface changes
  - use `completed` when the current cycle is done and should not be auto-picked again
- Repo-change rules:
  - `propose`: no repo edits; write the minimal required changes and stop
  - `implement_research_surface`: agent may implement the smallest research-surface change needed to continue, on a fresh `codex/` branch only

## Agent Report
Pending.
