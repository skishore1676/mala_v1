# Bionic Loop Checklist

This is the one canonical operator checklist for the full Bionic loop.

Use this doc for the day-to-day flow:

- `Mala` researches and maintains the armory
- `Bionic_Bias_v1` authorizes what is allowed to fire today
- the translator/session compiler selects exact playbooks into one session file
- `Bhiksha` executes only that session file
- post-close review feeds evidence back into `Mala`

## Roles

- `Mala`
  - nightly M1/M2 scout
  - human-reviewed M3/M4/M5 follow-ups
  - exit optimization
  - master playbook catalog maintenance
- `Master_Playbook_Catalog`
  - durable armory of validated playbooks
  - `active`, `stale`, `retired`
- `Bionic_Bias_v1`
  - daily authorization and risk-governor sheet
  - not a parameter sheet
  - not the armory
- `entry_v1`
  - manual override rail
  - compiles into `manual_trigger` deployments
- `Bhiksha`
  - loads one `active_session.json`
  - executes entries, exits, and risk
  - writes post-close observations

## Nightly Forge

Run the nightly research loop:

```bash
cd /Users/suman/kg_env/projects/mala_v1
./.venv/bin/python scripts/run_nightly_regime_matrix.py
```

Expectation:

- most nights stop at `M2`
- queue/workbook/charts are the primary outputs
- a night with zero top-level deployment artifacts can still be healthy

Review and update:

- [m2_human_review_queue.csv](/Users/suman/kg_env/projects/mala_v1/data/results/nightly_regime_matrix/research_control/m2_human_review_queue.csv)
- [human_review_workbook.xlsx](/Users/suman/kg_env/projects/mala_v1/data/results/nightly_regime_matrix/research_control/review_bundle/human_review_workbook.xlsx)
- [charts](/Users/suman/kg_env/projects/mala_v1/data/results/nightly_regime_matrix/research_control/charts)

Human actions:

- `promote_to_m3`
- `retune`
- `expand_symbols`
- `kill`

Expectation:

- only queue-approved follow-ups should spend deep validation budget
- only `M5` survivors with optimized thesis exits become `bionic_ready`

## Armory

The durable armory lives in:

- [master_playbook_catalog.json](/Users/suman/kg_env/projects/mala_v1/data/playbooks/master_playbook_catalog.json)
- [master_playbook_catalog.csv](/Users/suman/kg_env/projects/mala_v1/data/playbooks/master_playbook_catalog.csv)

Operator mirror sheet tab:

- `Master_Playbook_Catalog`

Expectations:

- `M5` survivors should be preserved in the master catalog
- nightly research adds new weapons and refreshes old ones
- `stale` is automatic after 60 days without revalidation
- `retired` is a human override from the sheet
- the translator should use the master catalog, not a random nightly bundle

## Pre-Open Authorization

Open `Bionic_Bias_v1` and authorize what is allowed to fire today.

Example row:

- `SPY | Bearish | Trend_Continuation`

Meaning:

- you are not choosing parameters
- you are authorizing a symbol/thesis lane for today
- if it is not authorized here, the translator should not arm it

If needed, add manual override rows in `entry_v1`.

## Pre-Open Compile And Launch

Canonical command:

```bash
cd /Users/suman/kg_env/projects/bhiksha
PYTHONPATH=src .venv/bin/python -m bhiksha.tools.bionic_session prepare
```

What it does:

- calls Mala's session compiler
- reads `Bionic_Bias_v1`
- reads `entry_v1`
- reads the master catalog
- selects the best matching `bionic_ready` playbook per authorized symbol/context
- applies manual overrides by symbol
- writes and publishes one [active_session.json](/Users/suman/kg_env/projects/bhiksha/artifacts/playbook/active_session.json)
- runs provider health
- runs a warm-start check

Expectation:

- `PLAYBOOK_DEPLOYMENT_COUNT > 0` only when today’s authorized rows match active `bionic_ready` playbooks
- if the count is `0`, this is usually a bias/catalog mismatch, not a broken compiler

If you want prepare to launch immediately:

```bash
PYTHONPATH=src .venv/bin/python -m bhiksha.tools.bionic_session prepare --start-live
```

Or start live separately:

```bash
PYTHONPATH=src .venv/bin/python -m bhiksha.tools.bionic_session run --live
```

## Live Session Expectations

Bhiksha should:

- ignore `config/deployments/` in session-payload mode
- load only the deployments in `active_session.json`
- warm 1-minute bars
- wait on newly closed bars
- evaluate only armed deployments
- execute only if entry and risk conditions pass

Bhiksha should not:

- search for strategies
- load the full armory
- arm conflicting strategies on the same symbol

## Post-Close Review

Canonical command:

```bash
cd /Users/suman/kg_env/projects/bhiksha
PYTHONPATH=src .venv/bin/python -m bhiksha.tools.bionic_session review
```

What it writes:

- local feedback bundle under `artifacts/playbook/session_feedback/<session_id>/`
- mirrored feedback bundle in `Mala` under `data/live_feedback/<session_id>/`

Expectation:

- compact summary for the session
- per-deployment observation packets
- clear separation of `mala_playbook` vs `operator_manual`

## Common Checks

If prepare yields zero playbook deployments:

- check that `Bionic_Bias_v1` matches an `active` and `bionic_ready` playbook
- check that the playbook is not `stale` or `retired`
- check that the strategy is Bhiksha-supported

If a playbook exists in the master catalog but does not arm:

- confirm `bionic_ready = true`
- confirm it has optimized `thesis_exit_*`
- confirm today’s bias row matches its bias template

If Bhiksha loads the wrong thing:

- inspect [active_session.json](/Users/suman/kg_env/projects/bhiksha/artifacts/playbook/active_session.json)
- remember that this is the sole live authority in Bionic mode

## Mental Model

- `Master Catalog` = armory
- `Bionic_Bias_v1` = today’s authorization sheet
- `Translator / SessionCompiler` = loadout selector
- `Bhiksha` = sniper

That is the intended end-to-end loop.

