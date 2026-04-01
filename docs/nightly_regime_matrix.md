# Nightly Regime Matrix

The nightly regime matrix is Mala's operator-facing nightly research loop.

It now does two jobs in one pass:

1. Run the broad nightly M1/M2 scout across the default strategy families and Tier 1 watchlist.
2. Maintain the durable human review queue, refresh local charts, and execute approved follow-up actions under hard nightly caps.

The broad scout now stops at `M2` by default. Deep `M3/M4/M5` work is reserved for queue-approved follow-up actions.

This is the intended architecture:

- nightly scout finds and organizes candidates
- the human review queue decides what deserves expensive validation
- only validated follow-up runs produce deployable playbooks

## Current Scope

- Watchlist source: `config/nightly_regime_matrix.yaml`
- Tier 1 defaults:
  - `SPY`
  - `QQQ`
  - `IWM`
  - `NVDA`
  - `TSLA`
  - `AAPL`
- Tier 2 defaults:
  - `META`
  - `MSFT`
  - `AMZN`
  - `AMD`
  - `SMH`
  - `XLF`
- Strategy families:
  - `market_impulse`
  - `jerk_pivot_momentum`
  - `elastic_band_reversion`
- Follow-up budgets:
  - `max_new_m2_rows_per_night = 10`
  - `max_retune_tasks_per_night = 4`
  - `max_symbol_expansion_tasks_per_night = 3`
  - `max_m3_promotions_per_night = 3`
  - `max_total_followup_tasks_per_night = 8`
- Operator contexts emitted for every watched symbol:
  - `bullish_trend_intraday`
  - `bullish_mean_reversion_intraday`
  - `bearish_trend_intraday`
  - `bearish_mean_reversion_intraday`

Trend contexts are populated from supported families.

Mean-reversion contexts are currently researched from `elastic_band_reversion`
and stay manual-lane for Bhiksha until execution capability expands.

## Run It

```bash
./.venv/bin/python scripts/run_nightly_regime_matrix.py
```

Optional:

```bash
./.venv/bin/python scripts/run_nightly_regime_matrix.py --config config/nightly_regime_matrix.yaml
./.venv/bin/python scripts/run_nightly_regime_matrix.py --bundle-dir /tmp/nightly_matrix_bundle
```

## Outputs

Each run writes a dated bundle under:

```text
data/results/nightly_regime_matrix/<YYYY-MM-DD>/nightly_regime_matrix/<HH-MM-SS>/
```

Important files:

- `deployment_candidates.json`
- `playbook_catalog.json`
- `nightly_matrix_manifest.json`

Because the broad scout stops at `M2`, the top-level deployment/export JSONs may be empty on nights where no queued follow-up run emits fresh `M5`-level execution artifacts. That is expected; the queue/workbook/charts surface is the primary operating output for the scout.

The nightly summary should be read literally. On a normal scout-only night it may say:

```text
scout_only_run = true
deployment_candidates_generated = 0
reason = no M3-M5 follow-up executed
```

That means the pipeline is healthy, the scout stopped where it was supposed to stop, and deployable playbooks were not expected from that pass.

The nightly loop also writes a stable research-control area under:

```text
data/results/nightly_regime_matrix/research_control/
```

Important review artifacts there:

- `m2_human_review_queue.csv`: canonical human-editable queue
- `m2_human_review_history.csv`: rolling nightly observation history
- `review_bundle/human_review_workbook.xlsx`
- `review_bundle/m2_review.csv`
- `review_bundle/recent_history.csv`
- `review_bundle/execution_queue.csv`
- `review_bundle/full_survivors.csv`
- `review_bundle/charts_index.csv`
- `charts/<candidate_key>.html`

Even on an all-clear night with no M2 survivors, the queue/history/workbook/review-bundle files should still be written so the operator sees an explicit empty review surface instead of a missing control plane.

## Queue Semantics

Each queue row represents one stable candidate identity built from:

- strategy family
- ticker
- direction
- normalized config signature
- research slice identity

The queue is the source of truth for human actions and review state. It includes:

- current M1/M2 metrics and gate outcomes
- `research_slice_id`
- `chart_link`
- `human_decision`
- `human_notes`
- `priority`
- `queue_status`
- `latest_stage_reached`
- `latest_stage_decision`
- `passes_m1` through `passes_m5`
- `is_full_m1_m5_survivor`
- `last_seen_run_date`
- `last_action_run_date`

Queue states:

- `NEW`
- `PENDING`
- `EXECUTING`
- `EXECUTED`
- `KILLED`
- `STALE`
- `ERROR`

Terminal-state rule:

- If a candidate is already `EXECUTED` or `KILLED`, a later nightly M2 sighting only refreshes observational fields such as metrics, `last_seen_run_date`, and `chart_link`.
- The nightly scout must not silently reopen that row back to `PENDING`.
- If the human edits the terminal row and changes `queue_status` or `human_decision`, the row becomes actionable again on a later nightly run.

## Follow-Up Actions

Approved rows are consumed nightly from the queue by:

- highest `priority`
- then newest human-edited pending rows
- then newest supporting evidence

Supported actions:

- `promote_to_m3`: run the narrow M3/M4/M5 path and mark the row `EXECUTED` when the task run completes
- `retune`: run a focused local M1/M2 neighborhood search around the current config
- `expand_symbols`: run the same family/config neighborhood on Tier 2 symbols
- `kill`: mark the row `KILLED`

Rows beyond the nightly caps remain `PENDING`.

## Charts

Every nightly M2 survivor gets a self-contained local Plotly candlestick HTML chart.

- Source data: raw/enriched OHLCV bars, not just compact trade logs
- Window: last 3 trading days of recent available context
- Overlays: long/short signal markers
- Metadata: candidate family, symbol, direction, and config in the chart title/subtitle
- Repeats: the same candidate key refreshes the same chart file instead of creating duplicates

## Review Surface

The workbook/CSV bundle is a projection of the queue, not a second state store.

It includes these sheets/views:

- `M2 Review`
- `Recent History`
- `Execution Queue`
- `Full Survivors`
- `Charts Index`

Full M1-M5 survivors are explicit via:

- `passes_m1`
- `passes_m2`
- `passes_m3`
- `passes_m4`
- `passes_m5`
- `is_full_m1_m5_survivor`

## Contract

Both JSON handoff files are schema-v2 contracts.

Top-level required fields:

- `contract_name`
- `schema_version`
- `generated_at`
- `run_dirs`
- `watchlist`
- `enabled_strategy_families`

Contract names:

- `deployment_candidates`
- `playbook_catalog`

`playbook_catalog.json` is a full matrix, not a sparse survivor list.

For every `symbol|bias_template|intraday` context it emits:

- `supported_candidates`
- `proposed_candidates`
- `coverage_status`
- `covered_by_strategy_families`

Current `coverage_status` values:

- `researched_with_survivors`
- `researched_no_survivors`
- `not_covered_by_enabled_family`

## Family Runners

The nightly matrix currently delegates to:

- `scripts/run_agentic_market_impulse_pass.py`
- `scripts/run_agentic_jerk_pivot_pass.py`
- `scripts/run_agentic_first_pass.py`

Each family runner still writes its own dated run directory and can be executed
independently, but the nightly matrix is now the canonical operator-facing
entrypoint for the broad scout plus human-review follow-up loop.

When invoked by the nightly matrix, family runners are launched with `--max-stage M2`.
When invoked directly, they still default to the full `M1` through `M5` path.
