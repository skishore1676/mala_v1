# Nightly Regime Matrix

The nightly regime matrix is Mala's fixed-watchlist handoff into Bhiksha.

It runs the current research families across a configured symbol universe, then
publishes schema-v2 loop artifacts that Bhiksha can route from the next morning.

## Current Scope

- Watchlist source: `config/nightly_regime_matrix.yaml`
- Strategy families:
  - `market_impulse`
  - `jerk_pivot_momentum`
  - `elastic_band_reversion`
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
python scripts/run_nightly_regime_matrix.py
```

Optional:

```bash
python scripts/run_nightly_regime_matrix.py --config config/nightly_regime_matrix.yaml
python scripts/run_nightly_regime_matrix.py --bundle-dir /tmp/nightly_matrix_bundle
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
entrypoint for the Mala -> Bhiksha loop.
