# First Agentic Run

Date: 2026-03-23

## Thesis

Run one full M1-M5 research cycle as if the repo were already an autonomous quant research agent:

- Strategy family: `Elastic Band Reversion`
- Symbols: `AAPL, META, TSLA, QQQ, IWM`
- Data window: `2024-01-02` to `2026-02-28`
- Walk-forward: `6m train / 3m test`
- Ratio search: `1.0, 1.25, 1.5, 2.0`
- M1 friction: `8 bps`
- M2 friction grid: `5, 8, 12 bps`
- M2 gates:
  - `min_oos_windows >= 6`
  - `min_oos_signals >= 150`
  - `min_pct_positive_oos_windows >= 0.60`
  - `min_avg_test_exp_r >= 0.0`
- M4 holdout gates:
  - `min_calibration_signals >= 50`
  - `min_holdout_signals >= 20`
  - holdout `exp_r >= 0` at all cost points

Artifacts:

- `data/results/agentic_first_run_2026-03-23_21-23-22_elastic_band/`

## M1 Discovery

Top candidate per symbol after full 36-config sweep:

| ticker | dir | params | avg OOS exp(R) | pct positive windows | OOS signals |
|---|---|---|---:|---:|---:|
| AAPL | long | `z=3.0, w=360, dm=false` | `+0.2000` | `0.833` | `779` |
| META | short | `z=1.75, w=240, dm=true` | `+0.1381` | `1.000` | `774` |
| TSLA | short | `z=2.5, w=360, dm=false` | `+0.1777` | `0.833` | `2334` |
| QQQ | short | `z=3.0, w=240, dm=true` | `+0.1733` | `0.833` | `149` |
| IWM | short | `z=2.0, w=240, dm=true` | `+0.1135` | `0.833` | `571` |

Takeaway:

- The edge remained highly ticker-specific.
- Short-side mean reversion dominated.
- `directional_mass` was still helpful on `META/QQQ/IWM`, but the best `AAPL` and `TSLA` variants flipped to `dm=false`, which is a drift warning versus older repo memory.

## M2 Convergence

Passed convergence across `5/8/12 bps`:

| ticker | dir | params | decision |
|---|---|---|---|
| TSLA | short | `z=2.5, w=360, dm=false` | `promote_to_holdout` |
| AAPL | long | `z=3.0, w=360, dm=false` | `promote_to_holdout` |
| META | short | `z=1.75, w=240, dm=true` | `promote_to_holdout` |
| IWM | short | `z=2.0, w=240, dm=true` | `promote_to_holdout` |

Borderline:

- `QQQ short z=3.0, w=240, dm=true` stayed positive but failed stability/signal gates and landed in `candidate_needs_more_stability`.

## M3 Walk-Forward

Window-by-window OOS detail was generated for the 4 M2 survivors.

Headline stats:

| ticker | dir | params | windows | test signals | avg test exp(R) | pct positive | median ratio |
|---|---|---|---:|---:|---:|---:|---:|
| AAPL | long | `z=3.0, w=360, dm=false` | `6` | `779` | `+0.2001` | `0.833` | `2.0` |
| TSLA | short | `z=2.5, w=360, dm=false` | `6` | `2334` | `+0.1777` | `0.833` | `2.0` |
| META | short | `z=1.75, w=240, dm=true` | `6` | `774` | `+0.1381` | `1.000` | `2.0` |
| IWM | short | `z=2.0, w=240, dm=true` | `6` | `571` | `+0.1135` | `0.833` | `2.0` |

Takeaway:

- `2.0R` remained the dominant selected ratio across all surviving candidates.
- M3 looked healthy for four names, but M4 was where most of the optimism died.

## M4 Holdout

Untouched holdout result:

| ticker | dir | params | mean holdout exp(R) | status |
|---|---|---|---:|---|
| IWM | short | `z=2.0, w=240, dm=true` | `+0.1321` | pass |
| META | short | `z=1.75, w=240, dm=true` | `+0.0261` | fail at 12 bps |
| TSLA | short | `z=2.5, w=360, dm=false` | `-0.1081` | fail |
| AAPL | long | `z=3.0, w=360, dm=false` | `-0.2226` | fail |

Takeaway:

- Only `IWM short` survived all holdout friction points.
- `META short` was close, but it lost robustness at the widest cost assumption.
- `TSLA short` and `AAPL long` did not survive untouched holdout despite strong M1/M3 optics.

## M5 Execution Mapping

Sole survivor:

- `IWM short`
- Strategy: `Elastic Band z=2.0/w=240+dm`
- Selected ratio: `2.0`
- Holdout trades: `94`
- Holdout win rate: `0.4255`
- Base exp(R): `+0.1966`
- Monte Carlo:
  - `mc_exp_r_p50 = -0.0212`
  - `mc_prob_positive_exp = 0.4433`
  - `mc_max_dd_p50 = 15.72R`

Interpretation:

- The raw holdout edge survives.
- The execution-stressed distribution is not yet strong enough for live confidence.
- This is a research survivor, not a deployment-ready strategy.

## Good

- The staged research stack is now capable of a real end-to-end run on local cached data.
- Multi-symbol discovery surfaced a coherent shortlist instead of random one-off winners.
- One candidate (`IWM short`) survived all deterministic gates through execution mapping.
- New runner added: `scripts/run_agentic_first_pass.py`

## Bad

- The generic research tool surface still does not chain cleanly:
  - `parameter_sweep()` aggregate output and `convergence_grid()` expect different schemas.
- Repo memory is partially stale relative to this run:
  - prior notes favored `TSLA` and `META`; this run only carried `IWM` fully through M5.
- Summary/artifact handling was brittle enough that null rows and missing carried params could derail reporting.

## Ugly

- M4/M5 originally risked reconstructing the wrong strategy variant because promoted params were not reliably carried through late-stage helpers.
- Holdout detail originally dropped candidate metadata, which makes downstream agent reasoning harder and encourages string-parsing hacks.
- Strategy display names are still doing too much work as implicit identifiers; this will break on strategies whose labels do not encode params.
- There is still no single canonical experiment manifest tying:
  - assumptions
  - selected candidates
  - stage outputs
  - disposition
  - state updates

## Fixes Made During This Run

- Preserved candidate identity/params across M2-M5 stage helpers.
- Added carried candidate metadata to holdout and execution outputs.
- Added regression coverage for candidate identity preservation and strategy reconstruction.
- Added `scripts/run_agentic_first_pass.py` so the first-pass workflow is reproducible.

## Next Fixes For Full Agentic Autonomy

1. Unify M1/M2 tool schemas so `parameter_sweep -> convergence_grid` is a direct handoff.
2. Introduce a stable candidate ID / manifest instead of overloading strategy display names.
3. Auto-write a compact experiment manifest plus disposition file per run.
4. Add automatic `research_state.yaml` update suggestions rather than leaving promotion tracking manual.
5. Generalize the first-pass runner so strategy family and parameter grid come from registry metadata, not hardcoded Elastic logic.
6. Add a “why failed” explainer for gate failures so the agent can retune intentionally instead of guessing.
