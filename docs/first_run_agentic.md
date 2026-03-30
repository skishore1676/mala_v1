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

## 2026-03-30 Follow-on Run: Kinematic Ladder

Thesis:

- Strategy family: `Kinematic Ladder`
- Symbols: `TSLA, NVDA, AMD, QQQ, SPY`
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

- `data/results/agentic_runs/2026-03-30/kinematic_ladder/13-54-41/`

## M1 Discovery

Top candidate per surviving symbol after the full 72-config sweep:

| ticker | dir | params | avg OOS exp(R) | pct positive windows | OOS signals |
|---|---|---|---:|---:|---:|
| TSLA | short | `rw=45, aw=20, vol=false` | `+0.2157` | `1.000` | `205` |
| AMD | short | `rw=30, aw=20, vol=false` | `+0.0641` | `0.667` | `209` |
| QQQ | combined | `rw=45, aw=8, vol=false` | `+0.0571` | `0.667` | `289` |

Takeaway:

- `TSLA short` was the clear discovery winner.
- `AMD short` and `QQQ combined` were positive enough to justify a robustness check.
- `NVDA` and `SPY` produced no positive top candidate under the declared search space.

## M2 Convergence

Passed convergence across `5/8/12 bps`:

| ticker | dir | params | decision |
|---|---|---|---|
| TSLA | short | `rw=45, aw=20, vol=false` | `promote_to_holdout` |
| AMD | short | `rw=30, aw=20, vol=false` | `promote_to_holdout` |
| QQQ | combined | `rw=45, aw=8, vol=false` | `promote_to_holdout` |

Takeaway:

- All three shortlist names survived the deterministic M2 gates.
- This was a cleaner convergence result than the tracked repo memory implied for Kinematic Ladder.

## M3 Walk-Forward

Headline stats:

| ticker | dir | params | windows | test signals | avg test exp(R) | pct positive | median ratio |
|---|---|---|---:|---:|---:|---:|---:|
| TSLA | short | `rw=45, aw=20, vol=false` | `6` | `205` | `+0.2157` | `1.000` | `1.75` |
| AMD | short | `rw=30, aw=20, vol=false` | `6` | `209` | `+0.0641` | `0.667` | `1.5` |
| QQQ | combined | `rw=45, aw=8, vol=false` | `6` | `289` | `+0.0571` | `0.667` | `2.0` |

Takeaway:

- `TSLA short` stayed strongest and most stable.
- `AMD` and `QQQ` were still alive, but their edge quality was much thinner heading into holdout.

## M4 Holdout

Untouched holdout result:

| ticker | dir | params | mean holdout exp(R) | status |
|---|---|---|---:|---|
| TSLA | short | `rw=45, aw=20, vol=false` | `+0.0558` | pass |
| AMD | short | `rw=30, aw=20, vol=false` | `-0.0187` | fail |
| QQQ | combined | `rw=45, aw=8, vol=false` | `-0.2852` | fail |

Cost-point detail:

- `TSLA short`: `+0.0906` at `5 bps`, `+0.0593` at `8 bps`, `+0.0175` at `12 bps`
- `AMD short`: positive at `5 bps`, negative at `8/12 bps`
- `QQQ combined`: negative at every holdout friction point

Takeaway:

- Only `TSLA short` survived untouched holdout.
- `AMD short` was plausible in discovery but lost robustness once friction widened.
- `QQQ combined` was a false positive from M1/M2 and failed decisively on quarantine data.

## M5 Execution Mapping

Sole survivor:

- `TSLA short`
- Strategy: `Kinematic Ladder rw=45/aw=20-vol`
- Selected ratio: `1.5`
- Holdout trades: `35`
- Holdout win rate: `0.4571`
- Base exp(R): `+0.0629`
- Monte Carlo:
  - `mc_exp_r_p50 = -0.1301`
  - `mc_prob_positive_exp = 0.2750`
  - `mc_max_dd_p50 = 9.52R`

Interpretation:

- The raw holdout edge survives the deterministic M4 gate.
- The execution-stressed distribution is weak and not live-ready.
- This is a research survivor, not a deployment survivor.

## Good

- The research journal + saved artifacts were good enough to resume the run from M1 after an M2 crash instead of redoing discovery.
- Multi-symbol discovery found a broader Kinematic Ladder shortlist than the repo memory suggested.
- One candidate (`TSLA short`) survived all deterministic gates through execution mapping.

## Bad

- Two autonomy blockers still required code fixes during the run:
  - tracked repo memory passed `reward_risk_ratio` into the Jerk-Pivot factory even though it is not a strategy constructor arg
  - M1 shortlist rows carried derived `m1_score`, which late-stage candidate reconstruction misread as a strategy param
- `parameter_sweep()` still burns budget on no-op cells:
  - when `use_volume_filter=false`, changing `volume_multiplier` does not change the strategy
- The M1 summary still reports the first built strategy label instead of a clearer campaign-level name.

## Ugly

- The first full Kinematic pass died right after M1 because a ranking-only field leaked into strategy reconstruction.
- There is still no canonical resume-aware runner for arbitrary tracked strategies; the journal made recovery possible, but the orchestration layer did not.
- M5 reports a completed mapping, but there is still no explicit repo-level notion of “execution survives stress” versus “mapping merely ran.”

## Fixes Made During This Run

- Taught the Jerk-Pivot factory path to ignore evaluation-only `reward_risk_ratio` from `research_state.yaml`.
- Taught candidate reconstruction helpers to ignore derived ranking fields like `m1_score`.
- Added regression coverage for both fixes.

## Additional Next Fixes

1. Deduplicate no-op parameter cells before M1 runs so the agent does not waste budget on equivalent configs.
2. Separate candidate params from ranking/reporting columns with an explicit schema instead of negative filtering.
3. Add a resumable stage runner that can restart from `M1` or `M2` artifacts without inline recovery code.
4. Promote the multi-stage strategy pass into a canonical orchestration command instead of bespoke one-shot scripts.
5. Add an explicit M5 stress-survival disposition so “mapped” and “deployable” are not conflated.

## 2026-03-30 Follow-on Run: Kinematic Ladder Plateau Study

Thesis:

- Strategy family: `Kinematic Ladder`
- Primary symbol: `TSLA`
- Transfer basket: `AMD, QQQ, NVDA, SPY`
- Goal: separate the `point optimum` from the `stable plateau`
- Data window: `2024-01-02` to `2026-02-28`
- Walk-forward: `6m train / 3m test`
- Ratio search: `1.0, 1.25, 1.5, 2.0`
- M1 friction: `8 bps`
- M2 friction grid: `5, 8, 12 bps`

Artifacts:

- `data/results/agentic_runs/2026-03-30/kinematic_ladder_plateau/14-19-21/`

## M1 Discovery

Important change:

- Requested config count: `72`
- Effective deduped config count: `45`
- Duplicate no-op cells removed: `27`

Selected TSLA short candidates:

| role | params | avg OOS exp(R) | plateau score |
|---|---|---:|---:|
| point optimum | `rw=45, aw=20, vol=false` | `+0.2157` | `200.1` |
| plateau | `rw=45, aw=12, vol=false` | `+0.1179` | `202.1` |
| plateau | `rw=20, aw=12, vol=false` | `+0.0838` | `202.1` |
| plateau | `rw=45, aw=8, vol=false` | `+0.1183` | `200.1` |
| plateau | `rw=30, aw=12, vol=false` | `+0.0971` | `191.8` |

Takeaway:

- The single best point was still `rw=45 / aw=20`.
- The broader stable neighborhood centered around `aw=12` rather than only the point winner.

## M2 Convergence

Convergence result on the five-candidate TSLA shortlist:

| params | decision |
|---|---|
| `rw=45, aw=20, vol=false` | pass |
| `rw=45, aw=12, vol=false` | pass |
| `rw=30, aw=12, vol=false` | pass |
| `rw=20, aw=12, vol=false` | pass |
| `rw=45, aw=8, vol=false` | needs more stability |

Takeaway:

- Four TSLA short variants survived M2.
- This confirmed that the Kinematic edge is a region, not just a single lucky cell.

## M4 Holdout

Untouched holdout result for the four plateau survivors:

| params | mean holdout exp(R) | status |
|---|---:|---|
| `rw=45, aw=20, vol=false` | `+0.0558` | pass |
| `rw=20, aw=12, vol=false` | `+0.0531` | pass |
| `rw=45, aw=12, vol=false` | `+0.0313` | pass |
| `rw=30, aw=12, vol=false` | `-0.0455` | fail |

Takeaway:

- Three different TSLA short variants survived holdout.
- `rw=20, aw=12` was especially important because it validated the plateau thesis directly.

## M5 Execution Mapping

Execution-stress result for the three holdout survivors:

| params | base exp(R) | mc prob positive exp | mc exp(R) p50 | mc max DD p50 |
|---|---:|---:|---:|---:|
| `rw=45, aw=20, vol=false` | `+0.0629` | `0.2750` | `-0.1301` | `9.52R` |
| `rw=20, aw=12, vol=false` | `+0.0407` | `0.2525` | `-0.1497` | `8.66R` |
| `rw=45, aw=12, vol=false` | `+0.0200` | `0.2500` | `-0.1742` | `7.99R` |

Interpretation:

- The plateau is real through M4.
- The entire plateau still fails the “acceptable after stress” production test.
- This is stronger evidence that the idea is viable, but the execution layer is still the bottleneck.

## Transfer Check

Cross-symbol transfer of the TSLA plateau variants:

- `AMD`: only weak positive spillover, best around `+0.0326`
- `QQQ`: one mild positive at `rw=30, aw=12`, otherwise weak/negative
- `NVDA`: broadly negative
- `SPY`: broadly negative

Takeaway:

- Kinematic remains primarily a `TSLA short` phenomenon.
- Plateau optimization improved our understanding of robustness, but it did not turn the strategy into a universal basket edge.
