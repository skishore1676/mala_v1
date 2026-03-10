# Status And Next Steps (2026-03-09)

## What We Built

### New Strategy Ideas Implemented

1. **Elastic Band Reversion**
- Mean-reversion around VPOC stretch + exhaustion flip.
- Upgraded to dynamic z-score stretch and directional mass gating.
- File: `src/strategy/elastic_band_reversion.py`

2. **Kinematic Ladder**
- Multi-timeframe-inspired regime/setup/trigger approach using velocity, acceleration, jerk.
- File: `src/strategy/kinematic_ladder.py`

3. **Compression Expansion Breakout**
- Trade breakouts after volatility compression with trend/volume filters.
- File: `src/strategy/compression_breakout.py`

4. **Regime Router (Kinematic + Compression)**
- Routes entries by regime (trend vs compression) into specialized sub-strategies.
- File: `src/strategy/regime_router.py`

### New Evaluation Tooling

1. **Measurement sensitivity + Monte Carlo**
- File: `scripts/run_measurement_sensitivity.py`
- Purpose: compare `1.0, 1.25, 1.5, 2.0` reward:risk thresholds with expectancy robustness.

2. **Integrated novel strategy runner**
- File: `scripts/run_novel_ideas.py`
- Purpose: run all novel strategies and emit both summary + robustness outputs.

3. **Walk-forward out-of-sample validator**
- File: `scripts/run_walk_forward_novel.py`
- Purpose: choose ratio in train window, evaluate in next OOS window.

4. **Convergence pipeline (multi-cost gate runner)**
- File: `scripts/run_convergence_pipeline.py`
- Purpose: run walk-forward at multiple friction assumptions and promote only candidates that pass strict robustness gates.

### Data-Time Alignment Fix

- Added timezone utilities (`src/time_utils.py`) to convert stored UTC timestamps to ET.
- Updated strategy/session filters and walk-forward/date boundaries to use ET session time.
- This prevents accidental timezone drift in open/close filters.

### Physics Upgrade: Directional Mass

- Added directional mass columns in Newton:
  - `internal_strength`
  - `directional_mass`
  - `directional_mass_ma_20`
- Purpose: replace scalar volume participation with directional participation signal.

## Measurement Logic (Current)

For each strategy group (ticker + direction):

- Confidence at ratio `r`: `p = P(MFE >= r * MAE)`
- Friction-adjusted expectancy in R units:
  - `E[R] = p * r - (1 - p) - cost_r`
- Break-even confidence:
  - `p_be = (1 + cost_r) / (1 + r)`

Monte Carlo robustness uses bootstrap/binomial sampling of confidence to estimate:
- `P(E[R] > 0)`

## Findings So Far

### In-Sample / Full-Period Robustness

- **Best overall candidate:** `Elastic Band Reversion`
  - strongest at `1.5:1` and `2.0:1`.
  - high sample size and high `P(E[R] > 0)` across SPY/QQQ/IWM.
- `1.0:1` is not consistently positive once friction is applied.
- `Kinematic Ladder` is inconsistent and mostly weak except small pockets.
- `Compression Breakout` is mostly weak in robustness.

### Walk-Forward OOS (6m train / 3m test)

- **Most credible OOS candidate currently:** `SPY + Elastic Band Reversion` (especially short side).
- `QQQ + Compression Breakout (short)` showed slight positive OOS average, but still needs stronger confirmation.
- Most other combinations were negative or unstable out-of-sample.

### Expanded Universe + Longer Window (2024-01-01 to 2026-02-28)

- Universe expanded to: `SPY QQQ IWM NVDA TSLA META AMD PLTR AAPL`.
- This improved discovery of ticker-specific edge pockets vs index-only testing.
- Most credible OOS leaders now include:
  - `AAPL + Elastic Band (combined/long)`
  - `PLTR + Elastic Band (long)`
  - `TSLA` short-side variants (Elastic/Compression/Router)
- This supports a **ticker-specialized deployment** view rather than one universal parameter set.

See details: `docs/EXPANDED_UNIVERSE_RESULTS_2026-03-08.md`.
Convergence execution details: `docs/CONVERGENCE_PLAN_2026-03-08.md`.
Holdout confirmation details: `docs/HOLDOUT_VALIDATION_2026-03-08.md`.
Post-upgrade directional mass evaluation: `docs/POST_UPGRADE_DIRECTIONAL_MASS_EVAL_2026-03-08.md`.

### Post-Upgrade Reality Check & P1 Convergence (2026-03-09)

- Over 2026-03-08 and 2026-03-09, we successfully ran P1 ablation and a full convergence pipeline using relative friction (cost-bps) and a strict 0.60 OOS win-rate gate.
- **Directional Mass Filter:** Ablation proved this is highly protective for `Elastic Band Reversion`. Removing it floods the engine with noise and tanks E[R].
- **Volume Filter on Kinematic Ladder:** Ablation proved the volume gate was actually blocking the edge. Removing it revealed a localized `+0.22 E[R]` edge on `TSLA short`.
- **Top 10 Promoted Candidates (Untouched Holdout Ready):**
  - All 10 are variants of **Elastic Band Reversion**.
  - `META short` and `TSLA short` emerged as the most robust underlyings, both producing 100% win-rates across 6 out-of-sample walk-forward windows with `E[R] > +0.13`.
  - `Compression Breakout` and `Regime Router` failed to produce any edge and are marked `dead` in the new `research_state.yaml` registry.

See details: `docs/P1_CONVERGENCE_RESULTS_2026-03-09.md`.

## Practical Interpretation For Trading

- We officially have **one highly promising directional entry engine** (`Elastic Band Reversion`) with proven ticker-specialized parameters.
- `2:1` is useful and currently appears more robust than `1:1` for these strategies under relative friction assumptions (8 bps).

## Recommended Next Steps

1. **Narrow focus:**
- Optimize only upgraded `Elastic Band Reversion` with:
  - `z_score_threshold`
  - `z_score_window`
  - optional directional-mass magnitude filter

2. **Structured sweep:**
- Sweep new parameters above across short-side candidate tickers first (`IWM/QQQ/TSLA/META`).
- Keep one untouched OOS segment as a final holdout.

3. **Execution bridge to options:**
- Convert directional signal to option structure rules (DTE, delta band, spread width).
- Add conservative slippage/theta assumptions to avoid overstating edge.

4. **Promotion criteria (for live trial):**
- Positive OOS expectancy in multiple windows.
- Acceptable drawdown proxy.
- Stable edge under slightly worse friction assumptions.
